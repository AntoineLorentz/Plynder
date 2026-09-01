use cozy_chess::{Board, Color, GameStatus};

use crate::utils::{self, TokenId, check_insufficient_material};
use crate::TerminalTokens;

use ahash::AHashMap;


use tch::{Kind, Tensor, Device};

use crate::writer::send_sequence_data;
use crate::repetition_tracker::RepetitionTracker;


pub struct SequenceStatesPool {
    sequences: AHashMap<String, Board>,
    allowed_tokens: AHashMap<String, Vec<Vec<TokenId>>>,
    logprobs: AHashMap<String, Vec<f32>>,
    repetition_trackers: AHashMap<String, RepetitionTracker>,
    sequence_batch_ids: Vec<String>,
    terminal_tokens: TerminalTokens,
    device: Device,
    mask_pinned: Tensor,
    sender: Option<zmq::Socket>,
    global_engine_id: i32,
    send_sequence_data: bool,
}

impl SequenceStatesPool {
    pub fn new(vocab_size: i64, device: String, terminal_tokens: TerminalTokens, rollout_address: Option<String>, global_engine_id: i32) -> Self {
        tch::set_num_threads(1);
        let device = utils::parse_device(&device);

        let sender = rollout_address.as_ref().map(|address| {
            let ctx = zmq::Context::new();
            let sender = ctx.socket(zmq::PUSH).unwrap();
            sender.connect(address).unwrap();
            sender
        });

        let mut mask_pinned = Tensor::full(
                [1, vocab_size],
                0,
                (Kind::Bool, Device::Cpu),
            );

        if device != Device::Cpu {
            mask_pinned = mask_pinned.internal_pin_memory(device);
        }

        SequenceStatesPool {
            sequences: AHashMap::new(),
            allowed_tokens: AHashMap::new(),
            logprobs: AHashMap::new(),
            repetition_trackers: AHashMap::new(),
            sequence_batch_ids: Vec::new(),
            terminal_tokens,
            device,
            mask_pinned,
            sender,
            global_engine_id,
            send_sequence_data: rollout_address != None,
        }
    }

    pub fn add_sequence(&mut self, sequence_id: String, token_ids: &Vec<TokenId>) {
        let mut board = Board::default();
        let mut repetition_tracker = RepetitionTracker::new(Some(token_ids.len()));

        for &token_id in token_ids.iter() {
            if let Some(m) = utils::id_to_token(token_id) {
                if board.try_play(m).is_err() {
                    let s = format!("########### add_sequence => Board causing error {} while doing tokens {:#?}", sequence_id, token_ids);
                    panic!("{}", s)
                }
                repetition_tracker.push(&board);
            }
        }

        self.sequences.insert(sequence_id.clone(), board);
        self.repetition_trackers.insert(sequence_id.clone(), repetition_tracker);

        if self.send_sequence_data {
            self.allowed_tokens.insert(sequence_id.clone(), Vec::new());
            self.logprobs.insert(sequence_id, Vec::new());
        }
    }

    pub fn remove_sequence(&mut self, sequence_id: &String) {
        self.sequences.remove(sequence_id);
        self.repetition_trackers.remove(sequence_id);
    }

    pub fn send_sequence_data(&mut self, sequence_ids: &[String]) {
        if self.send_sequence_data {
            for sequence_id in sequence_ids {
                if let (Some(allowed_tokens), Some(logprobs)) = (
                    self.allowed_tokens.remove(sequence_id),
                    self.logprobs.remove(sequence_id),
                ) {
                    send_sequence_data(
                        self.sender.as_ref().unwrap(),
                        self.global_engine_id,
                        sequence_id,
                        allowed_tokens,
                        logprobs,
                    );
                }
            }
        }
    }

    pub fn apply_sampled_batch(&mut self, token_ids: &Vec<TokenId>, logprobs: &Vec<f32>) {
        for i in 0..self.sequence_batch_ids.len() {
            let sequence_id = self.sequence_batch_ids[i].clone();  // borrow
            let token_id = token_ids[i];
            let logprob = &logprobs[i];

            self.apply_sampled(&sequence_id, &vec![token_id], logprob);
        }
    }



    pub fn apply_sampled(&mut self, sequence_id: &String, token_ids: &Vec<TokenId>, logprob: &f32) {
        let board: &mut Board = &mut self.sequences.get_mut(sequence_id).unwrap();
        let repetition_tracker: &mut RepetitionTracker = &mut self.repetition_trackers.get_mut(sequence_id).unwrap();

        for &token_id in token_ids.iter() {
            if let Some(m) = utils::id_to_token(token_id) {
                if board.try_play(m).is_err() {
                    let mut valid_tokens = SequenceStatesPool::valid_tokens(board, repetition_tracker, &self.terminal_tokens);
                    valid_tokens.sort();
                    let s = format!("########### apply_sampled => Board causing error {} while doing tokens {:#?} but valid tokens are {:#?}", sequence_id, token_ids, valid_tokens);
                    panic!("{}", s)
                }
                repetition_tracker.push(board);
            }
        }

        if self.send_sequence_data {
            self.logprobs.get_mut(sequence_id).unwrap().push(*logprob);
        }
    }

    fn valid_tokens(
        board: &Board,
        repetition_tracker: &RepetitionTracker,
        terminal_tokens: &TerminalTokens,
    ) -> Vec<TokenId> {
        match board.status() {
            GameStatus::Drawn => vec![terminal_tokens.draw_id],
            GameStatus::Won => vec![match board.side_to_move() {
                Color::White => terminal_tokens.terminal_token_win_1,
                Color::Black => terminal_tokens.terminal_token_win_0,
            }],
            GameStatus::Ongoing => {
                if check_insufficient_material(board) || repetition_tracker.is_draw() {
                    vec![terminal_tokens.draw_id]
                }
                else {
                    let mut tokens = Vec::new();
                    board.generate_moves(|mv| {
                        tokens.extend(mv.into_iter().map(utils::token_to_id));
                        false
                    });
                    tokens
                }
            }
        }
    }

    pub fn all_valid_tokens_vec(&mut self, sequence_ids: &Vec<String>) -> Vec<Vec<TokenId>> {
        self.sequence_batch_ids = sequence_ids.clone();

        let all_tokens: Vec<Vec<TokenId>> = sequence_ids
            .iter()
            .map(|sequence_id| {
                let board = self.sequences.get(sequence_id).unwrap();
                let repetition_tracker = self.repetition_trackers.get(sequence_id).unwrap();
                let tokens = SequenceStatesPool::valid_tokens(board, repetition_tracker, &self.terminal_tokens);

                if self.send_sequence_data {
                    let allowed_tok = self.allowed_tokens.get_mut(sequence_id).unwrap();
                    allowed_tok.push(tokens.clone());
                }
                tokens
            })
            .collect();

        all_tokens
    }

    pub fn all_valid_tokens_mask(&mut self, sequence_ids: &Vec<String>) -> Tensor {
        self.sequence_batch_ids = sequence_ids.clone();

        let all_tokens: Vec<Vec<TokenId>> = sequence_ids
            .iter()
            .map(|sequence_id| {
                let board = self.sequences.get(sequence_id).unwrap();
                let repetition_tracker = self.repetition_trackers.get(sequence_id).unwrap();
                let tokens = SequenceStatesPool::valid_tokens(board, repetition_tracker, &self.terminal_tokens);

                if self.send_sequence_data {
                    let allowed_tok = self.allowed_tokens.get_mut(sequence_id).unwrap();
                    allowed_tok.push(tokens.clone());
                }
                tokens
            })
            .collect();

        let m = utils::create_mask(&all_tokens, &mut self.mask_pinned, &self.device);

        m
    }
}
