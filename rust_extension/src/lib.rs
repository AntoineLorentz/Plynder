use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use tokio::runtime::Runtime;

mod utils;
use utils::TokenId;

mod sequences_states_pool;
use sequences_states_pool::SequenceStatesPool;

use pyo3_tch::PyTensor;
use tch::Tensor;

mod writer;
mod repetition_tracker;


//
// ==========================
// TerminalTokens
// ==========================
//

#[pyclass]
#[derive(Clone)]
pub struct TerminalTokens {
    pub draw_id: i32,
    pub terminal_token_win_0: i32,
    pub terminal_token_win_1: i32,
}

#[pymethods]
impl TerminalTokens {
    #[new]
    fn new(draw_id: i32, terminal_token_win_0: i32, terminal_token_win_1: i32) -> Self {
        TerminalTokens {
            draw_id,
            terminal_token_win_0,
            terminal_token_win_1,
        }
    }
}

//
// ==========================
// BoardsVllmAsync
// ==========================
//

#[pyclass]
pub struct SequenceStatesAsync {
    sequences_checker: Arc<Mutex<SequenceStatesPool>>,
    runtime: Runtime,
    valid_tokens_mask: Option<tokio::task::JoinHandle<Tensor>>,
    valid_tokens_vec: Option<tokio::task::JoinHandle<Vec<Vec<TokenId>>>>,
}

#[pymethods]
impl SequenceStatesAsync {
    #[new]
    fn new(
        vocab_size: i64,
        device: Option<String>,
        terminal_tokens: TerminalTokens,
        rollout_address: Option<String>,
        global_engine_id: Option<i32>,
    ) -> Self {
        SequenceStatesAsync {
            sequences_checker: Arc::new(Mutex::new(SequenceStatesPool::new(
                vocab_size,
                device.unwrap_or(String::from("cpu")),
                terminal_tokens,
                rollout_address,
                global_engine_id.unwrap_or(0),
            ))),
            runtime: Runtime::new().unwrap(),
            valid_tokens_mask: None,
            valid_tokens_vec: None,
        }
    }

    fn add_sequence(&mut self, sequence_id: String, action_ids: Vec<TokenId>) {
        self.sequences_checker
            .lock()
            .unwrap()
            .add_sequence(sequence_id, &action_ids);
    }

    fn remove_sequence(&mut self, sequence_id: String) {
        self.sequences_checker
            .lock()
            .unwrap()
            .remove_sequence(&sequence_id);
    }

    fn spawn_send_sequence_data(&mut self, sequence_ids: Vec<String>) {
        let sequences_checker = Arc::clone(&self.sequences_checker);
        self.runtime.spawn(async move {
            sequences_checker
                .lock()
                .unwrap()
                .send_sequence_data(&sequence_ids);
        });
    }

    fn apply_sampled(
        &mut self,
        sequence_id: String,
        action_ids: Vec<TokenId>,
        logprob: f32,
    ) {
        self.sequences_checker
            .lock()
            .unwrap()
            .apply_sampled(&sequence_id, &action_ids, &logprob);
    }

    fn apply_sampled_batch(
        &mut self,
        action_ids: Vec<TokenId>,
        logprobs: Vec<f32>,
    ) {
        self.sequences_checker
            .lock()
            .unwrap()
            .apply_sampled_batch(&action_ids, &logprobs);
    }


    fn spawn_valid_tokens_mask(&mut self, sequence_ids: Vec<String>) {
        let sequences_checker = Arc::clone(&self.sequences_checker);
        self.valid_tokens_mask = Some(self.runtime.spawn(async move {
            sequences_checker
                .lock()
                .unwrap()
                .all_valid_tokens_mask(&sequence_ids)
        }));
    }

    fn join_get_valid_tokens_mask(&mut self) -> PyTensor {
        let mask = self
            .runtime
            .block_on(
                self.valid_tokens_mask
                    .take()
                    .expect("spawn_valid_tokens_mask must be called first"),
            )
            .unwrap();

        PyTensor(mask)
    }

    fn spawn_valid_tokens_vec(&mut self, sequence_ids: Vec<String>) {
        let sequences_checker = Arc::clone(&self.sequences_checker);
        self.valid_tokens_vec = Some(self.runtime.spawn(async move {
            sequences_checker
                .lock()
                .unwrap()
                .all_valid_tokens_vec(&sequence_ids)
        }));
    }

    fn join_get_valid_tokens_vec(&mut self) -> Vec<Vec<TokenId>> {
        self.runtime
            .block_on(
                self.valid_tokens_vec
                    .take()
                    .expect("spawn_valid_tokens_vec must be called first"),
            )
            .unwrap()
    }
}

//
// ==========================
// PyO3 module
// ==========================
//

#[pymodule]
fn plynder_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TerminalTokens>()?;
    m.add_class::<SequenceStatesAsync>()?;
    Ok(())
}