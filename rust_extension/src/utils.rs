use cozy_chess::{BitBoard, Board, Color, Move, Piece, Square, File, Rank};
use tch::{Device, Kind, Tensor};

use once_cell::sync::Lazy;
use std::collections::HashSet;

pub type TokenId = i32;

fn promotion_variant_index(opt: Option<Piece>) -> usize {
    match opt {
        None => 0,
        Some(Piece::Queen) => 1,
        Some(Piece::Rook) => 2,
        Some(Piece::Bishop) => 3,
        Some(Piece::Knight) => 4,
        Some(_) => 0, // never happens
    }
}

// --- 1) ALL_TOKENS: Vec of all possible tokens ---
pub static ALL_TOKENS: Lazy<Vec<Move>> = Lazy::new(|| {
    let mut tmp: Vec<Move> = Vec::new();

    // 8 sliding directions (queen = rook + bishop)
    const SLIDE_DIRS: &[(i8, i8)] = &[
        (1, 0),  (-1, 0), (0, 1),  (0, -1), // rook-like
        (1, 1),  (1, -1), (-1, 1), (-1, -1) // bishop-like
    ];

    // knight offsets
    const KNIGHT_OFFS: &[(i8, i8)] = &[
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2),
    ];

    // iterate every source square
    for from in Square::ALL {
        let (fx, fy) = (from.file() as usize, from.rank() as usize);

        // --- 1) queen sliding moves from `from` (covers rook, bishop, and king single-steps,
        // and also covers pawn forward/capture non-promotion moves as a superset) ---
        for &(dx, dy) in SLIDE_DIRS {
            (1..8)
                .filter_map(|step| {
                    let x = fx as i8 + dx * step;
                    let y = fy as i8 + dy * step;

                    Some((
                        File::try_index(x as usize)?,
                        Rank::try_index(y as usize)?,
                    ))
                })
                .for_each(|(file, rank)| {
                    tmp.push(Move {
                        from,
                        to: Square::new(file, rank),
                        promotion: None,
                    });
                });
        }

        KNIGHT_OFFS
            .iter()
            .filter_map(|(dx, dy)| {
                let x = fx as i8 + dx;
                let y = fy as i8 + dy;

                Some((
                    File::try_index(x as usize)?,
                    Rank::try_index(y as usize)?,
                ))
            })
            .for_each(|(file, rank)| {
                    tmp.push(Move {
                        from,
                        to: Square::new(file, rank),
                        promotion: None,
                    });
                });
    }

    // --- 3) promotion moves (explicit) ---
    for &f in &File::ALL {
        for &(r, dy) in &[(Rank::Seventh, 1), (Rank::Second, -1)] {
            for dx in -1i8..=1 {
                let x = f as i8 + dx;
                let y = r as i8 + dy;

                // Skip invalid squares
                if let (Some(f_dest), Some(r_dest)) = (File::try_index(x as usize), Rank::try_index(y as usize)) {
                    for &promotion in &[Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
                        tmp.push(Move {
                            from: Square::new(f, r),
                            to: Square::new(f_dest, r_dest),
                            promotion: Some(promotion),
                        });
                    }
                }
            }
        }
    }

    // --- Deduplicate and sort deterministically ---
    // Use a HashSet to unique-ify (Move must be Hash+Eq; cozy-chess's Move typically is).
    let mut set: HashSet<Move> = HashSet::with_capacity(tmp.len());
    for mv in tmp {
        set.insert(mv);
    }

    let mut all: Vec<Move> = set.into_iter().collect();

    all.sort_by_key(|m| {
        let from_i = m.from as u8 as u32;
        let to_i = m.to as u8 as u32;
        let p = promotion_variant_index(m.promotion) as u32;
        (from_i << 16) | (to_i << 8) | p
    });

    all
});

// --- 2) Direct lookup table for token -> TokenId ---
pub static TOKEN_TO_ID_TABLE: Lazy<Vec<Option<TokenId>>> = Lazy::new(|| {
    let mut table = vec![None; 64 * 64 * 5]; // 64 from × 64 to × 5 promotions

    for (i, mv) in ALL_TOKENS.iter().enumerate() {
        let idx = (mv.from as usize) * 64 * 5
                + (mv.to as usize) * 5
                + promotion_variant_index(mv.promotion);
        table[idx] = Some(i as TokenId);
    }

    table
});

pub fn id_to_token(id: TokenId) -> Option<Move> {
    ALL_TOKENS.get(id as usize).cloned()
}

pub fn token_to_id(mv: Move) -> TokenId {
    let idx = (mv.from as usize) * 64 * 5
            + (mv.to as usize) * 5
            + promotion_variant_index(mv.promotion);

    TOKEN_TO_ID_TABLE[idx].expect("Token not in ALL_TOKENS")
}    


pub fn check_insufficient_material(board: &Board) -> bool {
    let num_white_pieces = board.colors(Color::White).len();
    let num_black_pieces = board.colors(Color::Black).len();

    if num_white_pieces + num_black_pieces == 2 {
        // Bare kings
        return true;
    }

    if num_white_pieces == 1 {
        // Bare white king, so knight and bishop are sured to be black
        return num_black_pieces == 2
            && (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)).len() == 1;
    }

    if num_black_pieces == 1 {
        // Bare black king, so knight and bishop are sured to be white
        return num_white_pieces == 2
            && (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop)).len() == 1;
    }

    if num_white_pieces == 2 && num_black_pieces == 2 {
        return (board.pieces(Piece::Bishop) & BitBoard::DARK_SQUARES).len() == 2
            || (board.pieces(Piece::Bishop) & BitBoard::LIGHT_SQUARES).len() == 2
    }

    false
}

pub fn parse_device(s: &str) -> Device {
    let s = s.to_lowercase(); // make it case-insensitive

    if s.starts_with("cuda:") {
        let i = s[5..].parse::<usize>().unwrap();
        Device::Cuda(i)
    } else if s == "cuda" {
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

// pub fn create_mask(indices: &[Vec<i32>], zeroes: &mut Tensor, out: &mut Tensor) {
//     let batch_size = indices.len() as i64;
//     let vocab_size = out.size()[1];

//     // Resize pinned + gpu tensors if batch size changed
//     if out.size()[0] != batch_size {
//         *zeroes = Tensor::empty([batch_size * vocab_size], (Kind::Float, out.device()));
//         *out = Tensor::empty([batch_size, vocab_size], (Kind::Float, out.device()));
//     }

//     out.fill_(f64::NEG_INFINITY);

//     // let batch_repeats: Vec<i64> = indices.iter().map(|indices| indices.len() as i64).collect();

//     // let batch_repeats_tensor = Tensor::from_slice(&batch_repeats).to_device(out.device());
//     // let batch_indices = Tensor::arange(batch_size, (Kind::Int64, out.device()))
//     //     .repeat_interleave_self_tensor(&batch_repeats_tensor, None, None);

//     let batch_indices = indices.iter().enumerate().flat_map(|(batch_idx, row)| {
//         std::iter::repeat(batch_idx as i64).take(row.len())
//     }).collect::<Vec<i64>>();

//     let batch_indices_tensor = Tensor::from_slice(&batch_indices).to_device(out.device());    

//     let flat_indices: Vec<i64> = indices.iter().flatten().map(|&x| x as i64).collect();

//     let flat_indices_tensor = Tensor::from_slice(&flat_indices).to_device(out.device());

//     // let zero_values = Tensor::zeros(flat_indices_tensor.size()[0], (Kind::Float, out.device()));
//     let zero_values = zeroes.slice(0, 0, flat_indices_tensor.size()[0], 1);

//     out.index_put_(
//         &[Some(&batch_indices_tensor), Some(&flat_indices_tensor)],
//         &zero_values,
//         false,
//     );
// }

// pub fn create_mask(indices: &Vec<Vec<i32>>, pinned: &mut Tensor, out:& Tensor) -> Tensor {
//     let mut out = out.copy();

//     let batch_size = indices.len() as i64;
//     let vocab_size = pinned.size()[1];

//     // Resize pinned + gpu tensors if batch size changed
    
//     let batch_size = indices.len() as i64;

//     let _ = out.fill_(f64::NEG_INFINITY);

//     let batch_repeats: Vec<i64> = indices.iter().map(|indices| indices.len() as i64).collect();

//     let batch_repeats_tensor = Tensor::from_slice(&batch_repeats).to_device(out.device());
//     let batch_indices = Tensor::arange(batch_size, (Kind::Int64, out.device()))
//         .repeat_interleave_self_tensor(&batch_repeats_tensor, None, None);

//     let flat_indices: Vec<i64> = indices.iter().flatten().map(|x| *x as i64).collect();

//     let flat_indices_tensor = Tensor::from_slice(&flat_indices).to_device(out.device());

//     let zero_values = Tensor::zeros(flat_indices_tensor.size()[0], (Kind::Float, out.device()));

//     out.index_put_(
//         &[Some(&batch_indices), Some(&flat_indices_tensor)],
//         &zero_values,
//         false,
//     )
// }

pub fn create_mask(indices: &[Vec<i32>], pinned: &mut Tensor, device: &Device) -> Tensor{
    let batch_size = indices.len() as i64;
    let vocab_size = pinned.size()[1];

    // Resize pinned + gpu tensors if batch size changed
    if pinned.size()[0] != batch_size {
        *pinned = Tensor::empty([batch_size, vocab_size], (Kind::Bool, Device::Cpu));
        if *device != Device::Cpu {
            *pinned = pinned.internal_pin_memory(*device);
        }
    }

    
    // Write directly into page-locked memory - no alloc, no tch kernel
    let ptr = pinned.data_ptr() as *mut bool;
    let vs = vocab_size as usize;
    
    unsafe {
        // Fill entire buffer with -inf in one fast pass
        let total = batch_size as usize * vs;
        for i in 0..total {
            *ptr.add(i) = true;
        }
        // Punch zeros at legal positions
        for (batch_idx, row) in indices.iter().enumerate() {
            let row_start = batch_idx * vs;
            for &idx in row {
                *ptr.add(row_start + idx as usize) = false;
            }
        }
    }
    
    pinned.to_device(*device)
}
