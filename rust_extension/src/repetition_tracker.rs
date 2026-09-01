use ahash::AHashMap;
use cozy_chess::Board;

/// Track repetition counts per position using Board::hash() as the key.
/// - Counts saturate at 255 (u8).
pub struct RepetitionTracker {
    counts: AHashMap<u64, u8>,
    drawn: bool,
    _reserved: usize,
}

impl RepetitionTracker {
    pub fn new(capacity_hint: Option<usize>) -> Self {
        let mut counts = AHashMap::default();
        if let Some(cap) = capacity_hint {
            counts.reserve(cap);
        }

        RepetitionTracker {
            counts,
            drawn: false,
            _reserved: capacity_hint.unwrap_or(0),
        }
    }

    pub fn push(&mut self, board: &Board) {
        let key = board.hash();
        // Very hot path: update count in map.
        let entry = self.counts.entry(key);
        let new_count = match entry {
            std::collections::hash_map::Entry::Occupied(mut occ) => {
                let c = occ.get_mut();
                if *c < u8::MAX {
                    *c = c.saturating_add(1);
                }
                *c
            }
            std::collections::hash_map::Entry::Vacant(vac) => {
                vac.insert(1);
                1u8
            }
        };

        self.drawn = self.drawn || new_count >= 3;
    }

    pub fn is_draw(&self) -> bool {
        self.drawn
    }
}
