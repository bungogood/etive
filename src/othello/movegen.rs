const INNER_FILES: u64 = 0x7e7e7e7e7e7e7e7e;

macro_rules! moves_on_axis {
    ($player:expr, $opponent:expr, $empty:expr, $shift:expr) => {{
        let mut higher = ($player << $shift) & $opponent;
        let mut lower = ($player >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        ((higher << $shift) | (lower >> $shift)) & $empty
    }};
}

macro_rules! flips_on_axis {
    ($placed:expr, $player:expr, $opponent:expr, $shift:expr) => {{
        let mut higher = ($placed << $shift) & $opponent;
        let mut lower = ($placed >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;
        higher |= (higher << $shift) & $opponent;
        lower |= (lower >> $shift) & $opponent;

        let higher = if (higher << $shift) & $player != 0 {
            higher
        } else {
            0
        };
        let lower = if (lower >> $shift) & $player != 0 {
            lower
        } else {
            0
        };
        higher | lower
    }};
}

#[inline(always)]
pub(crate) fn legal_moves(player: u64, opponent: u64) -> u64 {
    let empty = !(player | opponent);
    let inner_opponent = opponent & INNER_FILES;
    moves_on_axis!(player, inner_opponent, empty, 1)
        | moves_on_axis!(player, opponent, empty, 8)
        | moves_on_axis!(player, inner_opponent, empty, 7)
        | moves_on_axis!(player, inner_opponent, empty, 9)
}

#[inline(always)]
pub(crate) fn flips(placed: u64, player: u64, opponent: u64) -> u64 {
    let inner_opponent = opponent & INNER_FILES;
    flips_on_axis!(placed, player, inner_opponent, 1)
        | flips_on_axis!(placed, player, opponent, 8)
        | flips_on_axis!(placed, player, inner_opponent, 7)
        | flips_on_axis!(placed, player, inner_opponent, 9)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_flips(placed: u64, player: u64, opponent: u64) -> u64 {
        if placed == 0 || placed.count_ones() != 1 || placed & (player | opponent) != 0 {
            return 0;
        }
        let index = placed.trailing_zeros() as i8;
        let file = index & 7;
        let rank = index >> 3;
        let mut result = 0;

        for (file_step, rank_step) in [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ] {
            let mut next_file = file + file_step;
            let mut next_rank = rank + rank_step;
            let mut captured = 0;
            while (0..8).contains(&next_file) && (0..8).contains(&next_rank) {
                let bit = 1_u64 << (next_rank * 8 + next_file);
                if opponent & bit != 0 {
                    captured |= bit;
                } else {
                    if player & bit != 0 {
                        result |= captured;
                    }
                    break;
                }
                next_file += file_step;
                next_rank += rank_step;
            }
        }
        result
    }

    fn next_random(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    #[test]
    fn bulk_generation_matches_directional_reference() {
        let mut random = 0x0065_7469_7665;
        for _ in 0..10_000 {
            let player = next_random(&mut random);
            let opponent = next_random(&mut random) & !player;
            let mut expected = 0;
            let mut empty = !(player | opponent);
            while empty != 0 {
                let placed = 1_u64 << empty.trailing_zeros();
                let reference = reference_flips(placed, player, opponent);
                assert_eq!(flips(placed, player, opponent), reference);
                if reference != 0 {
                    expected |= placed;
                }
                empty &= empty - 1;
            }
            assert_eq!(legal_moves(player, opponent), expected);
        }
    }
}
