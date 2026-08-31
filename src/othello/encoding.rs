//! Allocation-free game-state encoding for neural-network batches.

use super::{Board, Square};

pub struct OthelloEncoding;

impl OthelloEncoding {
    pub const LEN: usize = 128;

    pub fn encode(game: &Board, output: &mut [f32]) {
        assert_eq!(output.len(), Self::LEN);
        let side = game.side_to_move();
        encode_two_planes(
            game.discs(side).0,
            game.discs(!side).0,
            Square::COUNT,
            output,
        );
    }

    pub fn encode_batch(games: &[Board], output: &mut [f32]) {
        assert_eq!(
            output.len(),
            games.len() * Self::LEN,
            "incorrect batch encoding buffer length"
        );
        let (states, remainder) = output.as_chunks_mut::<{ Self::LEN }>();
        debug_assert!(remainder.is_empty());
        for (game, state) in games.iter().zip(states) {
            Self::encode(game, state);
        }
    }
}

fn encode_two_planes(mut player: u64, mut opponent: u64, area: usize, output: &mut [f32]) {
    output.fill(0.0);
    while player != 0 {
        let index = player.trailing_zeros() as usize;
        output[index] = 1.0;
        player &= player - 1;
    }
    while opponent != 0 {
        let index = opponent.trailing_zeros() as usize;
        output[area + index] = 1.0;
        opponent &= opponent - 1;
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Device, Tensor, TensorData};

    use super::*;
    use crate::othello::{Move, Square as OthelloSquare};

    #[test]
    fn othello_encoding_is_side_relative() {
        let mut board = Board::default();
        let mut output = [0.0; 128];
        OthelloEncoding::encode(&board, &mut output);

        assert_eq!(output.iter().sum::<f32>(), 4.0);
        assert_eq!(output[35], 1.0);
        assert_eq!(output[28], 1.0);
        assert_eq!(output[64 + 27], 1.0);
        assert_eq!(output[64 + 36], 1.0);

        board.play(Move::Place(OthelloSquare::new(3, 2).unwrap()));
        OthelloEncoding::encode(&board, &mut output);
        assert_eq!(output[..64].iter().sum::<f32>(), 1.0);
        assert_eq!(output[64..].iter().sum::<f32>(), 4.0);
    }

    #[test]
    fn batch_encoding_writes_contiguous_states() {
        let first = Board::default();
        let mut second = first;
        second.play(Move::Place(OthelloSquare::new(3, 2).unwrap()));
        let mut output = vec![0.0; 2 * OthelloEncoding::LEN];

        OthelloEncoding::encode_batch(&[first, second], &mut output);

        assert_eq!(output[..128].iter().sum::<f32>(), 4.0);
        assert_eq!(output[128..].iter().sum::<f32>(), 5.0);
    }

    #[test]
    fn encoded_batch_constructs_one_burn_tensor() {
        let games = [Board::default(); 2];
        let mut output = vec![0.0; games.len() * OthelloEncoding::LEN];
        OthelloEncoding::encode_batch(&games, &mut output);

        let tensor = Tensor::<1>::from_data(TensorData::from(output.as_slice()), &Device::flex())
            .reshape([games.len(), 2, 8, 8]);
        assert_eq!(tensor.dims(), [2, 2, 8, 8]);
        assert_eq!(tensor.sum().into_scalar::<f32>(), 8.0);
    }
}
