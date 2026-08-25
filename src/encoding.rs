//! Allocation-free game-state encoding for neural-network batches.

use crate::game::Game;
use crate::{othello, tic_tac_toe};

/// Writes game states directly into caller-owned contiguous `f32` storage.
pub trait StateEncoder<G: Game>: Send + Sync {
    /// Tensor dimensions for one state in channels-first order.
    const SHAPE: [usize; 3];

    /// Stable encoding revision for model and data compatibility.
    const VERSION: u32;

    fn encode(&self, game: &G, output: &mut [f32]);

    fn encoded_len() -> usize {
        Self::SHAPE.into_iter().product()
    }

    fn encode_batch(&self, games: &[G], output: &mut [f32]) {
        let encoded_len = Self::encoded_len();
        assert_eq!(
            output.len(),
            games.len() * encoded_len,
            "incorrect batch encoding buffer length"
        );
        for (game, state) in games.iter().zip(output.chunks_exact_mut(encoded_len)) {
            self.encode(game, state);
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OthelloEncodingV1;

impl StateEncoder<othello::Board> for OthelloEncodingV1 {
    const SHAPE: [usize; 3] = [2, 8, 8];
    const VERSION: u32 = 1;

    fn encode(&self, game: &othello::Board, output: &mut [f32]) {
        assert_eq!(output.len(), Self::encoded_len());
        let side = game.side_to_move();
        encode_two_planes(
            game.discs(side).0,
            game.discs(!side).0,
            othello::Square::COUNT,
            output,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TicTacToeEncodingV1;

impl StateEncoder<tic_tac_toe::Board> for TicTacToeEncodingV1 {
    const SHAPE: [usize; 3] = [2, 3, 3];
    const VERSION: u32 = 1;

    fn encode(&self, game: &tic_tac_toe::Board, output: &mut [f32]) {
        assert_eq!(output.len(), Self::encoded_len());
        let side = game.side_to_move();
        encode_two_planes(
            u64::from(game.marks(side)),
            u64::from(game.marks(!side)),
            tic_tac_toe::Square::COUNT,
            output,
        );
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
    use candle_core::{Device, Tensor};

    use super::*;
    use crate::othello::{Move, Square as OthelloSquare};
    use crate::tic_tac_toe::Square as TicTacToeSquare;

    #[test]
    fn othello_encoding_is_side_relative() {
        let mut board = othello::Board::default();
        let mut output = [0.0; 128];
        OthelloEncodingV1.encode(&board, &mut output);

        assert_eq!(output.iter().sum::<f32>(), 4.0);
        assert_eq!(output[35], 1.0);
        assert_eq!(output[28], 1.0);
        assert_eq!(output[64 + 27], 1.0);
        assert_eq!(output[64 + 36], 1.0);

        board.play(Move::Place(OthelloSquare::new(3, 2).unwrap()));
        OthelloEncodingV1.encode(&board, &mut output);
        assert_eq!(output[..64].iter().sum::<f32>(), 1.0);
        assert_eq!(output[64..].iter().sum::<f32>(), 4.0);
    }

    #[test]
    fn tic_tac_toe_encoding_tracks_the_player_to_move() {
        let mut board = tic_tac_toe::Board::default();
        let mut output = [0.0; 18];

        board.play(TicTacToeSquare::from_index(0).unwrap());
        TicTacToeEncodingV1.encode(&board, &mut output);
        assert_eq!(output[0], 0.0);
        assert_eq!(output[9], 1.0);

        board.play(TicTacToeSquare::from_index(4).unwrap());
        TicTacToeEncodingV1.encode(&board, &mut output);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[9 + 4], 1.0);
    }

    #[test]
    fn batch_encoding_writes_contiguous_states() {
        let first = othello::Board::default();
        let mut second = first;
        second.play(Move::Place(OthelloSquare::new(3, 2).unwrap()));
        let mut output = vec![0.0; 2 * OthelloEncodingV1::encoded_len()];

        OthelloEncodingV1.encode_batch(&[first, second], &mut output);

        assert_eq!(output[..128].iter().sum::<f32>(), 4.0);
        assert_eq!(output[128..].iter().sum::<f32>(), 5.0);
    }

    #[test]
    fn encoded_batch_constructs_one_candle_tensor() {
        let games = [othello::Board::default(); 2];
        let mut output = vec![0.0; games.len() * OthelloEncodingV1::encoded_len()];
        OthelloEncodingV1.encode_batch(&games, &mut output);

        let tensor = Tensor::from_slice(&output, (games.len(), 2, 8, 8), &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), [2, 2, 8, 8]);
        assert_eq!(tensor.sum_all().unwrap().to_scalar::<f32>().unwrap(), 8.0);
    }
}
