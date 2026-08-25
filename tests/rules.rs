use std::str::FromStr;

use etive::othello::{BitBoard, Board, Color, GameStatus, Move, Square, perft};

fn square(coordinate: &str) -> Square {
    Square::from_str(coordinate).unwrap()
}

fn bitboard(coordinates: &[&str]) -> BitBoard {
    BitBoard(
        coordinates
            .iter()
            .fold(0, |bits, coordinate| bits | square(coordinate).bitboard().0),
    )
}

#[test]
fn published_initial_position_perft() {
    // Aart Bik's pass-as-ply sequence, independently reproduced by Magpie.
    let expected = [
        1_u64, 4, 12, 56, 244, 1_396, 8_200, 55_092, 390_216, 3_005_288, 24_571_284,
    ];
    let board = Board::default();
    for (depth, nodes) in expected.into_iter().enumerate() {
        assert_eq!(perft(&board, depth as u8), nodes, "depth {depth}");
    }
}

#[test]
fn initial_position_and_move_application_are_canonical() {
    let mut board = Board::default();
    let moves: Vec<_> = board
        .legal_placements()
        .into_iter()
        .map(|square| square.to_string())
        .collect();

    assert_eq!(moves, ["d3", "c4", "f5", "e6"]);
    assert_eq!(board.discs(Color::Black).len(), 2);
    assert_eq!(board.discs(Color::White).len(), 2);

    let c4 = square("c4");
    assert_eq!(board.flips(c4), square("d4").bitboard());
    board.play(Move::Place(c4));
    assert_eq!(board.side_to_move(), Color::White);
    assert_eq!(board.discs(Color::Black).len(), 4);
    assert_eq!(board.discs(Color::White).len(), 1);
}

#[test]
fn checked_play_rejects_illegal_moves_without_changing_the_board() {
    let mut board = Board::default();
    let initial = board;

    assert!(board.try_play(Move::Place(square("a1"))).is_err());
    assert_eq!(board, initial);
    assert!(board.try_play(Move::Pass).is_err());
    assert_eq!(board, initial);
}

#[test]
fn cached_flip_application_matches_normal_play() {
    let square = square("c4");
    let mut normal = Board::default();
    let mut cached = normal;
    let flips = cached.flips(square);

    normal.play(Move::Place(square));
    cached.play_with_flips_unchecked(square, flips);
    assert_eq!(cached, normal);
}

#[test]
fn occupied_squares_do_not_produce_flip_masks() {
    let board = Board::default();
    assert!(board.flips(square("d4")).is_empty());
    assert!(board.flips(square("e4")).is_empty());
}

#[test]
fn overlapping_discs_are_rejected() {
    let occupied = square("a1").bitboard();
    assert!(Board::from_discs(occupied, occupied, Color::Black).is_err());
}

#[test]
fn board_text_round_trips() {
    let board = Board::default();
    let encoded = "......../......../......../...BW.../...WB.../......../......../........ b";
    assert_eq!(board.to_string(), encoded);
    assert_eq!(Board::from_str(encoded).unwrap(), board);
}

#[test]
#[ignore = "deep perft verification"]
fn published_deep_initial_position_perft() {
    let board = Board::default();
    assert_eq!(perft(&board, 11), 212_258_800);
    assert_eq!(perft(&board, 12), 1_939_886_636);
}

#[test]
fn magpie_single_move_fixture() {
    // Magpie numbers use A1 as the most-significant bit, so reverse them for
    // Etive's A1-as-bit-zero representation.
    let black = BitBoard(0x8801_0000_8100_0049_u64.reverse_bits());
    let white = BitBoard(0x0048_2a1c_761c_2a00_u64.reverse_bits());
    let expected = BitBoard(0x0000_0000_0800_0000_u64.reverse_bits());
    let board = Board::from_discs(black, white, Color::Black).unwrap();

    assert_eq!(board.legal_placements(), expected);
    assert_eq!(perft(&board, 8), 1);
}

#[test]
fn magpie_high_mobility_fixture() {
    let black = BitBoard(0x0011_660c_3c2c_0000_u64.reverse_bits());
    let white = BitBoard(0x0066_0052_4052_5600_u64.reverse_bits());
    let expected_moves = BitBoard(0xff88_99a1_8381_a9ff_u64.reverse_bits());
    let expected_perft = [1_u64, 34, 267, 7_671, 71_392, 1_783_477];
    let board = Board::from_discs(black, white, Color::Black).unwrap();

    assert_eq!(board.legal_placements(), expected_moves);
    for (depth, nodes) in expected_perft.into_iter().enumerate() {
        assert_eq!(perft(&board, depth as u8), nodes, "depth {depth}");
    }
}

#[test]
fn one_move_can_flip_all_eight_directions() {
    let black = bitboard(&["b2", "d2", "f2", "b4", "f4", "b6", "d6", "f6"]);
    let white = bitboard(&["c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"]);
    let board = Board::from_discs(black, white, Color::Black).unwrap();
    let placed = square("d4");

    assert!(board.legal_placements().has(placed));
    assert_eq!(board.flips(placed), white);
}

#[test]
fn corner_move_does_not_wrap_across_edges() {
    let black = bitboard(&["a3", "c1", "c3"]);
    let white = bitboard(&["a2", "b1", "b2"]);
    let board = Board::from_discs(black, white, Color::Black).unwrap();
    let placed = square("a1");

    assert!(board.legal_placements().has(placed));
    assert_eq!(board.flips(placed), white);
}

#[test]
fn forced_pass_is_a_ply_and_double_pass_ends_the_game() {
    let mut board = Board::from_discs(
        square("a1").bitboard(),
        square("b1").bitboard(),
        Color::White,
    )
    .unwrap();

    assert!(board.is_pass_legal());
    assert_eq!(perft(&board, 1), 1);
    board.play(Move::Pass);
    assert_eq!(board.legal_placements(), square("c1").bitboard());
    board.play(Move::Place(square("c1")));
    assert_eq!(board.status(), GameStatus::Won(Color::Black));
    assert!(!board.is_pass_legal());
}

#[test]
fn malformed_positions_are_rejected() {
    for position in [
        "......../......../......../...BW.../...WB.../......../........ b",
        "......../......../......../...BX.../...WB.../......../......../........ b",
        "......../......../......../...BW.../...WB.../......../......../........ x",
        "......../......../......../...BW.../...WB.../......../......../........ b trailing",
    ] {
        assert!(Board::from_str(position).is_err(), "{position}");
    }
}
