use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use etive::othello::{Board, Move, perft};

fn sample_positions() -> Vec<Board> {
    let mut board = Board::default();
    let mut positions = Vec::with_capacity(48);
    for ply in 0..48 {
        positions.push(board);
        let moves = board.legal_placements();
        if moves.is_empty() {
            if board.is_pass_legal() {
                board.pass_unchecked();
                continue;
            }
            break;
        }
        let index = ply as usize % moves.len() as usize;
        let square = moves.into_iter().nth(index).unwrap();
        board.play_unchecked(Move::Place(square));
    }
    positions
}

fn rules(c: &mut Criterion) {
    let positions = sample_positions();

    c.bench_function("legal moves across game", |bencher| {
        bencher.iter(|| {
            for board in &positions {
                black_box(black_box(*board).legal_placements());
            }
        });
    });

    c.bench_function("flips across game", |bencher| {
        bencher.iter(|| {
            for board in &positions {
                for square in board.legal_placements() {
                    black_box(black_box(*board).flips(black_box(square)));
                }
            }
        });
    });

    c.bench_function("initial perft 9", |bencher| {
        let board = Board::default();
        bencher.iter(|| perft(black_box(&board), 9));
    });
}

criterion_group!(benches, rules);
criterion_main!(benches);
