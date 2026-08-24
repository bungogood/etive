use std::hint::black_box;

use candle_core::{Device, Tensor};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use etive::encoding::{OthelloEncodingV1, StateEncoder, TicTacToeEncodingV1};
use etive::evaluator::{
    BatchEvaluator, Evaluator, OthelloCandleEvaluator, TicTacToeCandleEvaluator, UniformEvaluator,
};
use etive::game::Game;
use etive::mcts::{Mcts, MctsConfig};
use etive::model::{OthelloNetwork, TicTacToeNetwork};
use etive::{othello, tic_tac_toe};

fn search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    let device = benchmark_device();

    let mut candle = TicTacToeCandleEvaluator::new(device.clone(), 7).unwrap();
    let position = tic_tac_toe::Board::default();
    let mut logits = [0.0; 9];
    group.bench_function("tic-tac-toe Candle evaluation", |bencher| {
        bencher.iter(|| {
            black_box(
                candle
                    .evaluate(black_box(&position), black_box(&mut logits))
                    .unwrap(),
            )
        });
    });

    let network = TicTacToeNetwork::new(&device, 7).unwrap();
    for batch_size in [1_usize, 8, 32, 64, 128] {
        let positions = vec![position; batch_size];
        let mut input = vec![0.0; batch_size * TicTacToeEncodingV1::encoded_len()];
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("tic-tac-toe Candle batch", batch_size),
            &batch_size,
            |bencher, &batch_size| {
                bencher.iter(|| {
                    TicTacToeEncodingV1.encode_batch(&positions, &mut input);
                    let input = Tensor::from_slice(&input, (batch_size, 2, 3, 3), &device).unwrap();
                    let (policy, value) = network.forward(&input).unwrap();
                    black_box(policy.flatten_all().unwrap().to_vec1::<f32>().unwrap());
                    black_box(value.flatten_all().unwrap().to_vec1::<f32>().unwrap());
                });
            },
        );
    }

    let mut othello_candle = OthelloCandleEvaluator::new(device.clone(), 7).unwrap();
    for batch_size in [1_usize, 8, 32, 64, 128, 256, 512, 1_024] {
        let positions = vec![othello::Board::default(); batch_size];
        let mut policies = vec![0.0; batch_size * othello::Board::ACTION_COUNT];
        let mut values = vec![0.0; batch_size];
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("Othello Candle batch", batch_size),
            &batch_size,
            |bencher, _| {
                bencher.iter(|| {
                    othello_candle
                        .evaluate_batch(
                            black_box(&positions),
                            black_box(&mut policies),
                            black_box(&mut values),
                        )
                        .unwrap()
                });
            },
        );
    }

    let batch_size = 128;
    let positions = vec![othello::Board::default(); batch_size];
    let mut encoded = vec![0.0; batch_size * OthelloEncodingV1::encoded_len()];
    OthelloEncodingV1.encode_batch(&positions, &mut encoded);
    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("Othello pipeline upload/128", |bencher| {
        bencher.iter(|| {
            let input =
                Tensor::from_slice(black_box(&encoded), (batch_size, 2, 8, 8), &device).unwrap();
            device.synchronize().unwrap();
            black_box(input)
        });
    });

    let input = Tensor::from_slice(&encoded, (batch_size, 2, 8, 8), &device).unwrap();
    let othello_network = OthelloNetwork::new(&device, 7).unwrap();
    group.bench_function("Othello pipeline forward/128", |bencher| {
        bencher.iter(|| {
            let output = othello_network.forward(black_box(&input)).unwrap();
            device.synchronize().unwrap();
            black_box(output)
        });
    });

    group.throughput(Throughput::Elements((4 * batch_size) as u64));
    group.bench_function("Othello pipeline queued forward/4x128", |bencher| {
        bencher.iter(|| {
            let first = othello_network.forward(black_box(&input)).unwrap();
            let second = othello_network.forward(black_box(&input)).unwrap();
            let third = othello_network.forward(black_box(&input)).unwrap();
            let fourth = othello_network.forward(black_box(&input)).unwrap();
            device.synchronize().unwrap();
            black_box((first, second, third, fourth))
        });
    });

    let (policy, value) = othello_network.forward(&input).unwrap();
    let packed = Tensor::cat(&[&policy, &value], 1)
        .unwrap()
        .flatten_all()
        .unwrap();
    device.synchronize().unwrap();
    group.throughput(Throughput::Elements(batch_size as u64));
    group.bench_function("Othello pipeline readback/128", |bencher| {
        bencher.iter(|| black_box(packed.to_vec1::<f32>().unwrap()));
    });

    for simulations in [128_u32, 1_024] {
        group.throughput(Throughput::Elements(u64::from(simulations)));
        group.bench_with_input(
            BenchmarkId::new("tic-tac-toe uniform PUCT", simulations),
            &simulations,
            |bencher, &simulations| {
                bencher.iter(|| {
                    let mut tree = Mcts::new(position, MctsConfig::default());
                    tree.run(&mut UniformEvaluator, simulations).unwrap();
                    black_box(tree)
                });
            },
        );
    }

    let simulations = 128_u32;
    group.throughput(Throughput::Elements(u64::from(simulations)));
    group.bench_function("tic-tac-toe Candle PUCT 128", |bencher| {
        bencher.iter(|| {
            let mut tree = Mcts::new(position, MctsConfig::default());
            tree.run(&mut candle, simulations).unwrap();
            black_box(tree)
        });
    });

    let simulations = 1_024_u32;
    group.throughput(Throughput::Elements(u64::from(simulations)));
    group.bench_function("Othello uniform PUCT 1024", |bencher| {
        bencher.iter(|| {
            let mut tree = Mcts::new(othello::Board::default(), MctsConfig::default());
            tree.run(&mut UniformEvaluator, simulations).unwrap();
            black_box(tree)
        });
    });

    group.finish();
}

fn hybrid(c: &mut Criterion) {
    #[cfg(all(feature = "accelerate", feature = "metal"))]
    {
        let cpu_batch_size = 256;
        let metal_batch_size = 128;
        let cpu_positions = vec![othello::Board::default(); cpu_batch_size];
        let metal_positions = vec![othello::Board::default(); metal_batch_size];
        let mut cpu = OthelloCandleEvaluator::new(Device::Cpu, 7).unwrap();
        let mut metal = OthelloCandleEvaluator::new(Device::new_metal(0).unwrap(), 7).unwrap();
        let mut cpu_policies = vec![0.0; cpu_batch_size * othello::Board::ACTION_COUNT];
        let mut metal_policies = vec![0.0; metal_batch_size * othello::Board::ACTION_COUNT];
        let mut cpu_values = vec![0.0; cpu_batch_size];
        let mut metal_values = vec![0.0; metal_batch_size];
        let mut group = c.benchmark_group("hybrid");
        group.throughput(Throughput::Elements(
            (cpu_batch_size + metal_batch_size) as u64,
        ));
        group.bench_function("Othello Accelerate 256 + Metal 128", |bencher| {
            bencher.iter(|| {
                std::thread::scope(|scope| {
                    let cpu_run = scope.spawn(|| {
                        cpu.evaluate_batch(
                            black_box(&cpu_positions),
                            black_box(&mut cpu_policies),
                            black_box(&mut cpu_values),
                        )
                    });
                    let metal_run = scope.spawn(|| {
                        metal.evaluate_batch(
                            black_box(&metal_positions),
                            black_box(&mut metal_policies),
                            black_box(&mut metal_values),
                        )
                    });
                    cpu_run.join().unwrap().unwrap();
                    metal_run.join().unwrap().unwrap();
                })
            });
        });
        group.finish();
    }

    #[cfg(not(all(feature = "accelerate", feature = "metal")))]
    let _ = c;
}

criterion_group!(benches, search, hybrid);
criterion_main!(benches);

fn benchmark_device() -> Device {
    #[cfg(feature = "cuda")]
    return Device::new_cuda(0).unwrap();

    #[cfg(all(not(feature = "cuda"), feature = "cudnn"))]
    return Device::new_cuda(0).unwrap();

    #[cfg(all(not(feature = "cuda"), not(feature = "cudnn"), feature = "metal"))]
    return Device::new_metal(0).unwrap();

    #[cfg(not(any(feature = "cuda", feature = "cudnn", feature = "metal")))]
    Device::Cpu
}
