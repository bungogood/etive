use std::time::Instant;

use candle_core::{Device, Tensor};
use clap::{Parser, Subcommand};
use etive::othello::{Board, perft};

#[derive(Parser)]
#[command(version, about, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Verify Candle on the selected device.
    Candle,
    /// Count opening-position leaves.
    Perft {
        /// Search depth in plies.
        #[arg(default_value_t = 10)]
        depth: u8,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Command::Perft { depth } => {
            let board = Board::default();
            let start = Instant::now();
            let nodes = perft(&board, depth);
            let elapsed = start.elapsed();
            let nps = nodes as f64 / elapsed.as_secs_f64();
            println!("{nodes} nodes in {elapsed:.3?} ({nps:.0} nps)");
        }
        Command::Candle => {
            let device = candle_device()?;
            let tensor = Tensor::new(&[1_f32, 2.0, 3.0, 4.0], &device)?;
            let result = tensor.sqr()?.sum_all()?.to_scalar::<f32>()?;
            println!("Candle smoke test passed on {device:?}: {result}");
        }
    }
    Ok(())
}

fn candle_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    return Device::new_cuda(0);

    #[cfg(all(not(feature = "cuda"), feature = "cudnn"))]
    return Device::new_cuda(0);

    #[cfg(all(not(feature = "cuda"), not(feature = "cudnn"), feature = "metal"))]
    return Device::new_metal(0);

    #[cfg(not(any(feature = "cuda", feature = "cudnn", feature = "metal")))]
    Ok(Device::Cpu)
}
