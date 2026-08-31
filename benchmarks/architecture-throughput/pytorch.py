#!/usr/bin/env python3
import argparse
import time

import torch
from torch import nn

CHANNELS = 64
RESIDUAL_BLOCKS = 4
NORM_GROUPS = 8
ACTIONS = 65


class ResidualBlock(nn.Module):
    def __init__(self, norm: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding="same")
        self.norm1 = normalization(norm)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding="same")
        self.norm2 = normalization(norm)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.norm1(self.conv1(inputs)))
        return torch.relu(self.norm2(self.conv2(hidden)) + inputs)


class Network(nn.Module):
    def __init__(self, norm: str) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, CHANNELS, 3, padding="same")
        self.stem_norm = normalization(norm)
        self.blocks = nn.ModuleList(ResidualBlock(norm) for _ in range(RESIDUAL_BLOCKS))
        self.policy_conv = nn.Conv2d(CHANNELS, 2, 1)
        self.policy = nn.Linear(2 * 8 * 8, ACTIONS)
        self.value_conv = nn.Conv2d(CHANNELS, 1, 1)
        self.value_hidden = nn.Linear(8 * 8, CHANNELS)
        self.value = nn.Linear(CHANNELS, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.relu(self.stem_norm(self.stem(inputs)))
        for block in self.blocks:
            hidden = block(hidden)
        policy = torch.relu(self.policy_conv(hidden)).flatten(1)
        policy = self.policy(policy)
        value = torch.relu(self.value_conv(hidden)).flatten(1)
        value = torch.relu(self.value_hidden(value))
        return policy, torch.tanh(self.value(value))


def normalization(norm: str) -> nn.Module:
    if norm == "group":
        return nn.GroupNorm(NORM_GROUPS, CHANNELS)
    if norm == "batch":
        return nn.BatchNorm2d(CHANNELS)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalization: {norm}")


def benchmark(
    name: str,
    model: nn.Module,
    inputs: torch.Tensor,
    batch: int,
    warmup: int,
    iterations: int,
    norm: str,
) -> None:
    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        torch.cuda.synchronize()

        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started

    batches_per_second = iterations / seconds
    print(
        f"backend={name} dtype=float16 layout=nchw norm={norm} batch={batch} "
        f"channels={CHANNELS} blocks={RESIDUAL_BLOCKS} groups={NORM_GROUPS}"
    )
    print(f"iterations={iterations} elapsed_seconds={seconds:.6f}")
    print(f"milliseconds_per_batch={1000.0 / batches_per_second:.3f}")
    print(f"positions_per_second={batches_per_second * batch:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--mode", choices=("eager", "compile", "both"), default="both")
    parser.add_argument("--norm", choices=("group", "batch", "none"), default="group")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.batch <= 0 or args.iterations <= 0:
        raise SystemExit("batch and iterations must be positive")

    torch.manual_seed(7)
    torch.backends.cudnn.benchmark = True
    model = Network(args.norm).cuda().half().eval()
    inputs = torch.zeros(args.batch, 2, 8, 8, device="cuda", dtype=torch.float16)

    if args.mode in ("eager", "both"):
        benchmark(
            "pytorch-eager",
            model,
            inputs,
            args.batch,
            args.warmup,
            args.iterations,
            args.norm,
        )
    if args.mode in ("compile", "both"):
        compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        benchmark(
            "pytorch-compile",
            compiled,
            inputs,
            args.batch,
            args.warmup,
            args.iterations,
            args.norm,
        )


if __name__ == "__main__":
    main()
