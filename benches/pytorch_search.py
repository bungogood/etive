#!/usr/bin/env python3
"""Benchmark the Etive Othello network with PyTorch on CUDA."""

import gc
import math
import statistics
import time

import torch
from torch import nn


BATCH_SIZES = (512, 1024, 2048)
CHANNELS = 128
RESIDUAL_BLOCKS = 10
NORM_GROUPS = 8
ACTIONS = 65


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.norm1 = nn.GroupNorm(NORM_GROUPS, CHANNELS)
        self.conv2 = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
        self.norm2 = nn.GroupNorm(NORM_GROUPS, CHANNELS)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.norm1(self.conv1(inputs)))
        return torch.relu(self.norm2(self.conv2(hidden)) + inputs)


class OthelloNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(2, CHANNELS, 3, padding=1)
        self.stem_norm = nn.GroupNorm(NORM_GROUPS, CHANNELS)
        self.blocks = nn.ModuleList(ResidualBlock() for _ in range(RESIDUAL_BLOCKS))
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


class AutocastNetwork(nn.Module):
    def __init__(self, network: nn.Module, enabled: bool) -> None:
        super().__init__()
        self.network = network
        self.enabled = enabled

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.enabled):
            return self.network(inputs)


def benchmark_configuration(
    name: str,
    *,
    autocast: bool,
    channels_last: bool,
    compile_model: bool,
    references: dict[int, torch.Tensor],
) -> None:
    torch.manual_seed(7)
    network = OthelloNetwork().eval().cuda()
    if channels_last:
        network = network.to(memory_format=torch.channels_last)
    model: nn.Module = AutocastNetwork(network, autocast).eval()
    if compile_model:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    for batch_size in BATCH_SIZES:
        host_input = torch.zeros((batch_size, 2, 8, 8), dtype=torch.float32)
        if channels_last:
            host_input = host_input.contiguous(memory_format=torch.channels_last)

        def evaluate() -> torch.Tensor:
            inputs = host_input.cuda()
            policy, value = model(inputs)
            return torch.cat((policy, value), dim=1).float().cpu()

        with torch.inference_mode():
            for _ in range(5):
                output = evaluate()
            if not torch.isfinite(output).all():
                raise RuntimeError(f"{name} batch {batch_size} produced non-finite output")

            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            evaluate()
            one_iteration = time.perf_counter() - started
            iterations = max(10, min(100, math.ceil(3.0 / one_iteration)))
            durations = []
            for _ in range(iterations):
                started = time.perf_counter()
                output = evaluate()
                durations.append(time.perf_counter() - started)

        median_seconds = statistics.median(durations)
        throughput = batch_size / median_seconds
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        if name == "fp32-eager":
            references[batch_size] = output
            max_error = 0.0
        else:
            max_error = (output - references[batch_size]).abs().max().item()
        print(
            f"{name:27} batch={batch_size:4} "
            f"median={median_seconds * 1000:8.3f} ms "
            f"throughput={throughput:10.1f} positions/s "
            f"peak={peak_gib:6.2f} GiB max_error={max_error:.6f}",
            flush=True,
        )

    del model, network
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(0)} cudnn={torch.backends.cudnn.version()}")
    print("Timing includes FP32 host input upload and FP32 policy/value readback.")

    references: dict[int, torch.Tensor] = {}
    configurations = (
        ("fp32-eager", False, False, False),
        ("fp16-autocast-eager", True, False, False),
        ("fp16-autocast-channels-last", True, True, False),
        ("fp16-compiled-channels-last", True, True, True),
    )
    for name, autocast, channels_last, compile_model in configurations:
        benchmark_configuration(
            name,
            autocast=autocast,
            channels_last=channels_last,
            compile_model=compile_model,
            references=references,
        )


if __name__ == "__main__":
    main()
