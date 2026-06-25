#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_dump_path = Path(__file__).resolve().parents[1] / "IR" / "gemmini-max-autotune"
    env_dump_path = os.environ.get("TRITON_CHIPYARD_DUMP_PATH", "").strip()
    parser = argparse.ArgumentParser(
        description=(
            "Run one large nn.Module matmul through torch.compile + triton-chipyard "
            "with Gemmini max-autotune candidates enabled."
        )
    )
    parser.add_argument("--m1", type=int, default=1024)
    parser.add_argument("--k1", type=int, default=1024)
    parser.add_argument("--n1", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--strict-compare",
        action="store_true",
        help="Raise an error when compiled outputs differ from eager outputs.",
    )
    parser.add_argument(
        "--dump-path",
        type=str,
        default=env_dump_path or str(default_dump_path),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("m1", "k1", "n1"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup_runs < 0 or args.repeat_runs <= 0:
        raise ValueError("warmup-runs must be non-negative and repeat-runs must be positive")


def configure_environment(task_name: str, dump_path: str) -> tuple[Path, Path]:
    resolved_dump_path = Path(dump_path).resolve()
    resolved_dump_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")
    os.environ.setdefault("TRITON_ALWAYS_COMPILE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "TRITON")
    os.environ.setdefault("TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER", "1")
    os.environ.setdefault("TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE", "1")
    os.environ["TRITON_CHIPYARD_DUMP_PATH"] = str(resolved_dump_path)

    cache_dir = Path(
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-chipyard-cache/{task_name}")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dump_path, cache_dir


def env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dump_path, cache_dir = configure_environment("gemmini-max-autotune", args.dump_path)

    import torch
    import torch._inductor.config as inductor_config
    import torch._inductor.select_algorithm as inductor_select_algorithm
    import torch._inductor.template_heuristics as inductor_template_heuristics
    import triton
    from triton.backends.triton_chipyard.driver import ChipyardDriver

    class SingleMatmulModule(torch.nn.Module):
        def __init__(
            self,
            shape: tuple[int, int, int],
            use_bias: bool,
        ) -> None:
            super().__init__()
            _, k1, n1 = shape
            self.weight1 = torch.nn.Parameter(torch.randn(k1, n1) * 0.05)
            self.bias1 = (
                torch.nn.Parameter(torch.randn(n1) * 0.01) if use_bias else None
            )

        def forward(
            self,
            x1: torch.Tensor,
        ) -> torch.Tensor:
            out1 = x1 @ self.weight1
            if self.bias1 is not None:
                out1 = out1 + self.bias1
            self.matmul()
            self.add()
            self.
            return out1

    torch.manual_seed(args.seed)
    device = "cpu"
    dtype = torch.float32

    triton.runtime.driver.set_active(ChipyardDriver())
    target = triton.runtime.driver.active.get_current_target()
    inductor_config.force_disable_caches = True
    inductor_config.max_autotune = True
    inductor_config.max_autotune_gemm_backends = "TRITON"
    inductor_config.cpu_backend = "triton_chipyard"

    mm_shape = (args.m1, args.k1, args.n1)
    model = SingleMatmulModule(mm_shape, args.bias).to(
        device=device, dtype=dtype
    ).eval()
    input1 = torch.randn(args.m1, args.k1, device=device, dtype=dtype)

    with torch.no_grad():
        eager_output = model(input1)
        compiled_model = torch.compile(
            model,
            backend="inductor",
            fullgraph=args.fullgraph,
        )
        for _ in range(args.warmup_runs):
            compiled_model(input1)
        compiled_output = None
        for _ in range(args.repeat_runs):
            compiled_output = compiled_model(input1)
        assert compiled_output is not None

    diff = (compiled_output - eager_output).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    close = torch.allclose(compiled_output, eager_output, atol=args.atol, rtol=args.rtol)
    gemmini_candidate_patch = hasattr(
        inductor_template_heuristics.CPUConfigHeuristic,
        "_use_gemmini_max_autotune_candidates",
    )
    gemmini_meta_patch = hasattr(
        inductor_select_algorithm,
        "TRITON_TEMPLATE_BACKEND_ONLY_META_KEYS",
    )

    print("[config] task=gemmini_max_autotune")
    print(f"[config] device={device}, dtype=float32")
    print(f"[config] inductor_cpu_backend={inductor_config.cpu_backend}")
    print(f"[config] triton_target_backend={target.backend}")
    print(f"[config] triton_target_arch={target.arch}")
    print(f"[config] chipyard_dump_path={dump_path}")
    print(f"[config] triton_cache_dir={cache_dir}")
    print(
        "[config] runner_enabled="
        f"{os.environ.get('TORCHINDUCTOR_ENABLE_CHIPYARD_RUNNER', '0')}"
    )
    print(f"[config] use_gemmini={env_flag('TRITON_CHIPYARD_USE_GEMMINI')}")
    print(
        "[config] inductor_template_heuristics="
        f"{inspect.getfile(inductor_template_heuristics)}"
    )
    print(
        "[config] gemmini_candidate_patch="
        f"{gemmini_candidate_patch}"
    )
    print(f"[config] gemmini_meta_patch={gemmini_meta_patch}")
    print(
        "[config] gemmini_max_autotune="
        f"{env_flag('TORCHINDUCTOR_GEMMINI_MAX_AUTOTUNE')}"
    )
    print(f"[config] fullgraph={args.fullgraph}")
    print(f"[config] shape=mm1={mm_shape}")
    print(f"[result] output_shape={tuple(compiled_output.shape)}")
    print(f"[result] max_abs_diff={max_abs_diff:.8e}")
    print(f"[result] close={close}")
    output_mean = float(compiled_output.mean().item())
    output_std = float(compiled_output.std().item())
    print(f"[result] output_mean={output_mean:.8e}")
    print(f"[result] output_std={output_std:.8e}")

    if args.strict_compare and not close:
        torch.testing.assert_close(
            compiled_output,
            eager_output,
            atol=args.atol,
            rtol=args.rtol,
        )


if __name__ == "__main__":
    main()
