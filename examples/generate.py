"""Example CLI. Does not run on import; model paths and output are explicit."""
import argparse
from pathlib import Path

import torch

from diffusion_sampler import SamplingPipeline, Solver


def main():
    parser = argparse.ArgumentParser(description="Sample with composable diffusion models")
    parser.add_argument("--model", required=True, help="Local Diffusers directory or Hub repo")
    parser.add_argument("--family", default="auto", choices=["auto", "sd35", "flux1", "flux3", "qwen-image"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", choices=["default", "euler", "ab2"], default="default")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample.png"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.cpu_offload and args.device != "cuda":
        parser.error("--cpu-offload uses --device cuda")
    pipe = SamplingPipeline.from_pretrained(
        args.model, family=args.family, torch_dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only,
        solver=None if args.solver == "default" else Solver(args.solver),
    )
    if args.cpu_offload:
        pipe.enable_cpu_offload()
    else:
        pipe.to(args.device)
    optional = {
        "negative_prompt": args.negative_prompt, "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale, "width": args.width, "height": args.height,
    }
    result = pipe(args.prompt, seed=args.seed, **{k: v for k, v in optional.items() if v is not None})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(args.output)


if __name__ == "__main__":
    main()
