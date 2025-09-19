import argparse
from .utils import set_seed, env_cache_setup
from .config import load_config, merge_config
from .train import train
from .profile import run_sweep, generate_plots


def main():
    parser = argparse.ArgumentParser(description="Language Model Training and Profiling CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- Train ----
    p_train = subparsers.add_parser("train", help="Train a model (baseline: no AMP)")
    p_train.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    p_train.add_argument("--override", nargs="*", help="Optional overrides, e.g. data.seq_len=1024")

    # ---- Profile (synthetic) ----
    p_profile = subparsers.add_parser("profile", help="Run a profiling sweep (synthetic)")
    p_profile.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    p_profile.add_argument("--sweep", type=str, required=True, help="Path to experiments YAML (sweep spec)")
    p_profile.add_argument(
        "--profile_type",
        type=str,
        required=True,
        choices=["seq_len", "batch_size", "amp"],
        help="Which sweep to run"
    )
    p_profile.add_argument("--override", nargs="*", help="Optional overrides applied before sweep")
    p_profile.add_argument("--amp", action="store_true", help="Enable AMP for the sweep (if applicable)")

    # ---- Optimize (synthetic ablation) ----
    p_opt = subparsers.add_parser("optimize", help="Run optimization ablation (synthetic)")
    p_opt.add_argument("--config", type=str, required=True, help="Path to base config YAML")
    p_opt.add_argument("--experiments", type=str, required=True, help="Path to experiments YAML (experiments list)")
    p_opt.add_argument("--override", nargs="*", help="Optional overrides applied before ablation")

    # ---- Plot ----
    p_plot = subparsers.add_parser("plot", help="Generate plots from a training CSV")
    p_plot.add_argument("--outdir", type=str, default="results/plots", help="Directory to save plots")
    p_plot.add_argument("--gpu-seq-len-file", type=str, help="CSV for GPU memory vs. sequence length plot")
    p_plot.add_argument("--batch-noamp", type=str, help="CSV for batch size sweep without AMP")
    p_plot.add_argument("--batch-amp", type=str, help="CSV for batch size sweep with AMP")
    p_plot.add_argument("--baseline-file", type=str, help="CSV for baseline training run")
    p_plot.add_argument("--grad-ckpt-file", type=str, help="CSV for gradient checkpointing run")
    p_plot.add_argument("--flash-attention-file", type=str, help="CSV for flash attention run")

    args = parser.parse_args()
    env_cache_setup()

    if args.command == "train":
        cfg = load_config(args.config)
        if args.override:
            cfg = merge_config(cfg, args.override)
        set_seed(int(cfg.seed))
        # Enforced inside train(): no AMP for baseline
        train(cfg)

    elif args.command == "profile":
        base = load_config(args.config)
        if args.override:
            base = merge_config(base, args.override)
        sweep = load_config(args.sweep)
        set_seed(int(base.seed))
        run_sweep(base, sweep, args.profile_type, use_amp=args.amp)

    elif args.command == "optimize":
        base = load_config(args.config)
        if args.override:
            base = merge_config(base, args.override)
        exps = load_config(args.experiments)
        set_seed(int(base.seed))
        run_opt_ablation(base, exps)

    elif args.command == "plot":
        generate_plots(args)


if __name__ == "__main__":
    main()
