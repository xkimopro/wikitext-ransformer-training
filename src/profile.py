import argparse
import copy
import math
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from .utils import set_seed, CSVLogger, Timer, peak_mem_mb, count_params
from .config import DotDict, set_value_at_path
from .dataset import get_dataloaders
from .model import build_model_from_config


# ---------- HW knobs ----------

def _device_from_cfg(cfg: DotDict) -> str:
    dev = str(cfg.hardware.device)
    if dev == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return dev


def _apply_hw_knobs(cfg: DotDict):
    tf32 = bool(cfg.hardware.tf32)
    try:
        import torch.backends
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        torch.set_float32_matmul_precision("high" if tf32 else "medium")
    except Exception:
        pass

    cudnn_bench = bool(cfg.hardware.cudnn_benchmark)
    try:
        torch.backends.cudnn.benchmark = cudnn_bench
    except Exception:
        pass


def _logs_dir(cfg: DotDict) -> Path:
    p = Path(str(cfg.paths.logs_dir))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------- Builders ----------

def _build_optimizer(model, cfg):
    assert cfg.optim.name.lower() == "adamw", "Only AdamW is supported by this baseline (from YAML)."
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=(float(cfg.optim.adam_beta1), float(cfg.optim.adam_beta2)),
        eps=float(cfg.optim.adam_eps),
        foreach=bool(cfg.optim.foreach),
        fused=bool(cfg.optim.fused),
    )


def _build_scheduler(optimizer, cfg):
    from transformers import get_cosine_schedule_with_warmup
    name = str(cfg.scheduler.name).lower()
    if name == "none":
        class _NoSched:
            def step(self): ...
        return _NoSched()
    elif name == "cosine_with_warmup":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.scheduler.warmup_steps),
            num_training_steps=int(cfg.train.max_steps),
        )
    else:
        raise ValueError(f"Unknown scheduler.name: {cfg.scheduler.name}")


# ---------- Eval helpers ----------

@torch.no_grad()
def _evaluate_metrics(model, val_loader, device, cfg, use_amp: bool = False, amp_dtype: torch.dtype = torch.float16):
    """
    Returns (mean_val_loss, val_ppl, val_token_top1_accuracy).
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    steps = 0
    max_eval_steps = int(cfg.train.eval_max_steps)

    for batch in tqdm(val_loader, desc="Evaluating (metrics)", dynamic_ncols=True):
        if device == "cuda":
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                out = model(**batch)
        else:
            out = model(**batch)

        loss = out["loss"]
        logits = out["logits"]                  # [B, T-1, V]
        labels = batch["input_ids"][:, 1:]      # [B, T-1]
        preds  = logits.argmax(dim=-1)          # [B, T-1]

        if "attention_mask" in batch:
            mask = batch["attention_mask"][:, 1:].to(dtype=torch.bool)
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()
        else:
            correct += (preds == labels).sum().item()
            total += labels.numel()

        losses.append(loss)
        steps += 1
        if steps >= max_eval_steps:
            break

    mean_loss = torch.stack(losses).mean().float().item()
    try:
        ppl = float(math.exp(mean_loss))
    except OverflowError:
        ppl = float("inf")
    acc = float(correct) / max(1, total)
    return mean_loss, ppl, acc


# ---------- AMP profile (baseline-style loop just for AMP sweep) ----------

def _train_like_with_amp(cfg: DotDict, steps: int, amp_flag: bool, out_csv: Path):
    device = _device_from_cfg(cfg)
    _apply_hw_knobs(cfg)
    set_seed(int(cfg.seed))

    use_amp = bool(amp_flag and device == "cuda")
    use_bf16 = bool(getattr(cfg.hardware, "bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = build_model_from_config(cfg).to(device)
    model.train()

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    logger = CSVLogger(
        out_csv,
        fieldnames=[
            "experiment_name",
            "mode",
            "step",
            "cumulative_time",
            "tokens_per_sec",
            "peak_mem_mb",
            "loss",
            "ppl",
            "accuracy",
            "amp",
            "ddp",
            "params",
        ],
    )

    max_steps = int(steps)
    grad_accum = int(cfg.train.grad_accum)
    tps_alpha = float(cfg.train.tps_ema_alpha)
    rolling_tps = 0.0
    total_params = count_params(model)

    train_iter = iter(train_loader)
    est_train_examples = max(1, int(cfg.data.estimated_train_examples))
    processed_examples = 0
    sync_cuda = bool(cfg.logging.sync_cuda_timer)
    cumulative_time = 0.0

    pbar = tqdm(range(1, max_steps + 1), desc=f"AMP Profile ({'on' if use_amp else 'off'})", dynamic_ncols=True)

    for step in pbar:
        with Timer(sync_cuda=(device == "cuda" and sync_cuda)) as timer:
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                if device == "cuda":
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                if use_amp and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        outputs = model(**batch)
                        loss = outputs["loss"] / grad_accum
                    scaler.scale(loss).backward()
                else:
                    outputs = model(**batch)
                    loss = outputs["loss"] / grad_accum
                    loss.backward()

                if "input_ids" in batch:
                    processed_examples += int(batch["input_ids"].size(0))

            if use_amp and device == "cuda":
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()

        step_time = float(timer)
        cumulative_time += step_time
        tokens_in_batch = batch["input_ids"].numel() * grad_accum
        tokens_per_sec = tokens_in_batch / max(step_time, 1e-6)
        rolling_tps = tokens_per_sec if rolling_tps == 0.0 else (
            tps_alpha * tokens_per_sec + (1 - tps_alpha) * rolling_tps
        )

        percent_complete = min(100.0, (processed_examples / est_train_examples) * 100.0)
        if (step % int(cfg.logging.log_every) == 0) or (step == 1):
            pbar.set_description(f"AMP Profile ({'on' if use_amp else 'off'}) ({percent_complete:.2f}% consumed)")
            pbar.set_postfix(loss=f"{(loss.item()*grad_accum):.4f}", tps=f"{rolling_tps:.1f}")

        if step % int(cfg.train.eval_every) == 0:
            val_loss, val_ppl, val_acc = _evaluate_metrics(model, val_loader, device, cfg, use_amp=use_amp, amp_dtype=amp_dtype)
            logger.log(
                {
                    "experiment_name": cfg.experiment_name,
                    "mode": "eval",
                    "step": step,
                    "cumulative_time": cumulative_time,
                    "tokens_per_sec": rolling_tps,
                    "peak_mem_mb": peak_mem_mb(),
                    "loss": float(loss.item() * grad_accum),   # training loss
                    "ppl": float(val_ppl),                     # validation ppl
                    "accuracy": float(val_acc),                # validation top-1 accuracy
                    "amp": bool(use_amp),
                    "ddp": bool(cfg.hardware.ddp),
                    "params": total_params,
                }
            )
            model.train()


# ---------- Real-data profiler for seq_len / batch_size ----------

def _profile_run_real(cfg: DotDict, steps: int, amp_flag: bool):
    """
    Real (non-synthetic) micro benchmark with the real dataloader.
    Returns (tokens_per_sec_avg, peak_mem_mb, cumulative_time_sec, val_loss, val_ppl, val_accuracy).
    """
    device = _device_from_cfg(cfg)
    _apply_hw_knobs(cfg)
    set_seed(int(cfg.seed))

    train_loader, val_loader, _ = get_dataloaders(cfg)
    model = build_model_from_config(cfg).to(device)
    model.train()

    optimizer = _build_optimizer(model, cfg)
    use_amp = bool(amp_flag and device == "cuda")
    use_bf16 = bool(getattr(cfg.hardware, "bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    train_iter = iter(train_loader)
    warmup = min(10, max(0, steps // 10))
    cumulative_time = 0.0
    measured_time = 0.0
    measured_tokens = 0

    if torch.cuda.is_available() and device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for step in tqdm(range(steps), desc=f"Profiling {cfg.experiment_name}"):
        timer = Timer(sync_cuda=(device == "cuda"))
        with timer:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if device == "cuda":
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            if device == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    out = model(**batch)
                    loss = out["loss"]
            else:
                out = model(**batch)
                loss = out["loss"]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step_time = float(timer)
        cumulative_time += step_time

        if step >= warmup:
            measured_time += step_time
            measured_tokens += int(batch["input_ids"].numel())

    avg_tokens_per_sec = (measured_tokens / max(measured_time, 1e-6)) if measured_time > 0 else 0.0
    peak_mb = peak_mem_mb()

    # Final eval metrics at the end of profiling run
    val_loss, val_ppl, val_acc = _evaluate_metrics(model, val_loader, device, cfg, use_amp=False)
    return avg_tokens_per_sec, peak_mb, cumulative_time, val_loss, val_ppl, val_acc


# ---------- Public API (used by cli.py) ----------

def run_sweep(base_cfg: DotDict, sweep_cfg: DotDict, profile_type: str, use_amp: bool = False):
    if "profile" not in sweep_cfg or profile_type not in sweep_cfg["profile"]:
        raise ValueError(f"No profile.{profile_type} section found in experiments.yaml")

    spec = sweep_cfg["profile"][profile_type]
    try:
        values = list(spec["values"])
        param_path = str(spec["param"])
    except Exception as e:
        raise ValueError(f"Malformed profile.{profile_type} spec: {spec}") from e

    steps = int(sweep_cfg.get("run", {}).get("steps", 200))

    # AMP PROFILE
    if profile_type == "amp":
        for v in values:
            cfg = copy.deepcopy(base_cfg)
            set_value_at_path(cfg, param_path, v)
            set_value_at_path(cfg, "hardware.amp", bool(v))

            out_csv = _logs_dir(base_cfg) / f"{base_cfg.experiment_name}_profile_amp_{'on' if v else 'off'}.csv"
            print(f"==> AMP profile ({'on' if v else 'off'}), steps={steps}, logging -> {out_csv.name}")
            _train_like_with_amp(cfg, steps=steps, amp_flag=bool(v), out_csv=out_csv)
        return

    # OTHER PROFILES (seq_len / batch_size)
    amp_suffix = "_amp" if use_amp else ""
    out_csv = _logs_dir(base_cfg) / f"{base_cfg.experiment_name}_sweep_{profile_type}{amp_suffix}.csv"
    logger = CSVLogger(
        out_csv,
        fieldnames=[
            "experiment_name",
            "param",
            "value",
            "cumulative_time",
            "tokens_per_sec",
            "peak_mem_mb",
            "loss",
            "ppl",
            "accuracy",
        ],
    )

    print(f"==> Running sweep '{profile_type}' over {values} (param: {param_path}, steps={steps}, AMP: {use_amp})")
    for v in values:
        cfg = copy.deepcopy(base_cfg)
        set_value_at_path(cfg, param_path, v)

        tps, peak_mb, cum_time, val_loss, val_ppl, val_acc = _profile_run_real(cfg, steps=steps, amp_flag=use_amp)
        logger.log(
            {
                "experiment_name": f"{base_cfg.experiment_name}",
                "param": param_path,
                "value": v,
                "cumulative_time": cum_time,
                "tokens_per_sec": tps,
                "peak_mem_mb": peak_mb,
                "loss": float(val_loss),
                "ppl": float(val_ppl),
                "accuracy": float(val_acc),
            }
        )
        print(f"   value={v} -> {tps:.1f} tok/s, {peak_mb:.2f} MB, val_loss={val_loss:.4f}, acc={val_acc:.4f}, ppl={val_ppl:.2f}, time={cum_time:.1f}s")

    print(f"[sweep] wrote: {out_csv}")


# ---------- Plotting ----------

def plot_gpu_vs_seq_len(args, outdir):
    if not args.gpu_seq_len_file:
        return
    try:
        df = pd.read_csv(args.gpu_seq_len_file)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="value", y="peak_mem_mb", marker="o")
        plt.title("GPU Memory vs. Sequence Length")
        plt.xlabel("Sequence Length")
        plt.ylabel("Peak Memory (MB)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(outdir / "memory_vs_seq_len.png", dpi=300)
        plt.close()
        print(f"Saved memory_vs_seq_len.png")
    except FileNotFoundError:
        print(f"Warning: File not found for GPU vs. Seq Len plot: {args.gpu_seq_len_file}")


def plot_throughput_vs_batch_size(args, outdir):
    if not (args.batch_noamp and args.batch_amp):
        return
    try:
        df_noamp = pd.read_csv(args.batch_noamp)
        df_noamp["precision"] = "No AMP"
        df_amp = pd.read_csv(args.batch_amp)
        df_amp["precision"] = "AMP"
        df = pd.concat([df_noamp, df_amp])

        # GPU Memory Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="value", y="peak_mem_mb", hue="precision", marker="o")
        plt.title("GPU Memory vs. Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Peak Memory (MB)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title="Precision")
        plt.tight_layout()
        plt.savefig(outdir / "memory_vs_batch_size.png", dpi=300)
        plt.close()
        print(f"Saved memory_vs_batch_size.png")

        # Throughput Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="value", y="tokens_per_sec", hue="precision", marker="o")
        plt.title("Throughput vs. Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (tokens/sec)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title="Precision")
        plt.tight_layout()
        plt.savefig(outdir / "throughput_vs_batch_size.png", dpi=300)
        plt.close()
        print(f"Saved throughput_vs_batch_size.png")
    except FileNotFoundError as e:
        print(f"Warning: File not found for throughput vs. batch size plot: {e.filename}")

def plot_baseline_training(args, outdir):
    if not args.baseline_file:
        return
    try:
        df = pd.read_csv(args.baseline_file)
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=df, x="step", y="ppl", marker="o", label="Perplexity")
        plt.title("Baseline Model Training Progress")
        plt.xlabel("Step")
        plt.ylabel("Validation Perplexity")
        
        for i in range(df.shape[0]):
            ax.text(df.step[i], df.ppl[i], f' acc: {df.accuracy[i]:.2f}', 
                    horizontalalignment='left', size='small', color='black')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "baseline_training_progress.png", dpi=300)
        plt.close()
        print("Saved baseline_training_progress.png")
    except FileNotFoundError:
        print(f"Warning: File not found for baseline training plot: {args.baseline_file}")


def plot_optimization_comparison(args, outdir):
    files = {
        "Baseline": args.baseline_file,
        "Grad Ckpt": args.grad_ckpt_file,
        "Flash Attn": args.flash_attention_file,
    }
    if not all(files.values()):
        return

    dfs = {}
    for name, path in files.items():
        try:
            dfs[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: File not found for optimization comparison: {path}")
            return

    # --- Baseline vs. Grad Ckpt ---
    df_base = dfs["Baseline"].copy()
    df_base["method"] = "Baseline"
    df_ckpt = dfs["Grad Ckpt"].copy()
    df_ckpt["method"] = "Grad Ckpt"
    df_comp_ckpt = pd.concat([df_base, df_ckpt])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left plot: PPL vs Step
    sns.lineplot(data=df_comp_ckpt, x="step", y="ppl", hue="method", ax=axes[0], marker="o")
    for method in ["Baseline", "Grad Ckpt"]:
        df_method = df_comp_ckpt[df_comp_ckpt.method == method]
        for i in range(df_method.shape[0]):
            axes[0].text(df_method.step.iloc[i], df_method.ppl.iloc[i], f' {df_method.accuracy.iloc[i]:.2f}', 
                         horizontalalignment='left', size='small', color='black')
    axes[0].set_title("Perplexity vs. Step (Baseline vs. Grad Ckpt)")
    axes[0].set_ylabel("Validation Perplexity (lower is better)")
    axes[0].set_xlabel("Step (Accuracy annotated)")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend(title="Method")

    # Right plot: Bar chart for Memory and Throughput
    bar_data_ckpt = pd.DataFrame([
        {"method": "Baseline", "metric": "Memory (MB)", "value": df_base['peak_mem_mb'].iloc[-1]},
        {"method": "Baseline", "metric": "Throughput (tok/s)", "value": df_base['tokens_per_sec'].iloc[-1]},
        {"method": "Grad Ckpt", "metric": "Memory (MB)", "value": df_ckpt['peak_mem_mb'].iloc[-1]},
        {"method": "Grad Ckpt", "metric": "Throughput (tok/s)", "value": df_ckpt['tokens_per_sec'].iloc[-1]},
    ])
    sns.barplot(data=bar_data_ckpt, x="method", y="value", hue="metric", ax=axes[1])
    axes[1].set_title("Resource Usage (Baseline vs. Grad Ckpt)")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Method")
    axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    axes[1].legend(title="Metric")

    fig.tight_layout()
    plt.savefig(outdir / "comparison_grad_ckpt.png", dpi=300)
    plt.close()
    print(f"Saved comparison_grad_ckpt.png")

    # --- Baseline vs. Flash Attention ---
    df_flash = dfs["Flash Attn"].copy()
    df_flash["method"] = "Flash Attn"
    df_comp_flash = pd.concat([df_base, df_flash])
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left plot: PPL vs Step
    sns.lineplot(data=df_comp_flash, x="step", y="ppl", hue="method", ax=axes[0], marker="o")
    for method in ["Baseline", "Flash Attn"]:
        df_method = df_comp_flash[df_comp_flash.method == method]
        for i in range(df_method.shape[0]):
            axes[0].text(df_method.step.iloc[i], df_method.ppl.iloc[i], f' {df_method.accuracy.iloc[i]:.2f}', 
                         horizontalalignment='left', size='small', color='black')
    axes[0].set_title("Perplexity vs. Step (Baseline vs. Flash Attention)")
    axes[0].set_ylabel("Validation Perplexity (lower is better)")
    axes[0].set_xlabel("Step (Accuracy annotated)")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].legend(title="Method")

    # Right plot: Bar chart for Memory and Throughput
    bar_data_flash = pd.DataFrame([
        {"method": "Baseline", "metric": "Memory (MB)", "value": df_base['peak_mem_mb'].iloc[-1]},
        {"method": "Baseline", "metric": "Throughput (tok/s)", "value": df_base['tokens_per_sec'].iloc[-1]},
        {"method": "Flash Attn", "metric": "Memory (MB)", "value": df_flash['peak_mem_mb'].iloc[-1]},
        {"method": "Flash Attn", "metric": "Throughput (tok/s)", "value": df_flash['tokens_per_sec'].iloc[-1]},
    ])
    sns.barplot(data=bar_data_flash, x="method", y="value", hue="metric", ax=axes[1])
    axes[1].set_title("Resource Usage (Baseline vs. Flash Attention)")
    axes[1].set_ylabel("Value")
    axes[1].set_xlabel("Method")
    axes[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    axes[1].legend(title="Metric")

    fig.tight_layout()
    plt.savefig(outdir / "comparison_flash_attention.png", dpi=300)
    plt.close()
    print(f"Saved comparison_flash_attention.png")


def generate_plots(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_gpu_vs_seq_len(args, outdir)
    plot_throughput_vs_batch_size(args, outdir)
    plot_baseline_training(args, outdir)
    plot_optimization_comparison(args, outdir)

    print(f"\nPlots saved to {outdir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results/plots", help="Directory to save plots")
    parser.add_argument("--gpu-seq-len-file", type=str, help="CSV for GPU memory vs. sequence length plot")
    parser.add_argument("--batch-noamp", type=str, help="CSV for batch size sweep without AMP")
    parser.add_argument("--batch-amp", type=str, help="CSV for batch size sweep with AMP")
    parser.add_argument("--baseline-file", type=str, help="CSV for baseline training run")
    parser.add_argument("--grad-ckpt-file", type=str, help="CSV for gradient checkpointing run")
    parser.add_argument("--flash-attention-file", type=str, help="CSV for flash attention run")
    args = parser.parse_args()

    generate_plots(args)
