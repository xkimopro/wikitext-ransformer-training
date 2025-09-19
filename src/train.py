import math
import contextlib
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from .utils import CSVLogger, Timer, peak_mem_mb, count_params
from .dataset import get_dataloaders
from .model import build_model_from_config


class CUDAPrefetcher:
    def __init__(self, loader, device: str, enabled: bool):
        self.loader = loader
        self.device = device
        self.enabled = enabled and (device == "cuda") and torch.cuda.is_available()
        self._iter = iter(loader)
        self.stream = torch.cuda.Stream() if self.enabled else None
        self._next = None

    def _move_to_device(self, batch):
        if not self.enabled:
            return {k: v.to(self.device, non_blocking=False) for k, v in batch.items()}
        moved = {}
        with torch.cuda.stream(self.stream):
            for k, v in batch.items():
                moved[k] = v.to(self.device, non_blocking=True)
        return moved

    def _prefetch(self):
        try:
            nxt = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            nxt = next(self._iter)
        self._next = self._move_to_device(nxt)

    def __iter__(self):
        if self._next is None:
            self._prefetch()
        return self

    def __next__(self):
        if self._next is None:
            self._prefetch()
        if self.enabled:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self._next
        self._prefetch()
        return batch


def _sdpa_context_if_any(cfg):
    if not (torch.cuda.is_available() and bool(getattr(cfg.hardware, "flash_attention", False))):
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    except (ImportError, AttributeError):
        # Fallback for older PyTorch versions
        try:
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            return contextlib.nullcontext()


def _apply_hardware_knobs(cfg):
    allow_amp = bool(getattr(cfg.hardware, "allow_amp_in_train", False))
    if not allow_amp:
        assert not bool(getattr(cfg.hardware, "amp", False)), \
            "Baseline training forbids AMP. Use the fused-attention make target if you want AMP."

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


def _build_optimizer(model, cfg):
    assert cfg.optim.name.lower() == "adamw", "Only AdamW is supported by this baseline (from YAML)."
    # IMPORTANT: only optimize trainable params (works for baseline)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(
        trainable_params,
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=(float(cfg.optim.adam_beta1), float(cfg.optim.adam_beta2)),
        eps=float(cfg.optim.adam_eps),
        foreach=bool(cfg.optim.foreach),
        fused=bool(cfg.optim.fused),  # stays False in your YAML
    )


def _build_scheduler(optimizer, cfg):
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


def train(cfg):
    device = cfg.hardware.device
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, switching to CPU.")
        device = "cpu"

    _apply_hardware_knobs(cfg)
    print(f"Using device: {device}")

    model = build_model_from_config(cfg).to(device)
    train_loader, val_loader, _ = get_dataloaders(cfg)

    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    allow_amp = bool(getattr(cfg.hardware, "allow_amp_in_train", False))
    use_amp = bool(getattr(cfg.hardware, "amp", False) and allow_amp and device == "cuda")
    use_bf16 = bool(getattr(cfg.hardware, "bf16", False)) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device == "cuda" else None

    logger = CSVLogger(
        cfg.logging.csv_path,
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

    max_steps = int(cfg.train.max_steps)
    grad_accum = int(cfg.train.grad_accum)
    tps_alpha = float(cfg.train.tps_ema_alpha)
    rolling_tps = 0.0
    total_params = count_params(model)

    prefetch_enabled = bool(cfg.loader.prefetch.enable)
    train_iter = iter(CUDAPrefetcher(train_loader, device, enabled=prefetch_enabled))

    est_train_examples = max(1, int(cfg.data.estimated_train_examples))
    processed_examples = 0
    sync_cuda = bool(cfg.logging.sync_cuda_timer)
    cumulative_time = 0.0

    model.train()
    pbar = tqdm(range(1, max_steps + 1), desc="Training", dynamic_ncols=True)

    with _sdpa_context_if_any(cfg):
        for step in pbar:
            try:
                with Timer(sync_cuda=sync_cuda) as timer:
                    for _ in range(grad_accum):
                        batch = next(train_iter)

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

                    if cfg.train.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.clip_grad_norm))

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
                    pbar.set_description(f"Training ({percent_complete:.2f}% consumed)")
                    pbar.set_postfix(loss=f"{(loss.item()*grad_accum):.4f}", tps=f"{rolling_tps:.1f}")

                if step % int(cfg.train.eval_every) == 0:
                    val_loss, val_ppl, val_acc = evaluate_metrics(model, val_loader, device, cfg, use_amp=use_amp, amp_dtype=amp_dtype)
                    logger.log(
                        {
                            "experiment_name": cfg.experiment_name,
                            "mode": "eval",
                            "step": step,
                            "cumulative_time": cumulative_time,
                            "tokens_per_sec": rolling_tps,
                            "peak_mem_mb": peak_mem_mb(),
                            "loss": float(loss.item() * grad_accum),   # training loss (kept same semantics)
                            "ppl": float(val_ppl),                     # validation ppl
                            "accuracy": float(val_acc),                # validation top-1 token accuracy
                            "amp": bool(use_amp),
                            "ddp": bool(cfg.hardware.ddp),
                            "params": total_params,
                        }
                    )
                    model.train()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nCaught OOM at step {step}. Exiting.")
                    break
                else:
                    raise

    pbar.close()
    print("Training finished.")


@torch.no_grad()
def evaluate_metrics(model, val_loader, device, cfg, use_amp=False, amp_dtype=torch.float16):
    """
    Returns (mean_val_loss, val_ppl, val_token_top1_accuracy).
    Accuracy masks out padding if attention_mask is provided.
    """
    model.eval()
    losses = []
    correct = 0
    total = 0
    steps = 0
    max_eval_steps = int(cfg.train.eval_max_steps)

    with _sdpa_context_if_any(cfg):
        for batch in tqdm(val_loader, desc="Evaluating", dynamic_ncols=True):
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
            preds = logits.argmax(dim=-1)           # [B, T-1]

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
