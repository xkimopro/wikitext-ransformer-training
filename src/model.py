import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


# -----------------------------
# Baseline config (unchanged)
# -----------------------------
class VanillaTransformerConfig(PretrainedConfig):
    model_type = "vanilla"

    def __init__(
        self,
        vocab_size=50257,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        max_position_embeddings=512,
        pad_token_id=None,
        gradient_checkpointing=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(pad_token_id=pad_token_id, **kwargs)


# -----------------------------
# Baseline model (unchanged)
# -----------------------------
class VanillaTransformer(PreTrainedModel):
    """
    Simple causal LM built only from YAML config (baseline).
    """
    def __init__(self, config: VanillaTransformerConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=4 * config.n_embd,
                dropout=config.dropout,
                batch_first=True,
            ) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Precompute full causal mask (boolean upper-triangular). Slice per forward length.
        self.register_buffer(
            "causal_mask_bool",
            torch.triu(
                torch.ones(config.max_position_embeddings, config.max_position_embeddings, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def _additive_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((T, T), dtype=torch.float32, device=device)
        mask = mask.masked_fill(self.causal_mask_bool[:T, :T], float("-inf"))
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Next-token objective: shift internally so callers pass only input_ids (+ optional labels)
        input_ids_shifted = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous() if labels is None else labels

        B, T = input_ids_shifted.size()
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids_shifted.device)
        pos = pos.unsqueeze(0).expand(B, -1)

        tok_emb = self.embed_tokens(input_ids_shifted)   # [B, T, C]
        pos_emb = self.pos_embed(pos)                    # [B, T, C]
        x = self.drop(tok_emb + pos_emb)

        add_mask = self._additive_causal_mask(T, x.device)
        src_key_padding_mask = (attention_mask[:, :-1] == 0) if attention_mask is not None else None

        use_checkpointing = self.config.gradient_checkpointing and self.training
        for block in self.blocks:
            if use_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, add_mask, src_key_padding_mask, use_reentrant=False
                )
            else:
                x = block(x, src_mask=add_mask, src_key_padding_mask=src_key_padding_mask)

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, V]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.contiguous().view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# -----------------------------
# Builder
# -----------------------------
def build_model_from_config(cfg):
    vt_cfg = VanillaTransformerConfig(
        vocab_size=int(cfg.model.vocab_size),
        n_layer=int(cfg.model.n_layer),
        n_head=int(cfg.model.n_head),
        n_embd=int(cfg.model.n_embd),
        dropout=float(cfg.model.dropout),
        max_position_embeddings=int(cfg.data.seq_len),
        gradient_checkpointing=bool(getattr(cfg.model, "gradient_checkpointing", False)),
    )

    model = VanillaTransformer(vt_cfg)
    # HACK: Monkey-patch the forward pass of the encoder layer to always be causal.
    # This is needed because checkpoint() does not support the is_causal kwarg.
    # The "correct" way to do this would be to subclass TransformerEncoderLayer.
    if cfg.hardware.flash_attention:
        for block in model.blocks:
            # block.forward is a bound method, so we need to patch the class's method
            # but that would affect all instances. Instead, we can replace the bound method
            # with a new function that calls the original with the desired arguments.
            original_forward = block.forward
            def _causal_forward(x, src_mask=None, src_key_padding_mask=None):
                # We ignore src_mask and enforce is_causal=True
                return original_forward(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            block.forward = _causal_forward

    # Optional compile
    if bool(getattr(cfg.hardware, "compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model
