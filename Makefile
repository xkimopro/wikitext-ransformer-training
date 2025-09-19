PY=python
PIP=pip

# Cache roots (override by env if desired)
export HF_HOME?=/home1/10899/kimopro/WORK/.cache/huggingface
export HF_DATASETS_CACHE?=/home1/10899/kimopro/WORK/.cache/hf_datasets
export CUDA_HOME?=/home1/10899/kimopro/WORK/miniconda3/envs/fndml

install:
	$(PIP) install -U pip
	$(PIP) install numpy
	$(PIP) install -r requirements.txt

train:
	$(PY) -m src.cli train --config configs/base.yaml

train-larger:
	$(PY) -m src.cli train --config configs/base.yaml --override \
		model.n_layer=24 \
		model.n_embd=1536 

train-flash:
	$(PY) -m src.cli train --config configs/base.yaml --override \
		hardware.flash_attention=true \
		hardware.allow_amp_in_train=true \
		hardware.amp=true \
		hardware.bf16=true

train-ckpt:
	$(PY) -m src.cli train --config configs/base.yaml --override model.gradient_checkpointing=true


profile-len:
	$(PY) -m src.cli profile --config configs/base.yaml --sweep configs/experiments.yaml --profile_type seq_len

profile-batch-noamp:
	$(PY) -m src.cli profile --config configs/base.yaml --sweep configs/experiments.yaml --profile_type batch_size

profile-batch-amp:
	$(PY) -m src.cli profile --config configs/base.yaml --sweep configs/experiments.yaml --profile_type batch_size --amp

plots:
	$(PY) -m src.cli plot --outdir results/plots \
		--gpu-seq-len-file results/logs/baseline_sweep_seq_len.csv \
		--batch-noamp results/logs/baseline_sweep_batch_size_noamp.csv \
		--batch-amp results/logs/baseline_sweep_batch_size_amp.csv \
		--baseline-file results/logs/baseline_seq512_batch8.csv \
		--grad-ckpt-file results/logs/ckpt_seq512_batch8.csv \
		--flash-attention-file results/logs/flash_seq512_batch8.csv

clean:
	rm -rf results/logs/* results/plots/*
