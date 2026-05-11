# Hölder-MPO

Reference implementation of **Hölder-MPO** — an RL fine-tuning objective for
math-reasoning LLMs that replaces the GRPO importance ratio with a
**Hölder $p$-mean ratio** over the response. Built on
[`oat`](https://github.com/sail-sg/oat) and the vendored
[`understand-r1-zero`](https://github.com/sail-sg/understand-r1-zero) math
pipeline (under `understand_r1_zero_main/`).

> The agent / ALFWorld variant lives on the [`agentic`](../../tree/agentic) branch.

## What's in the loss

Two aggregation variants (set via `--critic_type_modify`):

| value          | aggregation                                                       |
|----------------|-------------------------------------------------------------------|
| `holder`       | **sequence-level** Hölder $p$-mean of token ratios (paper default) |
| `holder_token` | **token-level** Hölder $p$-mean (ablation)                         |

The Hölder exponent $p$ supports a per-step schedule on
$[\,p_{\min},\,p_{\max}\,]$:

| `--holder_p_schedule` | curve over `holder_p_schedule_steps`                          |
|-----------------------|----------------------------------------------------------------|
| `constant`            | $p \equiv$ `--holder_p`                                       |
| `linear` / `linear_dec` | linear up / linear down                                     |
| `sin` / `cos`         | sinusoidal between $p_{\min}$ and $p_{\max}$                  |
| `poly` / `poly_dec`   | $p_{\min} + (p_{\max}-p_{\min})\cdot \text{progress}^n$, $n =$ `--holder_p_power` |
| `cubic` / `cubic_dec` | cubic                                                          |

## Setup

```
conda create -n holder python==3.10
conda activate holder
pip install vllm==0.8.4 && pip install oat-llm==0.1.3.post1
cd understand_r1_zero_main && pip install -e . && cd ..
```

Weights & Biases is opt-in: `export WANDB_API_KEY=...` and set `USE_WB=1` when
launching the script.

## Train

```
bash scripts/qwen2.5-math-7b-holder.sh
```

Key knobs (edit at the top of the script):

| variable                  | default       | meaning                              |
|---------------------------|---------------|--------------------------------------|
| `HOLDER_P`                | `3.0`         | initial / constant $p$               |
| `HOLDER_P_SCHEDULE`       | `linear_dec`  | one of the schedules above           |
| `HOLDER_P_MIN` / `_MAX`   | `-2.0` / `2.0`| bounds of the schedule               |
| `HOLDER_P_SCHEDULE_STEPS` | `500`         | steps the schedule spans             |
| `HOLDER_P_POWER`          | `2.0`         | $n$ for `poly` / `poly_dec`          |
| `CLIPRANGE`               | `0.2`         | PPO-style clip                       |
| `MODEL_NAME`              | `deepseek-r1-distill-qwen-7b` | base policy            |
| `N_GPU` / `N_SAMPLE`      | `4` / `8`     | parallelism / responses per prompt   |

## Evaluate

Edit the `model=...` path in `scripts/eval.sh`, then:

```
bash scripts/eval.sh
```

Default suite (under `datasets/evaluation_suite_v2/`): AIME24, AIME25, AMC,
MATH500, Minerva, OlympiadBench.

## Layout

```
.
├── train_zero_math_holder.py       # oat PPOLearner subclass — Hölder / Hölder-token loss
├── scripts/
│   ├── qwen2.5-math-7b-holder.sh   # training launcher
│   └── eval.sh                     # offline eval
├── utils/evaluation/               # vLLM-based evaluator + math grader
├── datasets/evaluation_suite{,_v2}/  # eval prompts
└── understand_r1_zero_main/        # vendored math grader & data loader
```

## License

Apache-2.0 (see `LICENSE`). The vendored `understand_r1_zero_main/` retains its
upstream Apache-2.0 license.
