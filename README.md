# Hölder-MPO

RL fine-tuning of math-reasoning LLMs with the Hölder-MPO objective. Built on the
[`oat`](https://github.com/sail-sg/oat) RL library and the vendored
[`understand-r1-zero`](https://github.com/sail-sg/understand-r1-zero) math
pipeline (under `understand_r1_zero_main/`).

> The agent / ALFWorld variant lives on the [`agentic`](../../tree/agentic) branch.

## Setup

```
conda create -n holder python==3.10
conda activate holder
pip install vllm==0.8.4 && pip install oat-llm==0.1.3.post1
cd understand_r1_zero_main && pip install -e . && cd ..
```

Optional: to log to Weights & Biases, `export WANDB_API_KEY=...` and set
`USE_WB=1` when launching scripts.

## Train

```
bash scripts/qwen2.5-math-7b-holder.sh
```

`--critic_type_modify` selects the loss variant (`holder` for the sequence-level
Hölder $p$-mean ratio, `holder_token` for the token-level variant);
`--holder_p`, `--holder_p_schedule`, `--holder_p_min/_max`,
`--holder_p_schedule_steps`, `--holder_p_power` control the $p$ schedule.

## Evaluate

Edit the `model=...` path in `scripts/eval.sh`, then:

```
bash scripts/eval.sh
```

The default suite (under `datasets/evaluation_suite_v2/`) is
AIME24, AIME25 (pass@8), AMC, MATH500, Minerva, OlympiadBench.

## Layout

```
.
├── train_zero_math_holder.py       # Hölder / Hölder-token trainer (oat PPOLearner subclass)
├── scripts/                        # launcher + eval
├── utils/evaluation/               # offline eval suite
├── datasets/evaluation_suite{,_v2}/  # eval prompts
└── understand_r1_zero_main/        # vendored math grader & data loader
```

## License

Apache-2.0 (see `LICENSE`). The vendored `understand_r1_zero_main/` retains its
upstream Apache-2.0 license.
