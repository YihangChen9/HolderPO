# Hölder-PO on ALFWorld

This subtree is a fork of `verl-agent` extended with the Hölder-PO loss for
agent-style training. The upstream `README.md` documents environment / vLLM
setup; the additions for Hölder-PO are minimal and listed here.

## Entry point

```
bash examples/holder_po_trainer/run_alfworld.sh <engine> <variant>
```

- `<engine>` — `vllm` or `sglang`
- `<variant>` — `sequence` (`holder_seq`) or `token` (`holder_token`)

Hölder-$p$ is configured via env vars:

| variable                  | meaning                                                                                    |
|---------------------------|--------------------------------------------------------------------------------------------|
| `HOLDER_P`                | constant $p$ (used when schedule is constant)                                              |
| `HOLDER_P_SCHEDULE`       | `constant`, `linear`, `linear_dec`, `sin`, `cos`, `quad`, `quad_dec`, `cubic`, `cubic_dec` |
| `HOLDER_P_MIN` / `_MAX`   | schedule bounds                                                                            |
| `HOLDER_P_SCHEDULE_STEPS` | steps the schedule spans                                                                   |
| `HOLDER_P_POWER`          | exponent $n$ for `quad` / `quad_dec`                                                       |

## Where the loss is implemented

- `verl/trainer/ppo/core_algos.py` — Hölder-$p$ ratio + sequence/token aggregation
- `verl/workers/actor/dp_actor.py` — calls into the loss

Search the tree for `holder` to see all touched files; no upstream verl files
were renamed.
