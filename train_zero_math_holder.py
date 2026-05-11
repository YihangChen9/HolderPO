# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from multiprocessing import Pool, TimeoutError
from typing import Any, List, Literal, Tuple

import numpy as np
import torch
import torch.distributed as dist
import tree
from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TrajectoryData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import entropy_from_logits, masked_mean, masked_sum, masked_whiten
from oat.utils.ops import masked_mean, masked_sum
from torch.utils.data import DataLoader

from datasets import load_from_disk
from understand_r1_zero_main.understand_r1_zero.math_grader import (answer_tag_reward_fn,
                                            boxed_reward_fn)

import os
import gc
import json
import vllm
import copy
from datetime import datetime
from collections import defaultdict
# Don't disable wandb here - let it be controlled by --use-wb argument
# os.environ["WANDB_MODE"] = "disabled"

"""
1. To do RL from base models, we use proper prompt template to make the base model answer questions.
"""

def debug_func(cond=True):
    if cond:
        print("debugpy connected!")
        import time
        import debugpy
        try:
            debugpy.listen(("0.0.0.0", 5678))
            print("Waiting for debugger attach...")
            debugpy.wait_for_client()
            debugpy.breakpoint()
        except:
            time.sleep(99999)


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )


def apply_no_template(question: str):
    return question


TEMPLATE_FACTORY = {
    "qwen_math": apply_qwen_math_template,
    "r1": apply_r1_template,
    "no": apply_no_template,
}


"""
2. To train reasoning models that solve math questions, we need to define an oracle (environment) that provides rule-based verification rewards.
We instantiate the oracle based on Oat's OracleBase and implement the grading logic.
"""


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the math answer grading."""

    def __init__(self, template, verifier_version) -> None:
        super().__init__()
        if template == "r1":
            math_reward_fn = answer_tag_reward_fn
        else:
            math_reward_fn = boxed_reward_fn
        self.math_reward_fn = functools.partial(
            math_reward_fn, fast=verifier_version == "fast"
        )
        # Process pool is used to enable the timeout mechanism for answer grading in our distributed training setup.
        self.mp_pool = Pool(2)

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        # Parameters used by Oat when using model-based reward, here we don't need.
        del inputs, batch_size

        rewards = []
        infos = []
        for resp, ref in zip(responses, references):
            res = self.mp_pool.apply_async(self.math_reward_fn, (resp, ref))
            try:
                info, r = res.get(timeout=1)
                rewards.append(r)
                infos.append(info)
            except TimeoutError:
                rewards.append(0.0)
                infos.append({"formatted": False})

        return torch.tensor(rewards), infos

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


"""
2. Define extra arguments needed besides Oat's PPOArgs, mainly about choosing the prompt template.
"""


@dataclass
class ZeroMathArgs(PPOArgs):
    # Template.
    prompt_template: Literal["qwen_math", "no", "r1"] = field(default="qwen_math")
    # Evaluation benchmarks used.
    test_split: str = "all"  # Use "aime,math" to only evaluate on selected benchmarks.
    # Verifier.
    verifier_version: Literal["fast", "math_verify"] = field(default="fast")
    critic_type_modify: str = ""
    critic_type_modify_advantage: str = ""
    do_record_learning_step: str = ""
    step_method_modify: str = ""
    holder_p: float = 1.0  # p-norm parameter for Holder-MPO ratio calculation
    holder_p_schedule: Literal[
        "constant",
        "linear", "linear_dec",
        "sin", "cos",
        "poly", "poly_dec",
        "cubic", "cubic_dec",
    ] = field(default="constant")
    holder_p_min: float = -2.0
    holder_p_max: float = 2.0
    holder_p_schedule_steps: int = 1000
    holder_p_power: float = 2.0  # exponent n for poly / poly_dec schedules
    # Per-benchmark eval sampling override (e.g. "aime25" -> pass@8).
    # Comma-separated benchmark names; empty disables override.
    eval_pass_n_tasks: str = ""
    eval_pass_n: int = 8
    eval_pass_temperature: float = 0.7
    eval_pass_top_p: float = 1.0


"""
3. Instantiate the actor based on Oat's PPOActor, which controls the reasoning trace generation (`self.sampling_params`) and the rewarding (`self.oracle`).
"""


class ZeroMathActor(PPOActor):
    def init(self, actor_id, save_path):
        super().init(actor_id, save_path)

        self.oracle = MATHOracle(
            template=self.args.prompt_template,
            verifier_version=self.args.verifier_version,
        )

        if self.args.prompt_template in ["qwen_math", "no"]:
            # These two templates are better used for Qwen models, which can themselves stop generation. Hence we unset all external stopping conditions.
            self.sampling_params.stop = None
            self.sampling_params.stop_token_ids = None
            self.eval_sampling_params.stop = None
            self.eval_sampling_params.stop_token_ids = None
        elif self.args.prompt_template == "r1":
            # Let's stop when the model completes its answer.
            self.sampling_params.stop = ["</answer>"]
            self.sampling_params.include_stop_str_in_output = True
            self.eval_sampling_params.stop = ["</answer>"]
            self.eval_sampling_params.include_stop_str_in_output = True

        # Cache the configured defaults so the learner can swap eval sampling
        # params per benchmark and restore them afterwards.
        self._eval_sp_default_n = self.eval_sampling_params.n
        self._eval_sp_default_temperature = self.eval_sampling_params.temperature
        self._eval_sp_default_top_p = self.eval_sampling_params.top_p

    def set_eval_sampling_params(self, n=None, temperature=None, top_p=None):
        """Mutate eval_sampling_params in place. Pass None to leave a field unchanged.

        Use ``reset_eval_sampling_params`` to restore the configured defaults.
        Stop-tokens / max_tokens / top_k are intentionally untouched here so the
        prompt-template-specific stop config from ``init`` survives.
        """
        if n is not None:
            self.eval_sampling_params.n = int(n)
        if temperature is not None:
            self.eval_sampling_params.temperature = float(temperature)
        if top_p is not None:
            self.eval_sampling_params.top_p = float(top_p)
        return (
            self.eval_sampling_params.n,
            self.eval_sampling_params.temperature,
            self.eval_sampling_params.top_p,
        )

    def reset_eval_sampling_params(self):
        return self.set_eval_sampling_params(
            n=self._eval_sp_default_n,
            temperature=self._eval_sp_default_temperature,
            top_p=self._eval_sp_default_top_p,
        )

    def step(
        self,
        prompts: List[str],
        formatted_prompts: List[str],
        references: List[str] = None,
    ) -> List[TrajectoryData]:
        """Main logic for the actor to generate trajectories (reasoning traces)."""
        assert not self.eval_mode
        info = {}
        logging.info(f"actor start")

        # step 1. generate
        st = time.time()
        outputs = self.generate(formatted_prompts, self.sampling_params)

        candidates = []
        prompt_token_ids = []
        no_eos = []
        response_ids = []
        response_logprobs = []
        resp_lens = []
        for i in range(len(outputs)):
            # for each prompt
            prompt_token_ids.append(outputs[i].prompt_token_ids)
            candidates.append([])
            response_logprobs.append([])
            response_ids.append([])
            for k in range(self.sampling_params.n):
                # for each response
                candidates[i].append(outputs[i].outputs[k].text)
                no_eos.append(outputs[i].outputs[k].finish_reason == "length")
                token_ids = outputs[i].outputs[k].token_ids
                logps = outputs[i].outputs[k].logprobs
                logps = [item[token_ids[i]].logprob for i, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))

        info["actor/generate_time"] = time.time() - st

        # step 2. verify
        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in prompts
                )
            ),
            tree.flatten(candidates),
            list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, self.sampling_params.n) for x in references
                )
            ),
        )

        info["actor/verify_time"] = time.time() - st
        logging.info(f"actor reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            candidates_per_prompt = candidates[i]
            for j in range(len(candidates_per_prompt)):
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    # Set zero reward for truncated outputs.
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(
                    TrajectoryData(
                        prompt=prompt,
                        prompt_ids=prompt_token_ids[i],
                        response=candidates_per_prompt[j],
                        response_ids=response_ids[i][j],
                        response_logprobs=response_logprobs[i][j],
                        rewards=dense_rewards,
                        loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                        info=info,
                    )
                )
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        handle = self.ipc_client.serialize_ipc(trajectory_data)
        return handle


"""
4. Instantiate the learner based on PPOLearner. Here we adapt the `evaluate` logic to run multiple math benchmarks.
"""


class ZeroMathLearner(PPOLearner):
    def _init(self, args: ZeroMathArgs, actors: List[ActorBase]) -> None:
        super()._init(args, actors)
        self.eval_dataset_dict = load_from_disk(args.eval_data)  # TODO: get fro HF.
        if args.test_split != "all":
            self.eval_dataset_dict = {
                k: v for k, v in self.eval_dataset_dict.items() if k in args.test_split
            }
        self.args = args
        # Dr. GRPO Modification 1: Remove length bias by using masked_sum with a constant normalizer:
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo"
            else masked_mean
        )

        if self.strategy.is_rank_0():
            # 保存 logs_dict 到 logs/{datetime}.log
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs("logs", exist_ok=True)
            log_path = os.path.join("logs", f"{timestamp}.log")
            self.log_path = log_path

    def eval_and_log(self, train_info, eval=False, save=False):
        best_metric = getattr(self, "metrics_best", {}).get("eval/average/accuracy", -1)
        update_best_metric = False

        # eval
        eval_info = {}
        if (self.args.eval_steps > 0 and eval) or self._should_do(self.args.eval_steps):
            eval_info = self.evaluate(self.eval_prompts_dataloader, self.steps)

            # the eval_info is broasted into all gpus, so no need to gather
            if "eval/average/accuracy" in eval_info and eval_info["eval/average/accuracy"] > best_metric:
                update_best_metric = True
                best_metric = eval_info["eval/average/accuracy"]

        # save
        if (self.args.save_steps > 0 and save) or (
            self.steps > 0
            and self._should_do(self.args.save_steps)
            and self.steps >= self.args.save_from
        ):
            self.strategy.save_model(
                self.model,
                self.tokenizer,
                os.path.join(self.save_path, "saved_models"),
                tag="step_{:05d}".format(self.steps),
                max_num=100,
                max_mem=self.args.max_save_mem,
            )

        # logs
        if eval_info or self.steps % self.args.logging_steps == 0:
            misc_info = self.get_misc_info()
            last_lr = self.scheduler.get_last_lr()[0]
            misc_info["lr"] = last_lr

            misc_info = {
                "misc/%s" % k: v
                for k, v in {
                    **misc_info,
                }.items()
            }
            logs_dict = {**train_info, **eval_info, **self.actor_info, **misc_info}
            logs_dict = self.strategy.all_reduce(logs_dict)
            logs_dict.update(
                self.strategy.all_reduce(
                    {
                        "misc/query_step": self.query_step,
                        "misc/prompt_consumed": self.prompt_consumed,
                    },
                    op="sum",
                )
            )

            if self.strategy.is_rank_0():
                if self.pi_buffer:
                    self.strategy.print(np.random.choice(self.pi_buffer))
                self.strategy.pprint(logs_dict)
                if self._wandb is not None:
                    self._wandb.log(logs_dict)

                with open(self.log_path, "a", encoding="utf-8") as f:
                    json.dump(logs_dict, f, indent=2, ensure_ascii=False)

        # if update_best_metric and self.steps > 0:
        #     max_save_num = self.args.max_save_num
        #     max_save_num = 5
        #     self.metrics_best = {
        #         "eval/average/accuracy": best_metric,
        #         "logs_dict": logs_dict,
        #     }
        #     self.strategy.save_model(
        #         self.model,
        #         self.tokenizer,
        #         os.path.join(self.save_path, "best_models"),
        #         tag="step_{:05d}".format(self.steps),
        #         max_num=max_save_num,
        #         max_mem=self.args.max_save_mem,
        #     )

    # Dr. GRPO Modification 2: Remove difficulty bias by just computing the MC advantage without dividing by std:
    def compute_monte_carlo_advantages(self, rewards, response_masks):
        del response_masks
        rewards = rewards.sum(-1)
        # Compute monte carlo trajectory-level advantage
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if self.args.critic_type == "grpo":
            # Additionally normalize by std.
            std_grouped_rewards = rewards.view(-1, self.args.num_samples).std(dim=1)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.args.num_samples, dim=0
            )
            advantages = advantages / (std_grouped_rewards + 1e-8)
        return advantages

    def _get_current_holder_p(self, current_step: int = None) -> float:
        """Compute dynamic Holder-p value from scheduler settings."""
        if self.args.holder_p_schedule == "constant":
            return float(self.args.holder_p)

        p_min = float(self.args.holder_p_min)
        p_max = float(self.args.holder_p_max)
        if p_min == p_max:
            return p_min

        schedule_steps = max(1, int(self.args.holder_p_schedule_steps))
        if current_step is None:
            current_step = self.steps
        current_step = int(max(0, current_step))

        progress = min(1.0, current_step / schedule_steps)
        sched = self.args.holder_p_schedule
        mid = (p_min + p_max) / 2.0
        amp = (p_max - p_min) / 2.0

        if sched == "linear":
            return p_min + (p_max - p_min) * progress
        if sched == "linear_dec":
            return p_max - (p_max - p_min) * progress
        if sched == "sin":
            # 2sin(xπ/4) → S-curve increase, symmetric through midpoint
            return mid - amp * np.cos(np.pi * progress)
        if sched == "cos":
            # -2sin(xπ/4) → S-curve decrease, symmetric through midpoint
            return mid + amp * np.cos(np.pi * progress)
        if sched == "poly":
            n = float(self.args.holder_p_power)
            return p_min + (p_max - p_min) * (progress ** n)
        if sched == "poly_dec":
            n = float(self.args.holder_p_power)
            return p_max - (p_max - p_min) * (progress ** n)
        if sched == "cubic":
            # x³/4 → symmetric S-curve increase through midpoint
            return mid + amp * (2.0 * progress - 1.0) ** 3
        if sched == "cubic_dec":
            # -x³/4 → symmetric S-curve decrease through midpoint
            return mid - amp * (2.0 * progress - 1.0) ** 3

        return float(self.args.holder_p)

    def _apply_template(self, example):
        problem = example[self.args.input_key]
        example[self.args.input_key] = TEMPLATE_FACTORY[args.prompt_template](problem)
        return example

    def prepare_data(self, strategy, tokenizer):
        prompt_dataset = load_data_from_disk_or_hf(self.args.prompt_data)
        prompts_data = prompt_dataset[self.args.train_split].select(
            range(min(self.args.max_train, len(prompt_dataset[self.args.train_split])))
        )

        # Prepare the data: templated questions & gt final answers.
        prompts_data = prompts_data.map(lambda x: self._apply_template(x))

        self.prompts_dataset = PromptDataset(
            prompts_data,
            tokenizer,
            strategy,
            input_key=self.args.input_key,
            output_key=self.args.output_key,
            apply_chat_template=False,  # Because we have applied already.
            get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            self.args.rollout_batch_size_per_device,
            pin_memory=True,
            shuffle=True,
        )
        self.eval_prompts_dataset = self.eval_prompts_dataloader = (
            None  # We use our own `self.eval_dataset_dict`.
        )

    def eval_dataloader_collate_fn(self, item_list):
        problems = []
        formatted_problems = []
        answers = []
        for item in item_list:
            problems.append(item["problem"])
            formatted_problems.append(
                TEMPLATE_FACTORY[self.args.prompt_template](item["problem"])
            )
            answers.append(item["answer"])
        return formatted_problems, problems, answers

    def evaluate(self, dataloader, steps):
        # Discard the default eval dataloader, and run eval on multiple benchmarks.
        del dataloader
        all_metrics = {}
        accuracies = []
        accuracies_pass = []
        scores = []
        lens = []
        pass_n_tasks = {
            t.strip() for t in (self.args.eval_pass_n_tasks or "").split(",") if t.strip()
        }
        for benchmark_name, dataset in self.eval_dataset_dict.items():
            eval_prompts_dataloader = DataLoader(
                dataset,
                batch_size=self.args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=self.eval_dataloader_collate_fn,
            )
            tag = f"{steps}_{benchmark_name}"
            self._configure_actor_eval_sampling(benchmark_name, pass_n_tasks)
            metrics = super().evaluate(eval_prompts_dataloader, tag)
            # Recover pass@n from the per-prompt scores oat dumped to disk.
            pass_at_n = self._compute_pass_at_n(tag)
            metrics["eval/accuracy_pass@n"] = pass_at_n
            all_metrics.update(
                {
                    k.replace("eval/", f"eval/{benchmark_name}/"): v
                    for k, v in metrics.items()
                }
            )
            accuracies.append(metrics["eval/accuracy"])
            accuracies_pass.append(pass_at_n)
            scores.append(metrics["eval/score"])
            lens.append(metrics["eval/response_tok_len"])
        all_metrics.update(
            {
                "eval/average/accuracy": np.mean(accuracies),
                "eval/average/accuracy_pass@n": np.mean(accuracies_pass),
                "eval/average/score": np.mean(scores),
                "eval/average/response_tok_len": np.mean(lens),
            }
        )
        return all_metrics

    def _configure_actor_eval_sampling(self, benchmark_name: str, pass_n_tasks: set):
        """Push per-benchmark eval sampling overrides to all actors.

        Tasks listed in ``pass_n_tasks`` get (eval_pass_n, eval_pass_temperature,
        eval_pass_top_p); other tasks are reset to the configured defaults.
        Mirrors oat's pattern of routing actor RPCs via group_rank_0 + barrier.
        """
        if benchmark_name in pass_n_tasks:
            n = self.args.eval_pass_n
            temperature = self.args.eval_pass_temperature
            top_p = self.args.eval_pass_top_p
        else:
            n = self.args.eval_n
            temperature = self.args.eval_temperature
            top_p = self.args.eval_top_p
        if self.strategy.is_group_rank_0():
            done = [
                actor.futures.set_eval_sampling_params(n, temperature, top_p)
                for actor in self.actors
            ]
            _ = [d.result() for d in done]
        dist.barrier()

    def _compute_pass_at_n(self, tag: str) -> float:
        """Read oat's per-prompt eval dump and compute pass@n.

        oat saves a JSON list at <save_path>/eval_results/<tag>.json with one row per
        prompt, where the `scores` field holds the n per-sample rewards. Pass@n is the
        fraction of prompts that have at least one correct sample.
        """
        pass_at_n = 0.0
        if self.strategy.is_rank_0():
            res_path = os.path.join(self.save_path, "eval_results", f"{tag}.json")
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
                hits = []
                for row in rows:
                    s = row.get("scores", [])
                    if not isinstance(s, list):
                        s = [s]
                    arr = np.asarray(s, dtype=float)
                    hits.append(float((arr > 0).any()))
                if hits:
                    pass_at_n = float(np.mean(hits))
            except Exception as e:
                logging.warning(f"pass@n compute failed for {tag}: {e}")
        # Sync across ranks so all_reduce in eval_and_log doesn't drift.
        return self.strategy.broadcast(pass_at_n)

    def learning_step(self, trajectory):
        args: PPOArgs = self.args
        infos = {}
        device = torch.cuda.current_device()
        input_ids = trajectory["input_ids"].to(device)
        att_mask = trajectory["attention_mask"].to(device)
        final_rewards = (
            torch.tensor([r[-1] for r in trajectory["rewards"]])
            .to(device)
            .reshape(-1, 1)
        ).float() * args.reward_scale
        prompt_id_lens = trajectory["prompt_ids_lens"]
        # action_logprobs = [
        #     torch.tensor(lp).to(device) for lp in trajectory["action_logprobs"]
        # ]
        loss_masks = torch.tensor(trajectory["loss_masks"]).float().to(device)
        completion_masks = self.get_completion_mask(att_mask, prompt_id_lens)
        response_masks = completion_masks[:, 1:]

        logging.info(f"learn data size {input_ids.shape}")

        indices = torch.arange(
            response_masks.size(1), device=response_masks.device
        ).expand_as(response_masks)
        masked_indices = torch.where(
            response_masks, indices, torch.full_like(indices, -1)
        )
        eos_indices = masked_indices.max(dim=1).values

        # Forward old models.
        ## 1) (Option 1) Policy log probabilities are directly from actors (vLLM).
        # logps = torch.zeros_like(response_masks).float()
        # for i in range(len(logps)):
        #     logps[i, torch.where(response_masks[i])[0]] = action_logprobs[i]
        ## 2) (Option 2) Reevaluate log probabilities using learner model.
        logps = torch.zeros(
            input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device
        )
        with torch.no_grad():
            for i in range(0, len(input_ids), args.train_batch_size_per_device):
                mini_batch_inds = torch.arange(i, i + args.train_batch_size_per_device)
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]

                batch_logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                batch_logits /= args.temperature
                batch_logps = self.get_batch_logps(
                    batch_logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                logps[mini_batch_inds, : mb_last_valid_token_pos - 1] = batch_logps

        ## 2) Reference.
        if self.ref_model is not None:
            all_ref_logps = []
            with torch.no_grad():
                for i in range(0, len(input_ids), args.train_batch_size_per_device):
                    batch_inds = torch.arange(i, i + args.train_batch_size_per_device)

                    batch_ref_logits = self.ref_model(
                        input_ids[batch_inds], attention_mask=att_mask[batch_inds]
                    )["logits"].float()
                    batch_ref_logits /= args.temperature
                    batch_ref_logps = self.get_batch_logps(
                        batch_ref_logits,
                        input_ids[batch_inds],
                        response_masks[batch_inds],
                    )
                    all_ref_logps.append(batch_ref_logps)
            ref_logps = torch.cat(all_ref_logps)

            # Combine final reward and kl penalty as rewards.
            kl_rewards = -args.kl_penalty_coef * (logps - ref_logps) * response_masks
            rewards = kl_rewards.clone()
            del all_ref_logps
            torch.cuda.empty_cache()
            gc.collect()
        else:
            rewards = torch.zeros_like(response_masks).float()

        rewards[torch.arange(len(rewards)), eos_indices] += final_rewards.squeeze()

        if self.args.critic_type == "ppo":
            advantages, returns, values = self.compute_ppo_advantages(
                rewards, input_ids, att_mask, response_masks
            )
        elif self.args.critic_type in ["grpo", "drgrpo"]:
            advantages = self.compute_monte_carlo_advantages(rewards, response_masks)[
                :, None
            ]

        # Compute losses and update models for multiple PPO epochs.
        stats = defaultdict(list)
        holder_p = self._get_current_holder_p(current_step=self.steps)
        local_grad_step = 0
        for _ in range(args.num_ppo_epochs):
            batch_inds = np.random.permutation(len(input_ids))
            for b_st in range(0, len(input_ids), args.train_batch_size_per_device):
                local_grad_step += 1
                stats["holder_p"].append(holder_p)
                mini_batch_inds = batch_inds[
                    b_st : b_st + args.train_batch_size_per_device
                ]
                mb_advantage = advantages[mini_batch_inds]
                mb_input_ids = input_ids[mini_batch_inds]
                mb_att_mask = att_mask[mini_batch_inds]
                mb_response_masks = response_masks[mini_batch_inds]
                mb_logps = logps[mini_batch_inds]
                mb_loss_masks = loss_masks[mini_batch_inds]

                # Remove unnecessary padding introduced by the large PPO batch.
                mb_valid_token_count_per_pos = mb_att_mask.sum(0)
                mb_last_valid_token_pos = torch.where(
                    mb_valid_token_count_per_pos == 0
                )[0]
                if len(mb_last_valid_token_pos) >= 1:
                    mb_last_valid_token_pos = mb_last_valid_token_pos[0]
                else:
                    mb_last_valid_token_pos = mb_att_mask.shape[1]
                # # Further reduce valid token num to speed up IF:
                # ## 1. We only have PG loss, i.e., args.beta == 0.
                # ## 2. Advantage is zero in bandit case (e.g., GRPO).
                # ## 3. train_batch_size_per_device is 1.
                # if (
                #     args.beta == 0
                #     and self.args.critic_type == "grpo"
                #     and len(mb_advantage) == 1
                # ):
                #     zero_adv = (mb_advantage == 0).item()  # bool
                #     if zero_adv:
                #         mb_last_valid_token_pos = 7  # An unimportant magic number.
                mb_input_ids = mb_input_ids[:, :mb_last_valid_token_pos]
                mb_att_mask = mb_att_mask[:, :mb_last_valid_token_pos]
                mb_response_masks = mb_response_masks[:, : mb_last_valid_token_pos - 1]
                mb_logps = mb_logps[:, : mb_last_valid_token_pos - 1]

                if self.args.critic_type == "ppo":
                    mb_return = returns[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_values = values[mini_batch_inds, : mb_last_valid_token_pos - 1]
                    mb_advantage = mb_advantage[:, : mb_last_valid_token_pos - 1]

                # Policy learning.
                logits = self.model(mb_input_ids, attention_mask=mb_att_mask)[
                    "logits"
                ].float()
                logits /= args.temperature
                new_logps = self.get_batch_logps(
                    logits,
                    mb_input_ids,
                    mb_response_masks,
                )
                if args.reinforce_update:
                    pg_loss_max = -mb_advantage * new_logps
                elif self.args.critic_type_modify == "holder":
                    p = holder_p
                    
                    # log(π_new/π_old) = new_logps - mb_logps 
                    logprobs_diff =  new_logps - mb_logps  # [batch_size, seq_len]

                    # |y_i| = number of response tokens per sample
                    response_lengths = mb_response_masks.sum(dim=-1, keepdim=True)  # [batch_size, 1]


                    if abs(p) < 1e-6:
                        # p -> 0 limit: geometric mean ratio = exp(mean(log(π_new/π_old)))
                        mean_log_ratio = (
                            (logprobs_diff * mb_response_masks).sum(dim=-1, keepdim=True)
                            / (response_lengths + 1e-8)
                        )
                        ratio = torch.exp(mean_log_ratio).squeeze(-1)  # [batch_size]
                    else:
                        # (π_new/π_old)^p = exp(p * log(π_new/π_old))
                        ratio_power = torch.exp(p * logprobs_diff)  # [batch_size, seq_len]

                        # Apply mask to only consider response tokens
                        masked_ratio_power = ratio_power * mb_response_masks  # [batch_size, seq_len]

                        # Σ_{t=1}^{|y_i|} (π_new/π_old)^p
                        sum_ratio_power = masked_ratio_power.sum(dim=-1, keepdim=True)  # [batch_size, 1]

                        # ρ_i,p(θ) = ((1/|y_i|) * Σ_t ...)^(1/p)
                        combined = sum_ratio_power / (response_lengths + 1e-8)  # [batch_size, 1]
                        ratio = combined.pow(1.0 / p).squeeze(-1)  # [batch_size]
                    
                    mb_advantage_flat = mb_advantage.squeeze(-1) if mb_advantage.dim() > 1 else mb_advantage
                    
                    # Policy gradient loss with clipping (PPO-style)
                    pg_losses = -mb_advantage_flat * ratio  # [batch_size]
                    pg_losses2 = -mb_advantage_flat * torch.clamp(
                        ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)
                    
                    # Statistics
                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )

                    # Aggregate loss and apply loss masks
                    pg_loss = (pg_loss_max * mb_loss_masks).mean()
                    infos["pg_loss"] = pg_loss.detach()
                    loss = pg_loss
                elif self.args.critic_type_modify == "sequence_w_clip":
                    # Sequence-level Hölder p-mean ratio with:
                    #   (1) per-token log-ratio clip:   Δ_t ← clip(Δ_t, -δ, +δ),  δ = log(1+ε)
                    #   (2) outer PPO max-of-two on sequence ratio.
                    p = holder_p

                    logprobs_diff_raw = new_logps - mb_logps  # [B, T]
                    delta = math.log(1.0 + args.cliprange)
                    logprobs_diff = torch.clamp(logprobs_diff_raw, -delta, delta)

                    response_lengths = mb_response_masks.sum(dim=-1, keepdim=True)  # [B, 1]

                    if abs(p) < 1e-6:
                        mean_log_ratio = (
                            (logprobs_diff * mb_response_masks).sum(dim=-1, keepdim=True)
                            / (response_lengths + 1e-8)
                        )
                        ratio = torch.exp(mean_log_ratio).squeeze(-1)
                    else:
                        ratio_power = torch.exp(p * logprobs_diff)
                        masked_ratio_power = ratio_power * mb_response_masks
                        sum_ratio_power = masked_ratio_power.sum(dim=-1, keepdim=True)
                        combined = sum_ratio_power / (response_lengths + 1e-8)
                        ratio = combined.pow(1.0 / p).squeeze(-1)

                    mb_advantage_flat = mb_advantage.squeeze(-1) if mb_advantage.dim() > 1 else mb_advantage

                    pg_losses = -mb_advantage_flat * ratio
                    pg_losses2 = -mb_advantage_flat * torch.clamp(
                        ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    with torch.no_grad():
                        token_clipped = (
                            (logprobs_diff_raw != logprobs_diff).float() * mb_response_masks
                        )
                        clipfrac_token = token_clipped.sum() / (mb_response_masks.sum() + 1e-8)
                        stats["logprobs_diff_clipfrac"].append(clipfrac_token.item())
                        stats["logprobs_diff_raw_max"].append(
                            torch.amax(logprobs_diff_raw.detach() * mb_response_masks).item()
                        )
                        stats["logprobs_diff_raw_min"].append(
                            torch.amin(logprobs_diff_raw.detach() * mb_response_masks).item()
                        )
                        stats["logprobs_diff_max"].append(
                            torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                        )
                        stats["logprobs_diff_min"].append(
                            torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                        )
                        stats["ratio_mean"].append(ratio.detach().mean().item())
                        stats["ratio_max"].append(ratio.detach().max().item())
                        stats["ratio_min"].append(ratio.detach().min().item())
                        stats["zero_pg_loss_count"].append(
                            (pg_loss_max == 0).detach().sum().item()
                        )

                    pg_loss = (pg_loss_max * mb_loss_masks).mean()
                    infos["pg_loss"] = pg_loss.detach()
                    loss = pg_loss
                elif self.args.critic_type_modify == "holder_token":
                    p = holder_p

                    # Per-token importance sampling ratios
                    logprobs_diff = new_logps - mb_logps  # [batch_size, seq_len]

                    token_ratio = torch.exp(logprobs_diff)  # r_{i,t}
                    token_ratio_clipped = torch.clamp(
                        token_ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )

                    # Response lengths
                    response_lengths = mb_response_masks.sum(dim=-1)  # [batch_size]

                    mb_advantage_flat = mb_advantage.squeeze(-1) if mb_advantage.dim() > 1 else mb_advantage

                    # For A_i > 0: C_i uses min(r, clip(r)) per token
                    # For A_i < 0: D_i uses max(r, clip(r)) per token
                    token_min = torch.min(token_ratio, token_ratio_clipped)  # [batch_size, seq_len]
                    token_max = torch.max(token_ratio, token_ratio_clipped)  # [batch_size, seq_len]

                    pos_mask = (mb_advantage_flat > 0).float()  # [batch_size]
                    neg_mask = (mb_advantage_flat < 0).float()  # [batch_size]

                    if abs(p) < 1e-6:
                        # p -> 0: geometric mean
                        log_min = torch.log(token_min + 1e-8) * mb_response_masks
                        log_max = torch.log(token_max + 1e-8) * mb_response_masks
                        C = torch.exp(log_min.sum(dim=-1) / (response_lengths + 1e-8))
                        D = torch.exp(log_max.sum(dim=-1) / (response_lengths + 1e-8))
                    else:
                        # Hölder mean: ((1/|y_i|) * Σ_t x_t^p)^(1/p)
                        # token_min/max are always > 0, .pow(p) safe for p in [-2, 2]
                        min_power = torch.exp(p * torch.log(token_min + 1e-8)) * mb_response_masks
                        max_power = torch.exp(p * torch.log(token_max + 1e-8)) * mb_response_masks

                        mean_min = min_power.sum(dim=-1) / (response_lengths + 1e-8)
                        mean_max = max_power.sum(dim=-1) / (response_lengths + 1e-8)
                        C = mean_min.pow(1.0 / p)
                        D = mean_max.pow(1.0 / p)

                    # loss_i = -C_i * A_i (A>0) or -D_i * A_i (A<0)
                    pg_loss_per_sample = -(C * pos_mask + D * neg_mask) * mb_advantage_flat

                    # Statistics
                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_per_sample == 0).detach().sum().item()
                    )

                    pg_loss = (pg_loss_per_sample * mb_loss_masks).mean()
                    infos["pg_loss"] = pg_loss.detach()
                    loss = pg_loss
                else:
                    logprobs_diff = new_logps - mb_logps
                    ratio = torch.exp(logprobs_diff)
                    pg_losses = -mb_advantage * ratio
                    pg_losses2 = -mb_advantage * torch.clamp(
                        ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                    )
                    pg_loss_max = torch.max(pg_losses, pg_losses2)

                    stats["logprobs_diff_max"].append(
                        torch.amax(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["logprobs_diff_min"].append(
                        torch.amin(logprobs_diff.detach() * mb_response_masks).item()
                    )
                    stats["zero_pg_loss_count"].append(
                        (pg_loss_max == 0).detach().sum().item()
                    )

                    pg_loss = self.masked_aggregator(pg_loss_max, mb_response_masks, axis=1)
                    pg_loss = (pg_loss * mb_loss_masks).mean()
                    infos["pg_loss"] = pg_loss.detach()
                    loss = pg_loss

                if args.beta > 0:
                    mb_ref_logps = ref_logps[mini_batch_inds]
                    mb_ref_logps = mb_ref_logps[:, : mb_last_valid_token_pos - 1]
                    # k3 kl: http://joschu.net/blog/kl-approx.html.
                    # clamp to avoid numerical instability.
                    log_ratio = (mb_ref_logps - new_logps).clamp(-40.0, 40.0)
                    kl3 = torch.expm1(log_ratio) - log_ratio  # expm1 is more stable.
                    infos["kl3"] = (kl3 * mb_response_masks).detach().sum(1).mean()

                    reg_loss = self.masked_aggregator(kl3, mb_response_masks, axis=1)
                    reg_loss = args.beta * (reg_loss * mb_loss_masks).mean()
                    infos["reg_loss"] = reg_loss.detach()
                    loss += reg_loss

                with torch.no_grad():
                    entropy = entropy_from_logits(logits[:, :-1])
                    entropy = masked_mean(entropy, mb_response_masks)
                    infos["entropy"] = entropy

                self.strategy.backward(loss, self.model, self.optimizer)

                if local_grad_step % self.strategy.grad_acc_step == 0:
                    _st = time.time()
                    stats["policy_grad_norm"].append(
                        self.strategy.get_gradient_norm(self.model)
                    )
                    stats["get_grad_norm_time"].append(time.time() - _st)

                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                if self.args.critic_type == "ppo":
                    # torch.cuda.empty_cache()
                    # gc.collect()

                    # Critic learning.
                    value_pred = self.critic(
                        input_ids=mb_input_ids, attention_mask=mb_att_mask
                    )[:, :-1]

                    value_pred_clipped = torch.clamp(
                        value_pred,
                        mb_values - args.cliprange_value,
                        mb_values + args.cliprange_value,
                    )
                    vf_losses1 = torch.square(value_pred - mb_return)
                    vf_losses2 = torch.square(value_pred_clipped - mb_return)
                    vf_loss_max = torch.max(vf_losses1, vf_losses2)

                    vf_loss = 0.5 * self.masked_aggregator(
                        vf_loss_max, mb_response_masks, axis=1
                    )
                    critic_loss = args.vf_coef * (vf_loss * mb_loss_masks).mean()

                    self.strategy.backward(
                        critic_loss, self.critic, self.critic_optimizer
                    )
                    self.strategy.optimizer_step(
                        self.critic_optimizer, self.critic, self.critic_scheduler
                    )
                    infos["critic_loss"] = critic_loss.detach()
                    infos["vf_clipfrac"] = masked_mean(
                        (vf_losses2 > vf_losses1).float(), mb_response_masks
                    ).detach()

                with torch.no_grad():
                    if not args.reinforce_update and self.args.critic_type_modify == "holder":
                        # holder uses clipping similar to GRPO
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        stats["pg_clipfrac"].append(pg_clipfrac.item())
                    elif not args.reinforce_update and self.args.critic_type_modify == "sequence_w_clip":
                        # sequence_w_clip: PPO max-of-two on sequence ratio
                        pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                        stats["pg_clipfrac"].append(pg_clipfrac.item())
                    elif not args.reinforce_update and self.args.critic_type_modify == "holder_token":
                        # fraction of tokens where clipping is active
                        token_clipped = (token_ratio != token_ratio_clipped).float() * mb_response_masks
                        pg_clipfrac = token_clipped.sum() / (mb_response_masks.sum() + 1e-8)
                        stats["pg_clipfrac"].append(pg_clipfrac.item())
                    elif not args.reinforce_update:
                        pg_clipfrac = masked_mean(
                            (pg_losses2 > pg_losses).float(), mb_response_masks, axis=1
                        )
                        stats["pg_clipfrac"].append(pg_clipfrac.mean().min().item())

        infos.update(
            {f"{k}_nan": torch.tensor(stats[k]).isnan().sum() for k in stats.keys()}
        )
        infos.update(
            {f"{k}_inf": torch.tensor(stats[k]).isinf().sum() for k in stats.keys()}
        )
        _pgn = torch.tensor(stats["policy_grad_norm"], dtype=torch.float32)
        infos["policy_grad_norm"] = _pgn.max()
        infos["policy_grad_norm_mean"] = _pgn.mean()
        # Variance of the policy gradient norm across micro-batches in this step.
        infos["policy_grad_norm_var"] = _pgn.var(unbiased=False) if _pgn.numel() > 1 else torch.tensor(0.0)
        # Variance of the squared grad norm ~ variance of ||g||^2, closer to classic PG variance.
        infos["policy_grad_sqnorm_var"] = (_pgn ** 2).var(unbiased=False) if _pgn.numel() > 1 else torch.tensor(0.0)
        infos["get_grad_norm_time"] = torch.tensor(sum(stats["get_grad_norm_time"]))
        infos["holder_p"] = torch.tensor(holder_p)
        if not args.reinforce_update:
            infos["logprobs_diff_max"] = torch.tensor(stats["logprobs_diff_max"]).max()
            infos["logprobs_diff_min"] = torch.tensor(stats["logprobs_diff_min"]).min()
            infos["zero_pg_loss_count"] = (
                torch.tensor(stats["zero_pg_loss_count"]).float().mean()
            )
            infos["pg_clipfrac"] = torch.tensor(stats["pg_clipfrac"]).mean()
        infos["adv_mean"] = advantages.mean().cpu()
        infos["adv_min"] = advantages.min().cpu()
        infos["adv_max"] = advantages.max().cpu()
        infos["all_zero_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 0).sum().cpu()
        )
        infos["all_one_rewards_count"] = (
            (final_rewards.view(-1, self.args.num_samples).mean(-1) == 1).sum().cpu()
        )

        return infos



def run_zero_math_rl(args: ZeroMathArgs):
    # Define a distributed program that composes Actors and Learners.
    program, local_resources = get_program(
        args, learner_cls=ZeroMathLearner, actor_cls=ZeroMathActor
    )
    # Launch the program in a local, multi-processing way!
    lp.launch(
        program,
        launch_type=args.launch_type,
        local_resources=local_resources,
        terminal="current_terminal",
    )


if __name__ == "__main__":
    args: ZeroMathArgs = get_default_args(ZeroMathArgs)
    # Customization:
    args.algo = "PPO"
    args.online_evaluation = True  # Use GT answer for online verification.
    if args.use_wb is True:
        wb_key = os.environ.get("WANDB_API_KEY", "").strip()
        if not wb_key:
            raise ValueError(
                "--use-wb is enabled but WANDB_API_KEY is empty. "
                "Please export WANDB_API_KEY or disable --use-wb."
            )
        args.use_wb = wb_key

    args = default_args_validation(args)
    run_zero_math_rl(args)
