
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.trainer import GRPOTrainer
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available, is_rich_available
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator


import wandb
from einops import rearrange,reduce
from copy import deepcopy


CONT_TEXT = "\nWait, but\n"
EOS_TOK = "<|im_end|>"

class CustomTrainer(GRPOTrainer):
    def __init__(self,
                 reward_funcs,
                max_rewards=None, *args, **kwargs):
        self.max_rewards = max_rewards if max_rewards is not None else [float('-inf')] *len(reward_funcs)
        assert len(reward_funcs) == len(self.max_rewards)
        super().__init__(reward_funcs=reward_funcs, *args, **kwargs)
        self.gen_time = 2


    def _generate_and_score_completions_w_bf(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        cont_tokens_id = self.processing_class.encode(CONT_TEXT)
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            global_bz = len(all_prompts_text) // self.gen_time
            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = list(dict.fromkeys(all_prompts_text))
                # Update SamplingParams for iterative generation
                first_samp_params = deepcopy(self.sampling_params)
                first_samp_params.max_tokens = (self.max_completion_length) // self.gen_time - len(cont_tokens_id)
                first_samp_params.stop_token_ids = [self.processing_class.eos_token_id]
                iter_samp_params = deepcopy(first_samp_params)
                iter_samp_params.n = 1
                completion_ids = []
                # TODO: if vllm cache not enough, should reduce the batch size and be cautious about the flatten list order
                for ordered_set_of_prompts_ in [ordered_set_of_prompts]:
                    # Force the model to generate for numbers of gen time
                    for i in range(self.gen_time):

                        if i == 0:
                            batched_outputs = self.llm.generate(
                                ordered_set_of_prompts_, sampling_params=first_samp_params, use_tqdm=False
                            )
                            flatten_input_ids = []
                            flatten_outputs = []
                            for outputs in batched_outputs:
                                flatten_input_ids.extend(
                                    [outputs.prompt_token_ids[:] for _ in range(self.num_generations)])
                                flatten_outputs.extend(outputs.outputs)


                        else:
                            batched_outputs = self.llm.generate([{"prompt_token_ids":id} for id in flatten_input_ids],
                                                       sampling_params=iter_samp_params, use_tqdm=False)
                            flatten_outputs = [b.outputs[0] for b in batched_outputs]
                    
                        completion_ids_i = []
                        for output, input_ids in zip(flatten_outputs, flatten_input_ids, strict=True):
                            # for output, input_ids in zip(outputs.outputs, inputs_ids, strict=True):
                                comp_ids = list(output.token_ids)
                                completion_ids_i.append(deepcopy(comp_ids))
                                comp_ids.extend(cont_tokens_id) 
                                input_ids += comp_ids
                        
                        completion_ids.extend(completion_ids_i)

                last_c = None
                # assert it is divisible
                assert len(completion_ids) % global_bz == 0
                # breakpoint()
                for i in range(self.gen_time):
                    comp = completion_ids[i* global_bz: (i + 1) * global_bz]
                    if i != 0:
                        for j in range(global_bz):
                            comp[j] = last_c[j] + cont_tokens_id + comp[j]
                    last_c = comp

                    # else:
                    #     last_c = completion_ids[i*self.gen_time: i*self.gen_time + len(prompts)]
            else:
                completion_ids = [None] * (len(all_prompts_text) * self.gen_time)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            # len(prompts) = (d * b * g,) completion_ids have shape ()
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            # process_slice = slice(
            #     self.accelerator.process_index * len(prompts) ,
            #     (self.accelerator.process_index + 1) * len(prompts),
            # )
            # completion_ids = completion_ids[process_slice]
            completion_ids_a = deepcopy(completion_ids)
            completion_ids = [completion_ids[
                self.accelerator.process_index * len(prompts) + i * global_bz:
                (self.accelerator.process_index + 1) * len(prompts) + i * global_bz]
                  for i in range(self.gen_time)
                ]
            # flatten the list
            completion_ids = [item for sublist in completion_ids for item in sublist]

            # # Pad the completions, and concatenate them with the prompts
            # completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            # completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            # prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            raise NotImplemented()
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate( 
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        

        # Decode the generated completions
        # breakpoint()
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts * self.gen_time, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])

        else:
            completions = completions_text
        

        rewards_per_func = torch.zeros(len(prompts), self.gen_time, len(self.reward_funcs), device=device)
        # breakpoint()
        for j in range(self.gen_time):
            comp_i = completions[j* len(prompts): (j+1)*len(prompts)]

            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    raise NotImplementedError("Not supported")
                    if is_conversational(inputs[0]):
                        messages = [{"messages": ordered_set_of_prompts_ + c} for ordered_set_of_prompts_, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [ordered_set_of_prompts_ + c for ordered_set_of_prompts_, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=comp_i, **reward_kwargs)
                    rewards_per_func[:, j, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # select the completion with the highest reward in gen_time
                # rewards_per_func = torch.zeros(len(prompts), self.gen_time, len(self.reward_funcs), device=device)
        rewards_per_func = rearrange(rewards_per_func, "b bf r -> (b bf) r") 
        rewards =  rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        rewards = rearrange(rewards, " (b bf) r -> b bf r", bf = self.gen_time) 
        best = torch.max(rewards.sum(dim=-1), dim=-1)
        best_completion_indices = best.indices
        best_completion_ids = [completion_ids[i * self.gen_time + j] for i, j in enumerate(best_completion_indices)]
        completion_ids  = best_completion_ids
        assert len(completion_ids) == len(prompts)
        # Pad the completions, and concatenate them with the prompts
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )


        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # TODO: Add tokenwise rewards
        # # completions may be distributed across processes
        # rewards_per_func = gather(rewards_per_func)

        # # Apply weights to each reward function's output and sum
        # rewards_per_func = rewards_per_func.max(dim=-1).values
        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        rewards = gather(rewards) # shape (mb bf r), weighted
        rewards = reduce(rewards, "mb bf r -> mb bf", "sum" ) # shape (mb, bf)
        rewards = reduce(rewards, "mb bf -> mb", "max" ) # shape (mb,) mb = batch * num_generations
        rewards_per_func = gather(rewards_per_func) # for logging

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float()
        self._metrics[mode]["completion_length"].append(completion_length.mean().item())
        self._metrics[mode]["max_completion_length"].append(completion_length.max().item())
        self._metrics[mode]["min_completion_length"].append(completion_length.min().item())

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            # rewards_to_log = rewards.tolist()
            ind= gather(best_completion_indices).cpu()
            rewards_to_log = rewards_per_func.cpu()
            rewards_to_log = rewards_to_log.view(-1, len(self.reward_funcs))[ind.view(-1)]

            if self.accelerator.is_main_process:
                if is_rich_available():

                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        [rf.__name__ for rf in self.reward_funcs],
                        rewards_to_log,
                        ind.tolist(),
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                # inputs = self._generate_and_score_completions(inputs)
                inputs = self._generate_and_score_completions_w_bf(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss


    def _draft(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self,prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        prompt_length = prompt_ids.size(1)


        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        max_completion_length = self.max_completion_length

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step


        all_prompts_text = gather_object(prompts_text)
        for gen_round in range(self.num_generations):
            
            max_rewards = torch.tensor(self.max_rewards, device=device)

            if self.args.use_vllm:

                        
                # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
                if gen_round > 0:
                    if self.accelerator.is_main_process:
                        all_prompts_text =[]
                        # vllm will not inclue EOS or stop tokens in textual output but in the ids
                        good_traj = rewards_per_func >= max_rewards
                        good_traj = good_traj.all(dim=-1)
                        for i, output_text in enumerate(outputs):
                            prompt = output_text.prompt
                            generated_text = output_text.outputs[0].text
                            if good_traj[i]:
                                all_prompts_text.append(prompt + generated_text)
                            else:
                                all_prompts_text.append(prompt + generated_text + CONT_TEXT)
                if self.accelerator.is_main_process:
                    outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                    completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                else:
                    completion_ids = [None] * len(all_prompts_text)
                # Broadcast the completions from the main process to all processes, ensuring each process receives its
                # corresponding slice.
                completion_ids = broadcast_object_list(completion_ids, from_process=0)
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

                # Pad the completions, and concatenate them with the prompts
                completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
                completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            else:
                # raise NotImplementedError("Regular generation is not supported in this version of the GRPOTrainer")
                # Regular generation path
                raise NotImplementedError("Regular generation is not supported in this version of the GRPOTrainer")
                
                if gen_round > 0:
                    if self.accelerator.is_main_process:
                        # all_prompts_text =[]
                        # vllm will not inclue EOS or stop tokens in textual output but in the ids

                        model_out_text = self.processing_class.batch_decode(prompt_completion_ids, skip_special_tokens=True)
                        good_traj = rewards_per_func >= max_rewards
                        good_traj = good_traj.all(dim=-1)
                        model_in_text = []
                        for i, output_text in enumerate(model_out_text):
                            if good_traj[i]:
                                model_in_text.append(output_text+ self.processing_class.eos_token)
                            else:
                                model_in_text.append(output_text + CONT_TEXT)

                        model_inputs = self.processing_class(
                            model_in_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                        )
                        model_inputs = Trainer._prepare_inputs(self,model_inputs)
                        input_ids, input_mask = model_inputs["input_ids"], model_inputs["attention_mask"]
                        

                        #lets suppose no attention mask is needed
                        # breakpoint()
                        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                            prompt_completion_ids = unwrapped_model.generate(
                                input_ids, attention_mask=input_mask, generation_config=self.generation_config
                            )
                else:
                    with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                    )

                # Compute prompt length and extract completion ids
                # prompt_length = prompt_ids.size(1)
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()


            # Decode the generated completions
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = []
                for prompt, completion in zip(prompts, completions_text):
                    bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                    completions.append([{"role": "assistant", "content": bootstrap + completion}])
            else:
                completions = completions_text

            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                    raise NotImplementedError("Not supported")
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
            # completions may be distributed across processes
            rewards_per_func = gather(rewards_per_func)


        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


