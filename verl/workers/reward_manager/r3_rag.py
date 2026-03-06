# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import re
import json
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.llm_judge import client

JUDGE_PROMPT = """\
You are an expert information retrieval evaluator.
Your task is to judge how helpful the currently retrieved documents are for answering the given question, compared to the documents retrieved in previous rounds.
Please assess, based on the criteria below, whether the newly retrieved documents provide additional valuable information for answering the question, relative to the information already found in earlier retrievals.

### Criteria
1. Match between search results and query intent.
2. Usefulness for question answering (completeness, accuracy).
3. Incremental value: Whether the current retrieval introduces new, beneficial information for answering the question compared to all previous retrievals.

### Return Format
Return a JSON object, wrapped in ```json and ```. Example:
```json
  {"analysis": "your detailed explanation, including discussion of incremental value", "retrieval_reward": float(0.0-1.0 scale)}
```\
"""

def _trace_to_text(trace: list[dict]):
    question = trace[0]["content"]
    if "Question:" in trace[0]["content"]:
        question = trace[0]["content"][trace[0]["content"].rfind("Question:"):]
    
    retrieved_docs = json.loads(trace[-1]["content"])["result"]
    thinking = trace[-2]["content"]
    search_query = '\n'.join(trace[-2]["tool_calls"][0]["function"]["arguments"]["query_list"]) if trace[-2].get("tool_calls") else "No search query."
    
    # collect documents retrieved by history rounds
    history_docs = []
    for msg in trace[2:-1:2]:
        docs = json.loads(msg["content"]).get("result", "")
        history_docs.append(docs)
    history_docs_str = ""
    for i, docs in enumerate(history_docs, start=1):
        history_docs_str += f"Round {i}:\n{docs}\n---\n"

    return f"""
[Question]
{question}

[Current Analysis]
{thinking}

[Search Query]
{search_query}

[Retrieved Documents]
{retrieved_docs}\

[Previous Rounds Retrieved Documents]
{history_docs_str}
""".strip()

def _parse_round_response(response_content: str):
    pattern = r"```json(.*)```"
    match = re.search(pattern, response_content, re.DOTALL)
    if match:
        response_content = match.group(1).strip()
        try:
            result = json.loads(response_content)
            return {"retrieval_reward": result["retrieval_reward"], "reason": result["analysis"]}
        except Exception:
            print(f"[llm-judge] Error in parsing round response: {response_content}")
            return None
    print(f"[llm-judge] No match found in round response: {response_content}")
    return None


def llm_judge(trace: list[dict], max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-oss-120b",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": _trace_to_text(trace)}
                ],
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            result = _parse_round_response(response_content)
            if result is not None:
                return result
        except Exception as e:
            print(f"[llm_judge] failed after {attempt+1} attempts: {e}")
    return {"retrieval_reward": 0, "reason": ""}


@register("r3-rag")
class R3RAGRewardManager(AbstractRewardManager):
    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the R3RAGRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
            **kwargs: Other custom parameters, which can be passed through the configuration file.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        
        self.max_workers = kwargs.get("max_llm_judge_workers", 64)

    def __call__(
        self, data: DataProto, return_dict: bool = False
    ) -> torch.Tensor | dict[str, Any]:
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        data_items_info = []
        judge_tasks = []  # List of (task_id, messages_slice)
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            # response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_str = data_item.non_tensor_batch["messages"]["messages"][-1]["content"]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # Compute answer reward
            task_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(task_score, dict):
                answer_reward = task_score["score"]
                # Store the information including original reward
                for key, value in task_score.items():
                    reward_extra_info[key].append(value)
            else:
                answer_reward = task_score

            # Process messages
            messages_dict = data_item.non_tensor_batch.get("messages", {})
            if isinstance(messages_dict, dict) and "messages" in messages_dict:
                messages = messages_dict["messages"]
            else:
                messages = messages_dict if isinstance(messages_dict, list) else []
            
            if messages:
                first_msg = messages[0]
                assert first_msg.get("role") == "system"
                messages = [msg for msg in messages if msg.get("role") != "system"]
            else:
                messages = []
            
            # Compute format rewards
            format_rewards = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls is None:
                        tool_calls = []
                    if len(tool_calls) != 1 and "<answer>" not in msg["content"]:
                        format_rewards.append(0)
                    else:
                        format_rewards.append(1)
            
            # Prepare llm_judge tasks
            task_idx = 0
            for j in range(3, len(messages), 2):
                judge_tasks.append((i, task_idx, messages[:j]))
                task_idx += 1
            
            data_items_info.append({
                "index": i,
                "valid_response_length": valid_response_length,
                "format_rewards": format_rewards,
                "answer_reward": answer_reward,
                "task_score": task_score,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "messages": messages,
                "num_judge_tasks": task_idx,
            })

        # Parallel execution of all llm_judge tasks
        judge_results_dict = {}  # {(sample_idx, task_idx): result}
        
        if judge_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_task = {
                    executor.submit(llm_judge, messages_slice): (sample_idx, task_idx)
                    for sample_idx, task_idx, messages_slice in judge_tasks
                }
                
                for future in as_completed(future_to_task):
                    sample_idx, task_idx = future_to_task[future]
                    try:
                        result = future.result()
                        judge_results_dict[(sample_idx, task_idx)] = result
                    except Exception as e:
                        print(f"llm_judge task failed for sample {sample_idx}, task {task_idx}: {e}")
                        judge_results_dict[(sample_idx, task_idx)] = {"retrieval_reward": 0, "reason": ""}

        # Process the results and compute the reward tensor
        for item_info in data_items_info:
            i = item_info["index"]
            valid_response_length = item_info["valid_response_length"]
            format_rewards = item_info["format_rewards"]
            answer_reward = item_info["answer_reward"]
            task_score = item_info["task_score"]
            data_source = item_info["data_source"]
            prompt_str = item_info["prompt_str"]
            response_str = item_info["response_str"]
            ground_truth = item_info["ground_truth"]
            messages = item_info["messages"]
            num_judge_tasks = item_info["num_judge_tasks"]

            if not all(format_rewards):
                factor = 1.0
            elif answer_reward:
                factor = 1.6
            else:
                factor = 0.8
            
            # Collect all judge_results for this item
            judge_results = []
            for task_idx in range(num_judge_tasks):
                result = judge_results_dict.get(
                    (i, task_idx),
                    {"retrieval_reward": 0, "reason": ""},
                )
                judge_results.append(result)
            
            reward_extra_info["answer_reward"].append(answer_reward)
            reward_extra_info["format_rewards"].append(format_rewards)
            reward_extra_info["judge_results"].append(judge_results)          

            round_last_token_mask = data[i].batch["round_last_token_mask"]
            true_indices = (round_last_token_mask == True).nonzero(as_tuple=True)[0].tolist()
            assert len(true_indices) <= len(judge_results) + 1, (
                f"Sample {i}: round_last_token_mask True count ({len(true_indices)}) > "
                f"judge_results count ({len(judge_results)})"
            )
            for idx, token_pos in enumerate(true_indices):
                if idx < len(judge_results):
                    reward = factor * (format_rewards[idx] * (judge_results[idx]["retrieval_reward"] + 1) - 1)
                elif len(judge_results) == 0:
                    reward = 0
                else:
                    reward = factor * (format_rewards[idx] * (answer_reward + 1) - 1)
                reward_tensor[i, token_pos] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(task_score, dict):
                    for key, value in task_score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", task_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
