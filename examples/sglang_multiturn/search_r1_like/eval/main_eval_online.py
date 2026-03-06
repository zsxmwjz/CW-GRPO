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
"""
Online eval script: use OpenAI client to call online service for evaluation.
"""

import os
import json
from collections import defaultdict
from typing import Optional, Any

import asyncio
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from openai import AsyncOpenAI, RateLimitError
from pprint import pprint
import aiohttp
from tqdm import tqdm

from verl.utils.reward_score import default_compute_score

new_user_prefix = """\
Your task is to retrieve knowledge from the knowledge base to answer the given question.

You must follow this conversational protocol exactly:
  - Reasoning: list steps and immediate plan for the next retrieval(s).
  - Issue a single retrieval with the search tool: provide ONE complete natural-language query (a full question) suitable for a dense retriever (FAISS).
  - After each retrieval, you MUST wait for external tool response before continuing. Do not proceed or speculate until information is received.
  - When ready to conclude, reply in <answer>...</answer>. Keep your answer as concise as possible, and it would be best if the answer is a single entity (such as <answer>Beijing</answer>, <answer>1997</answer>, <answer>yes</answer>).
  - If the answer entity has an alternative name, just choose one of them as the answer.
  - Every step must begin with reasoning before issuing tool call or answer.

Search guidance and KB metadata:
  - Queries must be natural-language questions (no keywords, no fragments). e.g. "Who is the author X?" not "author X".
  - The KB was last updated in 2018.
  - Documents with identical titles should be treated as parts of the same entry and may be combined when explicitly supported by their content. Otherwise treat docs independently.

Granularity and commonsense:
  - When the question requires multiple reasoning hops, decompose it into smaller sub-questions and search sequentially for the information needed at each hop.
  - You may use common-sense (general knowledge a broadly-educated adult would know) to guide reasoning, formulate searches, or fill obvious gaps — but mark its use in <think> (e.g., "using commonsense: ...").

Entity disambiguation and query rephrasing:
  - If the question is ambiguous, infer the likely entity type (person/place/org/work) and include that type as a constraint in your query.
  - If tool response lacks the target, retry with rephrasing: use synonyms or near-synonyms, reverse the relation (e.g., "Who founded X" ↔ "Who is the founder of X"), or adjust the information granularity (broaden or narrow the scope). Do not use the same query twice.

Behavioral rules:
  - Do not invent facts. Only assert in <answer> what follows from retrieved documents or clearly-marked commonsense used.
  - Keep all text outside the tags minimal and task-focused.\
"""

def load_tool_config(tool_config_path: str) -> tuple[list, dict]:
    """
    Load tool config and convert to OpenAI format.

    Args:
        tool_config_path: Path to tool config file.

    Returns:
        (tools, tool_config):
        - tools: List of tools in OpenAI format
        - tool_config: Tool config dict (includes retrieval_service_url and timeout)
    """
    tool_config_omega = OmegaConf.load(tool_config_path)
    tool_config_dict = OmegaConf.to_container(tool_config_omega, resolve=True)
    
    tools = []
    tool_config = {}
    
    for tool_item in tool_config_dict.get("tools", []):
        # Extract tool config
        config = tool_item.get("config", {})
        tool_config["retrieval_service_url"] = config.get("retrieval_service_url")
        tool_config["timeout"] = config.get("timeout", 30)
        
        # Convert tool schema to OpenAI format
        tool_schema = tool_item.get("tool_schema", {})
        if tool_schema.get("type") == "function":
            function_schema = tool_schema.get("function", {})
            parameters = json.loads(json.dumps(function_schema.get("parameters", {})))
            if "properties" in parameters:
                for prop_name, prop_def in parameters["properties"].items():
                    if "item" in prop_def:
                        prop_def["items"] = prop_def.pop("item")
            
            openai_tool = {
                "type": "function",
                "function": {
                    "name": function_schema.get("name"),
                    "description": function_schema.get("description", ""),
                    "parameters": parameters
                }
            }
            tools.append(openai_tool)
    
    return tools, tool_config

def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()

def _format_retrieval_result(retrieval_results):
    """Format retrieval results."""
    pretty_results = [_passages2string(retrieval) for retrieval in retrieval_results]
    final_result =  "\n---\n".join(pretty_results)
    return json.dumps({"result": final_result}, ensure_ascii=False)

async def execute_search_tool(
    tool_call: Any,
    tool_config: dict,
    session: aiohttp.ClientSession
) -> str:
    """
    Execute a search tool call.

    Args:
        tool_call: Tool call object
        tool_config: Tool config dict
        retrieval_service_url: Retrieval service URL
        timeout: Timeout in seconds

    Returns:
        JSON string of tool execution result
    """
    try:
        function_args = json.loads(tool_call.function.arguments)
        query_list = function_args["query_list"]
        
        request_body = {
            "queries": query_list,
            "topk": 3,
            "return_scores": True
        }
        
        retrieval_service_url = tool_config.get("retrieval_service_url")
        timeout = tool_config.get("timeout", 30)
        
        async with session.post(
            retrieval_service_url,
            json=request_body,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status == 200:
                result = await response.json()
                return _format_retrieval_result(result["result"])
            else:
                error_msg = f"Error status code: {response.status}"
                return json.dumps({"error": error_msg}, ensure_ascii=False)
                
    except asyncio.TimeoutError:
        error_msg = f"Retrieval service request timeout ({timeout} seconds)"
        return json.dumps({"error": error_msg}, ensure_ascii=False)
    except Exception as e:
        error_msg = f"Error executing tool call: {str(e)}"
        return json.dumps({"error": error_msg}, ensure_ascii=False)


async def generate_with_tools(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    max_assistant_turns: int,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_config: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """
    Generate with multi-turn tool calls using OpenAI client.

    Args:
        client: AsyncOpenAI client
        max_assistant_turns: Max assistant turns

    Returns:
        Final trajectory (full conversation across all turns)
    """
    current_messages = messages.copy()
    assistant_turn = 0
    
    async with aiohttp.ClientSession() as session:
        while assistant_turn < max_assistant_turns:
            request_params = {
                "model": model,
                "messages": current_messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            
            if top_k is not None:
                request_params["top_k"] = top_k
            
            if tools is not None:
                request_params["tools"] = tools
            
            cnt, attempt, max_tries = 0, 0, 10
            retry_intervals = [10, 20, 40, 80]
            while attempt < max_tries:
                try:
                    response = await client.chat.completions.create(**request_params)
                    break
                except RateLimitError as e:
                    wait_time = retry_intervals[cnt] if cnt < len(retry_intervals) else retry_intervals[-1]
                    print(f"Request failed, error: {e}")
                    await asyncio.sleep(wait_time)
                    cnt += 1
                except Exception as e:
                    if attempt == max_tries - 1:
                        raise e
                    wait_time = retry_intervals[attempt] if attempt < len(retry_intervals) else retry_intervals[-1]
                    print(f"Request failed, retrying in {wait_time}s (attempt {attempt + 1}/{max_tries}), error: {e}")
                    await asyncio.sleep(wait_time)
                    attempt += 1
            assistant_message = response.choices[0].message
            
            # Build assistant message
            assistant_msg = {
                "role": "assistant",
                "content": assistant_message.content,
            }
            if hasattr(assistant_message, "reasoning_content"):
                assistant_msg["reasoning_content"] = assistant_message.reasoning_content
            
            # Add tool_calls to message if present
            if assistant_message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            
            current_messages.append(assistant_msg)
            
            # No tool_calls means generation is done
            if not assistant_message.tool_calls or assistant_turn == max_assistant_turns - 1:
                break
            
            # Handle tool_calls: execute tools and append tool messages
            for tool_call in assistant_message.tool_calls:
                # Execute tool call
                tool_result = await execute_search_tool(
                    tool_call=tool_call,
                    tool_config=tool_config,
                    session=session
                )
                
                # Append tool result to messages
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_result,
                })
            
            assistant_turn += 1
            if assistant_turn == max_assistant_turns - 1:
                current_messages[-1]["content"] += "\nIt is your last turn for this task. You must give your final answer in <answer>...</answer>."
    
    # Return accumulated content when max turns reached
    return current_messages


async def generate_single(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    max_assistant_turns: int,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    max_tokens: int,
    tools: Optional[list] = None,
    tool_config: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """
    Generate trajectory for a single sample.

    Args:
        client: AsyncOpenAI client
        max_assistant_turns: Max assistant turns

    Returns:
        Generated trajectory
    """
    try:
        response = await generate_with_tools(
            client=client,
            model=model,
            messages=messages,
            max_assistant_turns=max_assistant_turns,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            tool_config=tool_config,
        )
        return response
    except Exception as e:
        print(f"Generation error: {e}")
        return []


async def generate_batch(
    client: AsyncOpenAI,
    model: str,
    chat_list: list,
    n_samples: int,
    max_assistant_turns: int,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    max_tokens: int,
    max_parallel: int = 5,
    tools: Optional[list] = None,
    tool_config: Optional[dict] = None,
) -> list[list[dict[str, Any]]]:
    """
    Generate trajectories in batch with concurrency control.

    Args:
        client: AsyncOpenAI client
        chat_list: List of message lists
        n_samples: Number of samples per prompt
        max_assistant_turns: Max assistant turns

    Returns:
        List of full messages, order: [prompt1_sample1, prompt1_sample2, ..., prompt2_sample1, prompt2_sample2, ...]
    """
    semaphore = asyncio.Semaphore(max_parallel)
    total_tasks = len(chat_list) * n_samples

    # Progress bar
    pbar = tqdm(total=total_tasks, desc="Generating trajectories")

    async def sem_task_wrapper(*args, **kwargs):
        async with semaphore:
            try:
                result = await generate_single(*args, **kwargs)
                return result
            finally:
                # Update progress when task completes
                pbar.update(1)

    tasks = []
    for messages in chat_list:
        for _ in range(n_samples):
            tasks.append(
                sem_task_wrapper(
                    client=client,
                    model=model,
                    messages=messages,
                    max_assistant_turns=max_assistant_turns,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_config=tool_config,
                )
            )

    # Use gather to preserve order
    try:
        results = await asyncio.gather(*tasks)
        return results
    finally:
        pbar.close()


def compute_scores(
    responses: list,
    ground_truths: list,
    data_sources: list,
    n_samples: int,
) -> tuple[list, dict]:
    """
    Compute scores.

    Args:
        responses: Trajectory list, shape (num_prompts * n_samples,)
        ground_truths: Ground truth list, shape (num_prompts,)
        data_sources: Data source list, shape (num_prompts,)
        n_samples: Number of samples per prompt

    Returns:
        (scores_per_sample, scores_by_source):
        - scores_per_sample: Score per sample, shape (num_prompts * n_samples,)
        - scores_by_source: Scores grouped by data source
    """
    num_prompts = len(ground_truths)
    scores_per_sample = []
    scores_by_source = defaultdict(list)
    
    for i in range(num_prompts):
        ground_truth = ground_truths[i]
        data_source = data_sources[i]
        
        # Score the n_samples trajectories for this prompt
        prompt_scores = []
        for j in range(n_samples):
            idx = i * n_samples + j
            response = responses[idx]
            
            try:
                score = default_compute_score(
                    data_source=data_source,
                    solution_str=response,
                    ground_truth=ground_truth,
                )
                # If score is a dict, extract the numeric score
                if isinstance(score, dict):
                    score = score.get("score", score.get("accuracy", 0.0))
                prompt_scores.append(float(score))
            except Exception as e:
                print(f"Score computation error (prompt {i}, sample {j}): {e}")
                prompt_scores.append(0.0)
        
        scores_per_sample.extend(prompt_scores)
        scores_by_source[data_source].append(prompt_scores)
    
    return scores_per_sample, scores_by_source


def compute_metrics(scores_per_sample: list, scores_by_source: dict, n_samples: int) -> dict:
    """
    Compute evaluation metrics.

    Args:
        scores_per_sample: Score per sample
        scores_by_source: Scores grouped by data source
        n_samples: Number of samples per prompt

    Returns:
        Metrics dict
    """
    num_prompts = len(scores_per_sample) // n_samples
    scores_array = np.array(scores_per_sample).reshape(num_prompts, n_samples)
    
    # avg@n: mean of (per-prompt mean over n_samples trajectories)
    avg_at_n = np.mean(scores_array.mean(axis=1))
    
    metrics = {
        f"avg@{n_samples}": float(avg_at_n),
    }
    
    # avg@n per data source
    for data_source, prompt_scores_list in scores_by_source.items():
        if len(prompt_scores_list) == 0:
            continue
        
        # Convert per-prompt score lists to array
        source_scores = np.array(prompt_scores_list)
        
        avg_at_n = np.mean(source_scores.mean(axis=1))
        metrics[f"avg@{n_samples}/{data_source}"] = float(avg_at_n)
    
    return metrics


def main(config):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # Load config
    base_url = config.service.base_url
    model = config.service.model
    api_key = config.service.api_key
    temperature = config.service.temperature
    top_k = config.service.get("top_k", None)
    top_p = config.service.top_p
    max_tokens = config.service.max_tokens
    max_parallel = config.service.max_parallel
    
    data_path = config.data.path
    prompt_key = config.data.prompt_key
    n_samples = config.data.n_samples
    max_assistant_turns = config.data.max_assistant_turns
    
    # Load tool config
    tool_config_path = config.get("tool_config_path")
    tools = None
    tool_config = None
    if tool_config_path:
        tools, tool_config = load_tool_config(tool_config_path)
        print(f"Loaded tool config: {tool_config_path}")
        print(f"Retrieval service URL: {tool_config.get('retrieval_service_url')}")
        print(f"Timeout: {tool_config.get('timeout')}s")
    
    # Load dataset
    dataset = pd.read_parquet(data_path)
    
    # Get prompts
    chat_list = dataset[prompt_key].tolist()
    # Convert numpy arrays in chat_list to lists
    chat_list = [chat.tolist() if isinstance(chat, np.ndarray) else chat for chat in chat_list]

    for chat in chat_list:
        chat[0] = {"role": "system", "content": chat[0]["content"]}
        chat[1] = {"role": "user", "content": new_user_prefix + '\n\n' + chat[1]["content"][chat[1]["content"].rfind("Question:"):]}
    
    reward_models = dataset["reward_model"].tolist()
    ground_truths = [reward_model["ground_truth"] for reward_model in reward_models]
    data_sources = dataset["data_source"].tolist()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Samples per prompt: {n_samples}")
    print(f"Total generations: {len(dataset) * n_samples}")
    print(f"Max parallel: {max_parallel}")
    
    # Create OpenAI client
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    
    # Batch generation
    print("Generating trajectories...")
    responses = asyncio.run(
        generate_batch(
            client=client,
            model=model,
            chat_list=chat_list,
            n_samples=n_samples,
            max_assistant_turns=max_assistant_turns,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            max_parallel=max_parallel,
            tools=tools,
            tool_config=tool_config,
        )
    )
    
    print(f"Generation done, {len(responses)} trajectories")
    answers = [response[-1]["content"] if len(response) > 0 and "tool_calls" not in response[-1] else "" for response in responses]
    
    # Compute scores
    print("Computing scores...")
    scores_per_sample, scores_by_source = compute_scores(
        responses=answers,
        ground_truths=ground_truths,
        data_sources=data_sources,
        n_samples=n_samples,
    )
    
    # Compute metrics
    print("Computing evaluation metrics...")
    metrics = compute_metrics(scores_per_sample, scores_by_source, n_samples)
    
    # Print results
    print("\n" + "=" * 50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    # Optionally save results to jsonl
    if config.get("output_path"):
        output_path = config.output_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, "a", encoding="utf-8") as fout:
            for idx in range(len(responses)):
                prompt_idx = idx // n_samples

                item = {
                    "messages": responses[idx],
                    "score": scores_per_sample[idx],
                    "gts": {"target": ground_truths[prompt_idx]["target"].tolist()}
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "eval_online.yaml"))
    main(config)
