# Copyright 2026 CW-GRPO Contributors
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

import json
import re
from openai import OpenAI

client = OpenAI(api_key="api_key_here", base_url="http://base_url_here/v1")
model_name = "gpt-oss-120b"

JUDGE_PROMPT = """\
You are an assistant that evaluates the validity of the agent’s reasoning in a question-answering task.

You will be given a partial trace consisting of alternating messages between the agent and the information source. Your task is to judge **only the latest action** in the trace, based on the agent’s latest <think> section and all previously retrieved information.

### Evaluation Criteria

#### Retrieval Reward
For the latest action, give a retrieval_reward of **1** if and only if:

1. **Relevance:**  
   The retrieved information is genuinely relevant to the main question and is likely to be helpful in answering it.

2. **Novelty:**  
   The retrieved information should offer new, useful content for answering the question that was not already obtained in previous rounds. If the same information (or its substance) was retrieved in previous rounds, do **not** assign a retrieval_reward, even if it is relevant.


#### Thinking Reward
For the latest action, give a thinking_reward of **1** if and only if:

1. **Reasoning Support:**
   The reasoning in the latest <think> section is logically grounded in the previously retrieved documents. The agent’s claims, assumptions, and deductions must be supported by the information already obtained.

2. **Action Usefulness:**
   - If the action is a **search**, the proposed retrieval query(s) must aim to obtain missing information that is necessary or beneficial for producing a correct answer.
   - If the action is an **answer**, the reasoning must align with and be properly supported by the retrieved information.
   - When assigning the thinking reward, you must only focus on the content and quality of the agent's latest <think> section; do NOT consider the relevance of the retrieved documents or whether the final answer matches the ground truth.

If **either** the reasoning is unsupported **or** the action is not helpful toward answering the question, assign a score of **0**.

### Additional Notes

- Your judgment must rely **only** on the conversation history and retrieved documents; do not use outside knowledge.
- The knowledge base (KB) used for retrieval was last updated in 2018.
- If multiple documents share the same title, treat them as parts of the same entry. But if the documents have similar but not the same title, treat them as separate and independent.
- The KB may contain documents with names similar to the target entity but that are actually irrelevant.  
  If the agent incorrectly treats such similar-but-unrelated documents as relevant and bases reasoning on them, you **must** assign 0.
- Be objective and consistent.

### Analysis Steps
Before you assign the rewards, you should first analyze the latest action in detail. Please follow these steps:

1. Extract the factual claims, reasoning logic, search intent, and assumptions made in the latest action's <think> section.
2. For each factual claim, check whether it is supported by previously retrieved passages or is a matter of common sense.
3. Analyze whether the reasoning logic is rigorous, and whether the search intent aligns with the reasoning and constitutes information still needed to answer the question. **Attention**: 
    - If the agent attempts to retrieve information that was previously searched for but not successfully obtained—by rephrasing, using synonyms, or otherwise varying the query—and if that information would be helpful for answering the question, you should consider this retrieval attempt useful.  
    - If the agent makes an assumption in place of unavailable information, and the assumption is logically justified, do not penalize the agent for this.
4. Extract information from the retrieved documents that may be relevant to the main question. For statements similar to the question, analyze carefully whether they are truly relevant or only superficially similar but unrelated. If no relevant information is present, skip step 5 and assign a retrieval_reward of 0.
5. For each relevant information, check whether it was already retrieved in previous rounds and, if so, in which round. Finally, give the retrieval_reward based on whether genuinely new relevant information was retrieved in this turn.

Please conduct your analysis in the order above and justify your scoring.


### Format

Input format: a partial multi-round conversation between the agent and the information source. Example:
```
Question: the question to answer
Agent: <think>...</think><tool_call>the first search tool call</tool_call>
Information: the information retrieved by the first search tool call
Agent: <think>...</think><tool_call>the second search tool call</tool_call>
Information: the information retrieved by the second search tool call
...
Agent (the last action): <think>...</think><tool_call|answer>...</tool_call|answer>
(Information: the information retrieved by the last search tool call)
```
Your should **only** evaluate the **last** action.

Return format: a JSON object, wrapped in ```json and ```. Example:
```json
  {"analysis": "your detailed analysis of the latest action", "thinking_reward": 0/1, "retrieval_reward": 0/1}
```\
"""

def _trace_to_text(trace: list[dict]):
    if "Question:" in trace[0]["content"]:
        trace_text = trace[0]["content"][trace[0]["content"].rfind("Question:"):]
    else:
        question = trace[0]["content"]
        trace_text = f"Question: {question}"
    
    for message in trace[1:]:
        role = "Agent" if message["role"] == "assistant" else "Information"
        content = message["content"]
        trace_text += f"\n\n{role}:\n{content}"
    return trace_text

def _parse_round_response(response_content: str):
    pattern = r"```json(.*)```"
    match = re.search(pattern, response_content, re.DOTALL)
    if match:
        response_content = match.group(1).strip()
        try:
            result = json.loads(response_content)
            return {"retrieval_reward": result["retrieval_reward"], "thinking_reward": result["thinking_reward"], "reason": result["analysis"]}
        except Exception:
            print(f"[llm-judge] Error in parsing round response: {response_content}")
            return None
    print(f"[llm-judge] No match found in round response: {response_content}")
    return None

def llm_judge(trace: list[dict], max_retries: int = 3):
    """
    Use LLM to judge the trace. If the LLM API fails or the response is not valid, retry up to `max_retries` times.
    Returns 0 if all attempts fail or result cannot be parsed.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": f"Here is the trace:\n{_trace_to_text(trace)}"}
                ],
                temperature=0.0
            )
            response_content = response.choices[0].message.content
            result = _parse_round_response(response_content)
            if result is not None:
                return result
        except Exception as e:
            print(f"[llm_judge] failed after {attempt+1} attempts: {e}")
    return {"retrieval_reward": 0, "thinking_reward": 0, "reason": ""}
