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
import argparse
from collections import defaultdict

test_file = "data/agentgym/searchqa_test.json"
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
args = parser.parse_args()
validation_file = args.input

with open(test_file, "r") as f:
    test_data = json.load(f)
subset_list = [item["item_id"].split('_')[0] for item in test_data]

subset2scores = defaultdict(list)
overall_scores = []
with open(validation_file, "r") as f:
    i = 0
    for line in f:
        data = json.loads(line)
        subset_id = subset_list[i // 4]
        score = 1 if data["score"] > 0.0 else 0
        subset2scores[subset_id].append(score)
        overall_scores.append(score)
        i += 1

for subset_id in ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]:
    scores = subset2scores[subset_id]
    print(f"{subset_id}: {sum(scores) / len(scores)}")
print(f"overall: {sum(overall_scores) / len(overall_scores)}")
