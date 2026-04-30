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
