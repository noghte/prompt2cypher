import json
import os
import argparse
from dotenv import load_dotenv
from tabulate import tabulate  # For printing tables

#DEFAULT_CYPHERRESULTS_FILENAME = "baseline_benchmark-gpt-4o-mini-2025_04_01-16_14_33_version_1-with_results.json" #ProKinO
DEFAULT_CYPHERRESULTS_FILENAME = "ablation-gpt-4o-mini-2025_04_08-13_49_22-benchmark_format-with_results.json" #ICKG
IGNORE_EMPTY_PREDICTIONS = False  # Set to True to ignore empty predictions

parser = argparse.ArgumentParser(description="Process CYPHERRESULTS_FILENAME argument")
parser.add_argument("--cypher_results",
                    type=str,
                    default=DEFAULT_CYPHERRESULTS_FILENAME,
                    help="The JSON file containing the results of Cypher queries generated by LLM")

args = parser.parse_args()

CYPHERRESULTS_FILENAME = args.cypher_results # "." + ".".join(args.cypher_results.split(".")[2:])

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE_NAME = os.getenv("NEO4J_DATABASE_NAME")
KG_NAME = None
if NEO4J_DATABASE_NAME == "neo4j":
    KG_NAME = "ickg"
elif NEO4J_DATABASE_NAME == "prokino-kg":
    KG_NAME = "prokino"
    
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def flatten_results(results):
    """
    Flattens a list of lists or dictionaries into a list of individual items.
    Collects all items from each inner list or dictionary.
    """
    flattened = []
    for result in results:
        if isinstance(result, list):
            for item in result:
                if item is not None and str(item).strip():
                    flattened.append(str(item).strip())
        elif isinstance(result, dict):
            for item in result.values():
                if item is not None and str(item).strip():
                    flattened.append(str(item).strip())
        else:
            s = str(result) if result is not None else ""
            if s.strip():
                flattened.append(s.strip())
    return flattened


baseline_file_path = f'./data/{KG_NAME}/test_queries-with_results.json'
prediction_file_path = f"./results/{KG_NAME}/{CYPHERRESULTS_FILENAME}"

if not prediction_file_path.endswith("-with_results.json"):
    raise ValueError("The prediction file path should end with '-with_results.json'. Ensure that you have executed the 'benchmarks_cypher_execution.py' script correctly.")

baseline_data = load_json(baseline_file_path)
prediction_data = load_json(prediction_file_path)

# Extract and flatten the cypher results for baseline
baseline_results = {}
for item in baseline_data:
    cypher_results = item.get("cypher_result", [])
    if not isinstance(cypher_results, list):
        raise ValueError(f"Expected 'cypher_result' to be a list for title {item['title']}, but got {type(cypher_results)}.")
    flattened = flatten_results(cypher_results)
    baseline_results[item["title"]] = set(flattened)

# Extract and flatten the cypher results for predictions, considering the IGNORE_EMPTY_PREDICTIONS flag
prediction_results = {}
for item in prediction_data:
    for version, details in item.get("results", {}).items():
        if "cypher_result" in details:
            cypher_results = details["cypher_result"]
            if not isinstance(cypher_results, list):
                raise ValueError(f"Expected 'cypher_result' to be a list for title {item['title']}, but got {type(cypher_results)}.")
            flattened = set(flatten_results(cypher_results))
            if not IGNORE_EMPTY_PREDICTIONS or any(p.strip() for p in flattened):
                if item["title"] not in prediction_results:
                    prediction_results[item["title"]] = set()
                prediction_results[item["title"]].update(flattened)

# Initialize lists to store aggregated metrics
precision_list = []
recall_list = []
f1_list = []
jaccard_list = []

# Initialize lists for individual results
individual_results = []

# Iterate over each query to compute metrics
for item in baseline_data:
    title = item["title"]
    refs = baseline_results.get(title, set())
    preds = prediction_results.get(title, set())

    if IGNORE_EMPTY_PREDICTIONS and not preds:
        continue

    if preds or refs:
        # Compute True Positives
        true_positives = len(preds.intersection(refs))

        # Compute Precision
        precision = true_positives / len(preds) if preds else 0.0

        # Compute Recall
        recall = true_positives / len(refs) if refs else 0.0

        # Compute F1-Score
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Compute Jaccard Similarity
        union = len(refs.union(preds))
        intersection = len(refs.intersection(preds))
        jaccard = intersection / union if union != 0 else 0.0

        # Append to aggregated lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        jaccard_list.append(jaccard)

        # Append to individual results
        individual_results.append({
            "title": title,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Jaccard Similarity": jaccard
        })
    else:
        # Handle cases with no predictions and no references (if any)
        individual_results.append({
            "title": title,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-Score": 0.0,
            "Jaccard Similarity": 0.0
        })

# Compute aggregated metrics
aggregated_results = {
    "Precision": sum(precision_list) / len(precision_list) if precision_list else 0.0,
    "Recall": sum(recall_list) / len(recall_list) if recall_list else 0.0,
    "F1-Score": sum(f1_list) / len(f1_list) if f1_list else 0.0,
    "Jaccard Similarity": sum(jaccard_list) / len(jaccard_list) if jaccard_list else 0.0
}

# Prepare the results to save
results_to_save = {
    "aggregated": aggregated_results,
    "individual": individual_results
}

# Update the output file path as requested
output_file_path = prediction_file_path.replace("-with_results.json", "-metrics(precision_recall_f1).json")

# Save the aggregated and individual results to a JSON file
with open(output_file_path, 'w') as file:
    json.dump(results_to_save, file, indent=4)

print(f"Results saved to {output_file_path}")

# ==========================
# Print Aggregated Metrics in Table Format
# ==========================

# Prepare data for tabulate
columns = [
    'KG', 'Num Queries', 'Precision', 'Recall',
    'F1-Score', 'Jaccard Similarity'
]

# Use KG_NAME as the KG identifier
kg_name = KG_NAME

num_queries = len(individual_results)

aggregated = aggregated_results
precision = aggregated.get("Precision", 0.0)
recall = aggregated.get("Recall", 0.0)
f1 = aggregated.get("F1-Score", 0.0)
jaccard = aggregated.get("Jaccard Similarity", 0.0)

row = [
    kg_name, num_queries, f"{precision:.4f}", f"{recall:.4f}",
    f"{f1:.4f}", f"{jaccard:.4f}"
]

# Print the statistics in a table format
print("\n=== Aggregated Metrics Summary ===")
print(tabulate([row], headers=columns, tablefmt='grid'))

# ==========================
# Optional: Print Individual Metrics
# ==========================

# Uncomment the following lines if you want to see individual query metrics
# print("\n=== Individual Query Metrics ===")
# individual_table = []
# for res in individual_results:
#     individual_table.append([
#         res["title"],
#         f"{res['Precision']:.4f}",
#         f"{res['Recall']:.4f}",
#         f"{res['F1-Score']:.4f}",
#         f"{res['Jaccard Similarity']:.4f}"
#     ])
# individual_columns = ['Title', 'Precision', 'Recall', 'F1-Score', 'Jaccard Similarity']
# print(tabulate(individual_table, headers=individual_columns, tablefmt='grid'))
