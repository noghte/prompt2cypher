import json
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

DEFAULT_CYPHER_RESULTS = "ablation-gpt-4o-mini-2025_04_01-17_02_40-benchmark_format-with_results.json"
DEFAULT_BASELINE_RESULTS = "baseline_benchmark-gpt-4o-mini-2025_04_01-16_14_33_version_1-metrics(precision_recall_f1).json"

parser = argparse.ArgumentParser(description="Process ablation results for metric calculation")
parser.add_argument("--cypher_results", 
                    type=str,
                    default=DEFAULT_CYPHER_RESULTS,
                    help="The JSON file containing the ablation benchmark results")
parser.add_argument("--baseline_results",
                    type=str,
                    default=DEFAULT_BASELINE_RESULTS,
                    help="The JSON file containing the baseline benchmark results for comparison")

args = parser.parse_args()
CYPHER_RESULTS = args.cypher_results
BASELINE_RESULTS = args.baseline_results

# Database and KG settings
NEO4J_DATABASE_NAME = os.getenv("NEO4J_DATABASE_NAME")
KG_NAME = None
if NEO4J_DATABASE_NAME == "neo4j":
    KG_NAME = "ionchannels"
elif NEO4J_DATABASE_NAME == "prokino-kg":
    KG_NAME = "prokino"

cypher_results_path = f'./results/{KG_NAME}/{CYPHER_RESULTS}'
baseline_results_path = f'./results/{KG_NAME}/{BASELINE_RESULTS}'
output_metrics_path = cypher_results_path.replace('-with_results.json', '-metrics(precision_recall_f1).json')

def calculate_precision_recall_f1(predicted_results, baseline_results):
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predicted_results: List of results from the ablation study
        baseline_results: List of ground truth results
        
    Returns:
        dict: Metrics including precision, recall, and F1 score
    """
    # Empty predictions or baseline case
    if not predicted_results or not baseline_results:
        if not predicted_results and not baseline_results:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        elif not predicted_results:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Convert to sets of strings for comparison
    predicted_set = set()
    for result in predicted_results:
        result_str = json.dumps(result, sort_keys=True)
        predicted_set.add(result_str)
    
    baseline_set = set()
    for result in baseline_results:
        result_str = json.dumps(result, sort_keys=True)
        baseline_set.add(result_str)
    
    # Calculate metrics
    tp = len(predicted_set.intersection(baseline_set))
    fp = len(predicted_set - baseline_set)
    fn = len(baseline_set - predicted_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }

# Load the benchmark results
try:
    with open(cypher_results_path, 'r') as file:
        benchmark_data = json.load(file)
except json.JSONDecodeError as e:
    print(f"Failed to decode benchmark file: {e}")
    exit(1)

# Load the baseline results if available
baseline_data = None
original_baseline_data = None
try:
    with open(baseline_results_path, 'r') as file:
        original_baseline_data = json.load(file)
        print(f"Loaded baseline metrics file with keys: {original_baseline_data.keys()}")
        
        # Check if this is a metrics file with individual and aggregated keys
        if isinstance(original_baseline_data, dict) and "individual" in original_baseline_data:
            print("Found metrics file format with 'individual' key")
            # Try to find the original results file
            baseline_with_results_path = baseline_results_path.replace('-metrics(precision_recall_f1).json', '-with_results.json')
            try:
                with open(baseline_with_results_path, 'r') as baseline_file:
                    baseline_data = json.load(baseline_file)
                    print(f"Successfully loaded baseline results from: {baseline_with_results_path}")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Could not load original baseline results: {e}. Will use ablation 'with_instructions' as baseline.")
        else:
            # This is already a results file
            baseline_data = original_baseline_data
            print("Using loaded file directly as baseline results")
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"Failed to load baseline file: {e}. Using ablation 'with_instructions' as baseline.")

# Calculate metrics for each ablation type
metrics_data = []

for entry in benchmark_data:
    title = entry.get("title", "Unknown")
    query = entry.get("description", "")
    results = entry.get("results", {})
    
    entry_metrics = {
        "title": title,
        "query": query,
        "metrics": {}
    }
    
    # Determine the baseline results
    baseline_results_for_query = []
    if baseline_data:
        # Find the matching query in the baseline data
        for baseline_entry in baseline_data:
            if baseline_entry.get("title") == title:
                version_key = list(baseline_entry.get("results", {}).keys())[0]  # Usually "version_1"
                baseline_results_for_query = baseline_entry.get("results", {}).get(version_key, {}).get("cypher_result", [])
                break
    
    # If no baseline was found, use "with_instructions" as the baseline
    if not baseline_results_for_query:
        with_instructions_results = results.get("with_instructions", {}).get("cypher_result", [])
        baseline_results_for_query = with_instructions_results
        print(f"Using 'with_instructions' as baseline for query: {title}")
    
    # Calculate metrics for each ablation type
    for ablation_type, ablation_results in results.items():
        cypher_result = ablation_results.get("cypher_result", [])
        metrics = calculate_precision_recall_f1(cypher_result, baseline_results_for_query)
        entry_metrics["metrics"][ablation_type] = metrics
    
    metrics_data.append(entry_metrics)

# Write metrics to file
with open(output_metrics_path, 'w') as file:
    json.dump(metrics_data, file, indent=4)

# Create a summary table
if not metrics_data:
    print("No metrics data to analyze!")
    exit(1)

ablation_types = list(metrics_data[0]["metrics"].keys()) if metrics_data else []
summary_data = {ablation_type: {"precision": [], "recall": [], "f1": []} for ablation_type in ablation_types}

for entry in metrics_data:
    for ablation_type in ablation_types:
        metrics = entry.get("metrics", {}).get(ablation_type, {})
        summary_data[ablation_type]["precision"].append(metrics.get("precision", 0))
        summary_data[ablation_type]["recall"].append(metrics.get("recall", 0))
        summary_data[ablation_type]["f1"].append(metrics.get("f1", 0))

# Calculate average metrics
average_metrics = {}
for ablation_type, metrics in summary_data.items():
    avg_precision = sum(metrics["precision"]) / len(metrics["precision"]) if metrics["precision"] else 0
    avg_recall = sum(metrics["recall"]) / len(metrics["recall"]) if metrics["recall"] else 0
    avg_f1 = sum(metrics["f1"]) / len(metrics["f1"]) if metrics["f1"] else 0
    
    average_metrics[ablation_type] = {
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1
    }

# Display summary table
summary_table = []
headers = ["Ablation Type", "Avg Precision", "Avg Recall", "Avg F1"]

for ablation_type, metrics in average_metrics.items():
    summary_table.append([
        ablation_type,
        f"{metrics['avg_precision']:.4f}",
        f"{metrics['avg_recall']:.4f}",
        f"{metrics['avg_f1']:.4f}"
    ])

print("\nAblation Study Summary:")
print(tabulate(summary_table, headers=headers, tablefmt="grid"))

# Create and save a CSV file with detailed per-query results
csv_data = []

for entry in metrics_data:
    title = entry["title"]
    for ablation_type in ablation_types:
        metrics = entry.get("metrics", {}).get(ablation_type, {})
        row = {
            "Query": title,
            "Ablation Type": ablation_type,
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "F1": metrics.get("f1", 0),
            "True Positives": metrics.get("true_positives", 0),
            "False Positives": metrics.get("false_positives", 0), 
            "False Negatives": metrics.get("false_negatives", 0)
        }
        csv_data.append(row)

# Create DataFrame and save to CSV
df = pd.DataFrame(csv_data)
csv_path = output_metrics_path.replace('.json', '.csv')
df.to_csv(csv_path, index=False)

print(f"\nDetailed metrics saved to: {output_metrics_path}")
print(f"CSV file saved to: {csv_path}")

# Display summary by query type (assuming query titles follow a pattern like A1, B2, etc.)
if any(entry["title"][0].isalpha() for entry in metrics_data):
    query_categories = set(entry["title"][0] for entry in metrics_data if entry["title"][0].isalpha())
    
    print("\nMetrics by Query Category:")
    for category in sorted(query_categories):
        category_entries = [entry for entry in metrics_data if entry["title"][0] == category]
        
        category_table = []
        for ablation_type in ablation_types:
            precisions = [entry["metrics"].get(ablation_type, {}).get("precision", 0) for entry in category_entries]
            recalls = [entry["metrics"].get(ablation_type, {}).get("recall", 0) for entry in category_entries]
            f1s = [entry["metrics"].get(ablation_type, {}).get("f1", 0) for entry in category_entries]
            
            avg_precision = sum(precisions) / len(precisions) if precisions else 0
            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0
            
            category_table.append([
                ablation_type,
                f"{avg_precision:.4f}",
                f"{avg_recall:.4f}",
                f"{avg_f1:.4f}"
            ])
        
        print(f"\nCategory {category}:")
        print(tabulate(category_table, headers=headers, tablefmt="grid"))