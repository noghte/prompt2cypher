import json
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

DEFAULT_CYPHER_RESULTS = "ablation-gpt-4o-mini-2025_04_01-17_02_40-benchmark_format-metrics(precision_recall_f1).json"
# The baseline file should already be produced with calculate_scores_precision_recall.py
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
output_metrics_path = cypher_results_path.replace('-metrics(precision_recall_f1).json', '-ablation-analysis.json')

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
        # Debug benchmark data structure
        print("\nDEBUG - Loaded benchmark file structure:")
        print(f"Type: {type(benchmark_data)}")
        if isinstance(benchmark_data, list) and len(benchmark_data) > 0:
            sample_entry = benchmark_data[0]
            print(f"Sample entry keys: {list(sample_entry.keys())}")
            if "results" in sample_entry:
                print(f"Sample results keys: {list(sample_entry['results'].keys())}")
            elif "ablation_results" in sample_entry:
                print(f"Sample ablation_results keys: {list(sample_entry['ablation_results'].keys())}")
            if "metrics" in sample_entry:
                print(f"Sample metrics keys: {list(sample_entry['metrics'].keys())}")
                for ablation_type, metrics in sample_entry["metrics"].items():
                    print(f"  {ablation_type} metrics: {metrics}")
except json.JSONDecodeError as e:
    print(f"Failed to decode benchmark file: {e}")
    exit(1)

# Load the baseline results if available
baseline_data = None
original_baseline_data = None
try:
    with open(baseline_results_path, 'r') as file:
        original_baseline_data = json.load(file)
        print(f"Loaded baseline file: {baseline_results_path}")
        
        # Debug baseline data structure
        print("\nDEBUG - Loaded baseline file structure:")
        print(f"Type: {type(original_baseline_data)}")
        if isinstance(original_baseline_data, dict):
            print(f"Keys: {list(original_baseline_data.keys())}")
            if "individual" in original_baseline_data:
                print("Found 'individual' key")
        elif isinstance(original_baseline_data, list) and len(original_baseline_data) > 0:
            sample_entry = original_baseline_data[0]
            print(f"Sample entry keys: {list(sample_entry.keys())}")
        
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
            # Check if the list items contain "results" with cypher_result
            if isinstance(original_baseline_data, list) and any("results" in item and isinstance(item["results"], dict) for item in original_baseline_data):
                baseline_data = original_baseline_data
                print("Using loaded file directly as baseline results")
            else:
                # This is a metrics file without actual results
                print("Loaded file appears to be a metrics file without actual results")
                # Try to find the corresponding results file
                if baseline_results_path.endswith('-metrics(precision_recall_f1).json'):
                    baseline_with_results_path = baseline_results_path.replace('-metrics(precision_recall_f1).json', '-with_results.json')
                    try:
                        with open(baseline_with_results_path, 'r') as baseline_file:
                            baseline_data = json.load(baseline_file)
                            print(f"Successfully loaded baseline results from: {baseline_with_results_path}")
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"Could not load original baseline results: {e}. Will use ablation 'with_instructions' as baseline.")
except (json.JSONDecodeError, FileNotFoundError) as e:
    print(f"Failed to load baseline file: {e}. Using ablation 'with_instructions' as baseline.")

# Let's check if the benchmark data is already in metrics format
is_metrics_format = False
if benchmark_data and len(benchmark_data) > 0:
    sample_entry = benchmark_data[0]
    if "metrics" in sample_entry and isinstance(sample_entry["metrics"], dict):
        is_metrics_format = True
        print("\nDetected benchmark file is already in metrics format, using it directly")

if is_metrics_format:
    # Data is already in metrics format, use it directly
    metrics_data = benchmark_data
    print(f"Loaded {len(metrics_data)} entries with metrics")
    
    # Print sample metrics from first entry
    if metrics_data and "metrics" in metrics_data[0]:
        sample_metrics = metrics_data[0]["metrics"]
        print(f"Sample metrics keys: {list(sample_metrics.keys())}")
        for key, value in list(sample_metrics.items())[:2]:  # Show first 2 metrics
            print(f"  {key}: {value}")
else:
    # Need to calculate metrics from results
    print("\nCalculating metrics from benchmark results")
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
                    # Check if this is a results file with version_1 or a metrics file
                    if "results" in baseline_entry and isinstance(baseline_entry["results"], dict):
                        version_key = list(baseline_entry.get("results", {}).keys())[0]  # Usually "version_1"
                        baseline_results_for_query = baseline_entry.get("results", {}).get(version_key, {}).get("cypher_result", [])
                    # If it's a metrics file, we can't use it directly
                    else:
                        print(f"Found metrics file for {title}, but it doesn't contain results. Using 'with_instructions' as baseline.")
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

# Print the first few entries to verify data structure
print("\nDEBUG - First few entries in metrics_data:")
for i, entry in enumerate(metrics_data[:2]):
    print(f"Entry {i+1} - Title: {entry.get('title')}")
    for ablation_type, metrics in entry.get("metrics", {}).items():
        print(f"  {ablation_type}: P={metrics.get('precision', 0):.2f}, R={metrics.get('recall', 0):.2f}, F1={metrics.get('f1', 0):.2f}")

# Make sure we're properly extracting the ablation types
# The data structure should have metrics like: entry["metrics"]["with_instructions"], etc.
all_ablation_types = set()

# Collect all unique ablation types across all entries
for entry in metrics_data:
    if "metrics" in entry and isinstance(entry["metrics"], dict):
        entry_ablation_types = entry["metrics"].keys()
        all_ablation_types.update(entry_ablation_types)

if all_ablation_types:
    ablation_types = list(all_ablation_types)
    print(f"\nDEBUG - Collected ablation types from all entries: {ablation_types}")
    
    # For each ablation type, count how many entries have it
    ablation_counts = {}
    for ablation_type in ablation_types:
        count = sum(1 for entry in metrics_data if ablation_type in entry.get("metrics", {}))
        ablation_counts[ablation_type] = count
        print(f"  {ablation_type}: Found in {count}/{len(metrics_data)} entries")
else:
    print("\nWARNING: Could not find any ablation types in the entries")
    ablation_types = []

# Aggregate metrics by ablation type
summary_data = {ablation_type: {"precision": [], "recall": [], "f1": []} for ablation_type in ablation_types}

# Count how many queries have non-zero metrics for each ablation type
non_zero_counts = {ablation_type: 0 for ablation_type in ablation_types}

for entry in metrics_data:
    for ablation_type in ablation_types:
        metrics = entry.get("metrics", {}).get(ablation_type, {})
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        f1 = metrics.get("f1", 0)
        
        summary_data[ablation_type]["precision"].append(precision)
        summary_data[ablation_type]["recall"].append(recall)
        summary_data[ablation_type]["f1"].append(f1)
        
        # Count if this query has non-zero metrics
        if f1 > 0:
            non_zero_counts[ablation_type] += 1

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
    
# Print the summary data to verify calculations
print("\nDEBUG - Summary Data:")
for ablation_type, metrics in average_metrics.items():
    print(f"{ablation_type}: P={metrics['avg_precision']:.4f}, R={metrics['avg_recall']:.4f}, F1={metrics['avg_f1']:.4f}")

print("\nDEBUG - Non-zero counts:")
for ablation_type, count in non_zero_counts.items():
    print(f"{ablation_type}: {count}/{len(metrics_data)} ({count/len(metrics_data)*100:.2f}%)")

# Display summary table
summary_table = []
headers = ["Ablation Type", "Avg Precision", "Avg Recall", "Avg F1", "Success Rate"]
total_queries = len(metrics_data)

for ablation_type, metrics in average_metrics.items():
    success_rate = non_zero_counts[ablation_type] / total_queries if total_queries > 0 else 0
    summary_table.append([
        ablation_type,
        f"{metrics['avg_precision']:.4f}",
        f"{metrics['avg_recall']:.4f}",
        f"{metrics['avg_f1']:.4f}",
        f"{success_rate:.2%} ({non_zero_counts[ablation_type]}/{total_queries})"
    ])

print("\nAblation Study Summary:")
print(tabulate(summary_table, headers=headers, tablefmt="grid"))

# Create and save a CSV file with detailed per-query results
# csv_data = []

# for entry in metrics_data:
#     title = entry["title"]
#     for ablation_type in ablation_types:
#         metrics = entry.get("metrics", {}).get(ablation_type, {})
#         f1 = metrics.get("f1", 0)
#         success = 1 if f1 > 0 else 0
#         row = {
#             "Query": title,
#             "Ablation Type": ablation_type,
#             "Precision": metrics.get("precision", 0),
#             "Recall": metrics.get("recall", 0),
#             "F1": f1,
#             "Success": success,
#             "True Positives": metrics.get("true_positives", 0),
#             "False Positives": metrics.get("false_positives", 0), 
#             "False Negatives": metrics.get("false_negatives", 0)
#         }
#         csv_data.append(row)

# # Create DataFrame and save to CSV
# df = pd.DataFrame(csv_data)
# csv_path = output_metrics_path.replace('.json', '.csv')
# df.to_csv(csv_path, index=False)

# print(f"\nDetailed metrics saved to: {output_metrics_path}")
# print(f"CSV file saved to: {csv_path}")

# Display summary by query type (assuming query titles follow a pattern like A1, B2, etc.)
if any(entry["title"][0].isalpha() for entry in metrics_data):
    query_categories = set(entry["title"][0] for entry in metrics_data if entry["title"][0].isalpha())
    
    print("\nMetrics by Query Category:")
    for category in sorted(query_categories):
        category_entries = [entry for entry in metrics_data if entry["title"][0] == category]
        category_size = len(category_entries)
        
        category_table = []
        for ablation_type in ablation_types:
            precisions = [entry["metrics"].get(ablation_type, {}).get("precision", 0) for entry in category_entries]
            recalls = [entry["metrics"].get(ablation_type, {}).get("recall", 0) for entry in category_entries]
            f1s = [entry["metrics"].get(ablation_type, {}).get("f1", 0) for entry in category_entries]
            
            # Count successful queries (non-zero F1 score)
            successful = sum(1 for f1 in f1s if f1 > 0)
            success_rate = successful / category_size if category_size > 0 else 0
            
            avg_precision = sum(precisions) / len(precisions) if precisions else 0
            avg_recall = sum(recalls) / len(recalls) if recalls else 0
            avg_f1 = sum(f1s) / len(f1s) if f1s else 0
            
            category_table.append([
                ablation_type,
                f"{avg_precision:.4f}",
                f"{avg_recall:.4f}",
                f"{avg_f1:.4f}",
                f"{success_rate:.2%} ({successful}/{category_size})"
            ])
        
        print(f"\nCategory {category}:")
        print(tabulate(category_table, headers=headers, tablefmt="grid"))

# Calculate component impact statistics
print("\nComponent Impact Analysis:")

# Define what each ablation removes from the full method
component_mapping = {
    "instructions": ("with_instructions", "without_instructions"),
    "schema_comments": ("with_instructions", "without_schema_comments"),
    "relevant_nodes": ("with_instructions", "without_relevant_nodes")
}

# DEBUG: Print ablation metrics to understand data
print("\nDEBUG - Ablation Metrics:")
for ablation_type, metrics in average_metrics.items():
    print(f"{ablation_type}: Precision={metrics.get('avg_precision', 0):.4f}, "
          f"Recall={metrics.get('avg_recall', 0):.4f}, "
          f"F1={metrics.get('avg_f1', 0):.4f}")

print(f"\nDEBUG - Successful Queries:")
for ablation_type, count in non_zero_counts.items():
    print(f"{ablation_type}: {count}/{total_queries} "
          f"({count/total_queries*100:.2f}%)")

component_impact_table = []
component_headers = ["Component", "Overall Impact", "Precision Impact", "Recall Impact", "F1 Impact", "Success Rate Impact"]

# Process each entry to find per-query differences (to help diagnose issues)
# This different approach should give us meaningful impact values
component_query_impacts = {component: {"precision": [], "recall": [], "f1": [], "success": []} 
                           for component in component_mapping.keys()}

print("\nDEBUG - Processing per-query component impacts:")
print(f"Number of entries to process: {len(metrics_data)}")
print(f"Component mapping: {component_mapping}")

query_count = 0
impact_count = 0

for entry in metrics_data:
    query_title = entry.get("title", "Unknown")
    query_count += 1
    
    # Debug the first few entries in detail
    if query_count <= 3:
        print(f"\nProcessing query {query_title}:")
        print(f"Available metrics keys: {list(entry.get('metrics', {}).keys())}")
    
    # Process each component's impact
    for component, (baseline_type, ablation_type) in component_mapping.items():
        # Check if both baseline and ablation types exist in the metrics
        if baseline_type in entry.get("metrics", {}) and ablation_type in entry.get("metrics", {}):
            baseline_metrics = entry.get("metrics", {}).get(baseline_type, {})
            ablation_metrics = entry.get("metrics", {}).get(ablation_type, {})
            
            # Debug metrics for the first few entries
            if query_count <= 3:
                print(f"  {component}: Baseline ({baseline_type}) - P={baseline_metrics.get('precision', 0):.2f}, R={baseline_metrics.get('recall', 0):.2f}, F1={baseline_metrics.get('f1', 0):.2f}")
                print(f"  {component}: Ablation ({ablation_type}) - P={ablation_metrics.get('precision', 0):.2f}, R={ablation_metrics.get('recall', 0):.2f}, F1={ablation_metrics.get('f1', 0):.2f}")
            
            # Calculate per-query impacts
            precision_impact = baseline_metrics.get("precision", 0) - ablation_metrics.get("precision", 0)
            recall_impact = baseline_metrics.get("recall", 0) - ablation_metrics.get("recall", 0)
            f1_impact = baseline_metrics.get("f1", 0) - ablation_metrics.get("f1", 0)
            
            # Calculate success impact (1 if successful, 0 if not)
            baseline_success = 1 if baseline_metrics.get("f1", 0) > 0 else 0
            ablation_success = 1 if ablation_metrics.get("f1", 0) > 0 else 0
            success_impact = baseline_success - ablation_success
            
            # Add to per-component impacts
            component_query_impacts[component]["precision"].append(precision_impact)
            component_query_impacts[component]["recall"].append(recall_impact)
            component_query_impacts[component]["f1"].append(f1_impact)
            component_query_impacts[component]["success"].append(success_impact)
            
            # Count non-zero impacts
            if abs(f1_impact) > 0.001 or abs(success_impact) > 0.001:
                impact_count += 1
                print(f"Query {query_title}, Component {component}: F1 impact={f1_impact:.4f}, Success impact={success_impact}")
        else:
            if query_count <= 3:
                print(f"  WARNING: Missing metrics for {component}: {baseline_type} or {ablation_type} not found")

print(f"\nProcessed {query_count} queries and found {impact_count} non-zero impacts")

# Calculate average impacts per component
for component, impacts in component_query_impacts.items():
    avg_precision_impact = sum(impacts["precision"]) / len(impacts["precision"]) if impacts["precision"] else 0
    avg_recall_impact = sum(impacts["recall"]) / len(impacts["recall"]) if impacts["recall"] else 0
    avg_f1_impact = sum(impacts["f1"]) / len(impacts["f1"]) if impacts["f1"] else 0
    success_impact_rate = sum(impacts["success"]) / len(impacts["success"]) if impacts["success"] else 0
    
    print(f"\nDEBUG - Component {component} Average Impacts:")
    print(f"  Precision: {avg_precision_impact:.4f}")
    print(f"  Recall: {avg_recall_impact:.4f}")
    print(f"  F1: {avg_f1_impact:.4f}")
    print(f"  Success Rate: {success_impact_rate:.2%}")
    
    component_impact_table.append([
        component,
        f"{avg_f1_impact:.4f}",
        f"{avg_precision_impact:.4f}",
        f"{avg_recall_impact:.4f}",
        f"{avg_f1_impact:.4f}",
        f"{success_impact_rate:.2%}"
    ])

# Sort by overall impact
component_impact_table.sort(key=lambda x: float(x[1]), reverse=True)
print(tabulate(component_impact_table, headers=component_headers, tablefmt="grid"))

# Calculate component impact by category
if any(entry["title"][0].isalpha() for entry in metrics_data):
    query_categories = set(entry["title"][0] for entry in metrics_data if entry["title"][0].isalpha())
    
    print("\nComponent Impact by Query Category:")
    
    for category in sorted(query_categories):
        category_entries = [entry for entry in metrics_data if entry["title"][0] == category]
        category_size = len(category_entries)
        
        print(f"\nCategory {category} - {category_size} queries")
        
        # Calculate per-query component impacts for this category
        category_component_impacts = {component: {"precision": [], "recall": [], "f1": [], "success": []} 
                                    for component in component_mapping.keys()}
        
        # Process each entry in this category
        category_impact_count = 0
        for entry in category_entries:
            query_title = entry.get("title", "Unknown")
            for component, (baseline_type, ablation_type) in component_mapping.items():
                # Check if both baseline and ablation types exist in the metrics
                if baseline_type in entry.get("metrics", {}) and ablation_type in entry.get("metrics", {}):
                    baseline_metrics = entry.get("metrics", {}).get(baseline_type, {})
                    ablation_metrics = entry.get("metrics", {}).get(ablation_type, {})
                    
                    # Calculate per-query impacts
                    precision_impact = baseline_metrics.get("precision", 0) - ablation_metrics.get("precision", 0)
                    recall_impact = baseline_metrics.get("recall", 0) - ablation_metrics.get("recall", 0)
                    f1_impact = baseline_metrics.get("f1", 0) - ablation_metrics.get("f1", 0)
                    
                    # Calculate success impact (1 if successful, 0 if not)
                    baseline_success = 1 if baseline_metrics.get("f1", 0) > 0 else 0
                    ablation_success = 1 if ablation_metrics.get("f1", 0) > 0 else 0
                    success_impact = baseline_success - ablation_success
                    
                    # Add to category-specific component impacts
                    category_component_impacts[component]["precision"].append(precision_impact)
                    category_component_impacts[component]["recall"].append(recall_impact)
                    category_component_impacts[component]["f1"].append(f1_impact)
                    category_component_impacts[component]["success"].append(success_impact)
                    
                    # Print individual query impacts for debugging, but only if there is an impact
                    if abs(f1_impact) > 0.001 or abs(success_impact) > 0.001:
                        category_impact_count += 1
                        print(f"  Query {query_title}, Component {component}: F1 impact={f1_impact:.4f}, Success impact={success_impact}")
                else:
                    print(f"  WARNING: Category {category}, Query {query_title} - Missing metrics for {component}: {baseline_type} or {ablation_type} not found")
        
        print(f"\n  Category {category}: Found {category_impact_count} non-zero impacts")
        
        # Calculate average impacts for each component in this category
        component_category_table = []
        
        for component, impacts in category_component_impacts.items():
            avg_precision_impact = sum(impacts["precision"]) / len(impacts["precision"]) if impacts["precision"] else 0
            avg_recall_impact = sum(impacts["recall"]) / len(impacts["recall"]) if impacts["recall"] else 0
            avg_f1_impact = sum(impacts["f1"]) / len(impacts["f1"]) if impacts["f1"] else 0
            success_impact_rate = sum(impacts["success"]) / len(impacts["success"]) if impacts["success"] else 0
            
            print(f"\n  DEBUG - Category {category}, Component {component} Average Impacts:")
            print(f"    Precision: {avg_precision_impact:.4f}")
            print(f"    Recall: {avg_recall_impact:.4f}")
            print(f"    F1: {avg_f1_impact:.4f}")
            print(f"    Success Rate: {success_impact_rate:.2%}")
            
            component_category_table.append([
                component,
                f"{avg_f1_impact:.4f}",
                f"{avg_precision_impact:.4f}",
                f"{avg_recall_impact:.4f}",
                f"{avg_f1_impact:.4f}",
                f"{success_impact_rate:.2%}"
            ])
        
        component_category_table.sort(key=lambda x: float(x[1]), reverse=True)
        print(f"\nCategory {category} - Component Impact:")
        print(tabulate(component_category_table, headers=component_headers, tablefmt="grid"))