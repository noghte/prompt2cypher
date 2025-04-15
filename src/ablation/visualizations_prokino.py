import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

# Set the aesthetics for the plots
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Function to load the ablation study data
def load_ablation_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to calculate raw F1 scores (Table 4 approach)
def calculate_raw_scores(data):
    components = ["without_instructions", "without_schema_comments", "without_relevant_nodes"]
    results = {}

    for component in components:
        scores = []
        for entry in data:
            if component in entry["metrics"]:
                scores.append(entry["metrics"][component]["f1"])
        
        results[component] = np.mean(scores) if scores else 0
    
    return results

# Function to calculate impact values (Table 5 approach)
def calculate_impact_values(data):
    # Group queries by category (A, B, C)
    categories = {}
    for entry in data:
        category = entry["title"][0]  # First character of title
        if category not in categories:
            categories[category] = []
        categories[category].append(entry)
    
    # Calculate impacts for each component and category
    components = ["instructions", "schema_comments", "relevant_nodes"]
    results = {}
    
    for category, entries in categories.items():
        results[category] = {}
        
        for component in components:
            impacts = []
            
            for entry in entries:
                baseline = entry["metrics"].get("with_instructions", {}).get("f1", 0)
                ablated = entry["metrics"].get(f"without_{component}", {}).get("f1", 0)
                
                # Impact = baseline - ablated (positive means removing hurts)
                impact = baseline - ablated
                impacts.append(impact)
            
            results[category][component] = np.mean(impacts) if impacts else 0
    
    return results

# Function to create visualizations
def visualize_ablation_study(data, output_path=None):
    # Calculate raw scores and impact values
    raw_scores = calculate_raw_scores(data)
    impact_values = calculate_impact_values(data)
    
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Plot raw F1 scores (Table 4)
    components = list(raw_scores.keys())
    scores = list(raw_scores.values())
    
    # Replace "with_" and "without_" for better labels
    clean_labels = [c.replace("without_", "Without ") for c in components]
    
    sns.barplot(x=clean_labels, y=scores, palette="Blues_d", ax=axes[0])
    axes[0].set_title("Raw F1 Scores (Table 4 Approach)", fontsize=14)
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Average F1 Score")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, score in enumerate(scores):
        axes[0].text(i, score + 0.02, f"{score:.3f}", ha='center')
    
    # 2. Plot impact values by category (Table 5)
    # Transform the impact values to a format suitable for seaborn
    impact_data = []
    for category, components in impact_values.items():
        for component, impact in components.items():
            impact_data.append({
                "Category": category,
                "Component": component.replace("_", " ").title(),
                "Impact": impact
            })
    
    impact_df = pd.DataFrame(impact_data)
    
    # Create a grouped bar chart for impact values
    sns.barplot(x="Component", y="Impact", hue="Category", data=impact_df, palette="Set2", ax=axes[1])
    axes[1].set_title("F1 Score Impact by Query Category (Table 5 Approach)", fontsize=14)
    axes[1].set_xlabel("Component Removed")
    axes[1].set_ylabel("Impact Value (Baseline - Ablated)")
    axes[1].set_ylim(0, 1.0)
    
    # Add value labels on bars
    for i, bar in enumerate(axes[1].patches):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02,
            f"{height:.2f}",
            ha='center'
        )
    
    plt.tight_layout()
    plt.suptitle("ProKinO", fontsize=16)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Example for individual query visualization (to understand the contradiction)
def visualize_query_impacts(data, output_path=None):
    # Extract impact values for each query
    component = "relevant_nodes"  # Focus on the component with contradiction
    query_impacts = []
    
    for entry in data:
        query_title = entry["title"]
        category = query_title[0]
        
        baseline = entry["metrics"].get("with_instructions", {}).get("f1", 0)
        ablated = entry["metrics"].get(f"without_{component}", {}).get("f1", 0)
        
        impact = baseline - ablated
        query_impacts.append({
            "Query": query_title,
            "Category": category,
            "Impact": impact,
            "Baseline F1": baseline,
            "Ablated F1": ablated
        })
    
    # Convert to DataFrame
    impacts_df = pd.DataFrame(query_impacts)
    
    # Sort alphabetically by query ID (A1, A2, ..., B1, ...)
    impacts_df['Query'] = pd.Categorical(
        impacts_df['Query'], 
        categories=natsorted(impacts_df['Query'].unique()),
        ordered=True
    )
    impacts_df = impacts_df.sort_values('Query')
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Create a bar plot showing the impact for each query
    ax = sns.barplot(x="Query", y="Impact", hue="Category", data=impacts_df, palette="Set2")
    
    plt.title(f"Per-Query Impact of Removing 'Relevant Nodes' Component", fontsize=16)
    plt.xlabel("Query")
    plt.ylabel("Impact Value (Baseline F1 - Ablated F1)")
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Add a horizontal line for the average impact
    avg_impact = impacts_df["Impact"].mean()
    plt.axhline(y=avg_impact, color='g', linestyle='--', label=f"Average Impact: {avg_impact:.3f}")
    
    # Add value labels on bars
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 if height >= 0 else height - 0.08,
            f"{height:.2f}",
            ha='center'
        )
    
    # Add values showing the exact F1 scores of baseline and ablated for each query
    for i, row in impacts_df.iterrows():
        y_pos = -0.05
        plt.text(
            i, y_pos,
            f"B:{row['Baseline F1']:.1f} A:{row['Ablated F1']:.1f}",
            ha='center', va='top', rotation=90,
            fontsize=8, alpha=0.7,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.5)
        )
    
    plt.legend(title="Query Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

# Main function
def main():
    # Path to your ablation study JSON file
    file_path = "results/prokino/ablation-gpt-4o-mini-2025_04_01-17_02_40-benchmark_format-metrics(precision_recall_f1).json"
    
    # Load the data
    ablation_data = load_ablation_data(file_path)
    
    # Create visualizations
    visualize_ablation_study(ablation_data, "prokino_ablation_study.png")
    # visualize_query_impacts(ablation_data, "prokino_query_impacts.png")

if __name__ == "__main__":
    main()