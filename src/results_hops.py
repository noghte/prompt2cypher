
import json
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict, OrderedDict

# Mapping first letter to query complexity
hop_map = {'A': '1-hop (simple)', 'B': '2-hop (medium)', 'C': '3-hop (complex)'}
hop_order = ['1-hop (simple)', '2-hop (medium)', '3-hop (complex)']

# Input files and metadata

input_files = [
    {
        'path': 'results/prokino/benchmark-gpt-4o-mini-2025_04_01-16_14_33_version_1-metrics(precision_recall_f1).json',
        'kg': 'ProKinO',
        'approach': 'P2C'
    },
    {
        'path': 'results/ickg/benchmark-gpt-4o-mini-2025_04_08-13_31_41-metrics(precision_recall_f1).json',
        'kg': 'ICKG',
        'approach': 'P2C'
    }
]

results = []
f1_scores_by_kg = defaultdict(lambda: OrderedDict((hop, None) for hop in hop_order))

for file in input_files:
    with open(file['path']) as f:
        data = json.load(f)

    approach = file['approach']
    kg = file['kg']
    
    # Group entries by query complexity
    buckets = defaultdict(list)
    for entry in data['individual']:
        title = entry['title']
        if not title: continue
        category = hop_map.get(title[0])
        if category:
            buckets[category].append(entry)
    
    for category, entries in buckets.items():
        precision = sum(e['Precision'] for e in entries) / len(entries)
        recall = sum(e['Recall'] for e in entries) / len(entries)
        f1 = sum(e['F1-Score'] for e in entries) / len(entries)
        jaccard = sum(e['Jaccard Similarity'] for e in entries) / len(entries)
        
        results.append({
            'Knowledge Graph': kg,
            'Query Complexity': category,
            'Approach': approach,
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'Jaccard': round(jaccard, 4),
        })
        
        # Store for plotting (latest approach only per KG)
        f1_scores_by_kg[kg][category] = round(f1, 4)

# Sort results
results.sort(key=lambda x: (x['Knowledge Graph'], x['Query Complexity'], x['Approach']))

# Save CSV
# output_csv = './results/query_performance_metrics.csv'
# with open(output_csv, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=['Knowledge Graph', 'Query Complexity', 'Approach', 'Precision', 'Recall', 'F1 Score', 'Jaccard'])
#     writer.writeheader()
#     writer.writerows(results)

# print(f"Saved results to {output_csv}")

# -------- Visualization --------
data = []
for kg, hop_scores in f1_scores_by_kg.items():
    for complexity, f1 in hop_scores.items():
        data.append({'Knowledge Graph': kg, 'Query Complexity': complexity, 'F1 Score': f1})
df = pd.DataFrame(data)

# Set the aesthetic style of the plots
sns.set(style='whitegrid', context='paper', font_scale=1.3)

plt.figure(figsize=(8, 5))
sns.lineplot(
    data=df,
    x='Query Complexity',
    y='F1 Score',
    hue='Knowledge Graph',
    marker='o'
)

# plt.title("Performance Trends Across Query Complexity Levels", fontsize=14)
plt.xlabel("Query Complexity")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.legend(title='Knowledge Graph', loc='best')
plt.tight_layout()
plt.savefig("./results/f1_trend_by_complexity.png", dpi=300)
plt.show()