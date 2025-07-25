# “Figure 6 should be resized and made more readable by adding data-labels (Bar chart may be more suitable, though).”


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

# -------- Visualization (bar version) --------
data = []
for kg, hop_scores in f1_scores_by_kg.items():
    for complexity, f1 in hop_scores.items():
        data.append({'Knowledge Graph': kg,
                     'Query Complexity': complexity,
                     'F1 Score': f1})
df = pd.DataFrame(data)

# Order categories
df['Query Complexity'] = pd.Categorical(
    df['Query Complexity'],
    categories=hop_order,
    ordered=True
)

# ---------- Imports & rcParams ----------------------------------------------
import json, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

plt.rcParams.update({
    "figure.dpi": 300,               # high-resolution output
    "savefig.dpi": 600,              # even higher when saving
    "font.family": "serif",          # match manuscript text
    "font.serif": ["Times New Roman", "Times"],  # common journal request
    "axes.spines.top": False,        # cleaner axes
    "axes.spines.right": False,
})

# ---------- Load or build `df` as before ------------------------------------
# ... (your existing JSON-to-DataFrame code) ...

# Enforce category order
df['Query Complexity'] = pd.Categorical(
    df['Query Complexity'],
    categories=["1-hop (simple)", "2-hop (medium)", "3-hop (complex)"],
    ordered=True
)

# ---------- Plot ------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

palette = sns.color_palette("colorblind", n_colors=df["Knowledge Graph"].nunique())

# ---------- Plot Setup (updated) --------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

sns.barplot(
    data=df,
    x="Query Complexity",
    y="F1 Score",
    hue="Knowledge Graph",
    palette=palette,
    width=0.80,
    edgecolor="black",
    linewidth=0.6,
    ax=ax
)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10)

# Remove all spines (borders)
for spine in ax.spines.values():
    spine.set_visible(False)

# Axis styling
ax.set_xlabel("Query complexity", labelpad=8, fontsize=12, weight="bold")
ax.set_ylabel("F1 score",        labelpad=8, fontsize=12, weight="bold")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linewidth=0.4, linestyle="--", alpha=0.6)

# Legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Knowledge graph", loc="upper center",
          bbox_to_anchor=(0.5, 1.08), ncol=len(labels), frameon=False,
          fontsize=11, title_fontsize=11)

plt.tight_layout()
plt.savefig("./results/f1_trend_by_complexity_bar.png")
plt.show()
