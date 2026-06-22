import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
})

FIELDS = [
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10", "C11", "C12"
]

COLS = [
    "Mistral\nISR",
    "Mistral\nPR",
    "Mistral\nMPR",
    "LLaMA\nISR",
    "LLaMA\nPR",
    "LLaMA\nMPR",
    "Qwen\nISR",
    "Qwen\nPR",
    "Qwen\nMPR",
    "GPT\nISR"
]

DATASETS = {
    "LDBC": {
        "GPT\nISR": [
            [1,1,1], [1,1,1], [1,1,1], [1,1,1],
            [2,2,2], [1,1,1], [4,4,4], [2,2,2],
            [2,2,2], [1,1,1], [4,4,4], [1,1,1],
        ],
        "Mistral\nISR": [
            [1,1,1], [1,1,1], [3,3,3], [4,4,4],
            [1,1,1], [1,1,1], [2,2,2], [1,1,1],
            [2,2,2], [1,1,1], [3,3,3], [3,3,3],
        ],
        "Mistral\nPR": [
            [1,1,1], [1,1,1], [1,1,1], [1,1,1],
            [1,1,1], [1,1,1], [2,2,2], [3,3,3],
            [2,2,2], [3,3,3], [4,4,4], [3,3,3],
        ],
        "Mistral\nMPR": [
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [4, 4, 4],
            [1, 1, 1], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3],
        ],
        "LLaMA\nISR": [
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
        ],
        "LLaMA\nPR": [
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
        ],
        "LLaMA\nMPR": [
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [4, 4, 4],
            [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4],
            [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4],
        ],

        "Qwen\nISR": [
            [1,	1,	1], [1, 3, 1], [1, 4, 4], [4, 4, 4],
            [1,	1,	1], [1,	1,	1], [2, 2, 4], [1,	1,	1],
            [1,	1,	1], [1,	1,	1], [1,	2,	2], [3, 3, 1],
        ],
        "Qwen\nPR": [
            [3, 1, 1], [3, 1, 1], [3, 1, 1], [3, 4, 4],
            [3, 1, 1], [3, 1, 1], [3, 3, 1], [3, 1, 1],
            [3, 2, 1], [3, 1, 1], [3, 1, 1], [3, 1, 1],
        ],
        "Qwen\nMPR": [
            [1, 1, 1], [1, 1, 1], [4, 4, 4], [4, 4, 4],
            [1, 1, 1], [2, 2, 2], [4, 4, 4], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [3, 3, 3],
        ],


    },

    "Knows": {
        "GPT\nISR": [
            [1,1,1], [1,1,1], [3,3,3], [3,3,3],
            [2,2,2], [1,1,1], [4,4,4], [1,1,1],
            [4,4,4], [2,2,2], [4,4,4], [1,1,1],
        ],
        "Mistral\nISR": [
            [2,2,2], [3,3,3], [3,3,3], [3,3,3],
            [1,1,1], [2,2,2], [2,2,2], [1,1,1],
            [3,3,3], [3,3,3], [4,4,4], [3,3,3],
        ],
        "Mistral\nPR": [
            [2,2,2], [3,3,3], [3,3,3], [3,3,3],
            [1,1,1], [2,2,2], [2,2,2], [1,1,1],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
        ],
        "Mistral\nMPR": [
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2],
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [4, 4, 4], [4, 4, 4], [3, 3, 3],
        ],
        "LLaMA\nISR": [
            [1,1,1], [1,1,1], [3,3,3], [3,3,3],
            [1,1,1], [1,1,1], [3,3,3], [1,1,1],
            [3,3,3], [2,2,2], [4,4,4], [3,3,3],
        ],
        "LLaMA\nPR": [
            [1,1,1], [1,1,1], [3,3,3], [3,3,3],
            [1,1,1], [1,1,1], [3,3,3], [1,1,1],
            [4,4,4], [2,2,2], [4,2,2], [3,3,1],
        ],
        "LLaMA\nMPR": [
            [1, 1, 1], [1, 1, 1], [3, 3, 3], [3, 3, 3],
            [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4],
            [2, 2, 2], [4, 4, 4], [4, 4, 4], [4, 4, 4],
        ],
        "Qwen\nISR": [
            [3, 1, 1], [1, 3, 3], [4, 3, 3], [4, 4, 4],
            [1, 1, 1], [2, 1, 1], [4, 4, 1], [1, 1, 1],
            [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3],
        ],

        "Qwen\nPR": [
            [1, 1, 1], [3, 3, 3], [3, 3, 3], [4, 4, 4],
            [1, 1, 1], [1, 1, 1], [4, 1, 1], [2, 2, 2],
            [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 3],
        ],

        "Qwen\nMPR": [
            [2, 2, 2], [3, 3, 3], [3, 3, 3], [4, 4, 4],
            [1, 1, 1], [2, 2, 2], [3, 3, 3], [2, 2, 2],
            [2, 2, 2], [4, 4, 4], [2, 2, 2], [4, 4, 4],
        ],
    },

    "DBLP": {
        "GPT\nISR": [
            [2,2,2], [1,1,1], [1,3,3], [1,3,3],
            [1,1,1], [1,1,1], [4,4,4], [2,2,2],
            [2,2,2], [2,1,3], [2,2,2], [1,1,1],
        ],
        "Mistral\nISR": [
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
        ],
        "Mistral\nPR": [
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,3,3], [3,3,3],
            [3,3,3], [3,3,3], [3,2,3], [3,3,3],
        ],
        "Mistral\nMPR": [
            [2, 2, 2], [1, 1, 1], [1, 1, 1],[2, 2, 2],
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
        ],
        "LLaMA\nISR": [
            [3,3,3], [1,1,1], [3,3,3], [3,3,3],
            [1,1,1], [1,1,1], [4,4,4], [1,1,1],
            [3,3,3], [3,3,3], [2,2,2], [3,3,3],
        ],
        "LLaMA\nPR": [
            [3,3,3], [1,1,1], [3,3,3], [3,3,3],
            [2,2,2], [1,1,1], [4,2,2], [2,1,1],
            [3,3,3], [2,3,3], [2,2,2], [3,3,3],
        ],
        "LLaMA\nMPR": [
            [2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4],
            [1, 1, 1], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            [1, 1, 1], [4, 4, 4], [2, 2, 2], [3, 3, 3],
        ],
        "Qwen\nISR": [
            [3, 3, 3], [2, 2, 2], [3, 3, 3], [3, 3, 3],
            [1, 1, 1], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            [3, 3, 3], [4, 4, 4], [1, 1, 1], [3, 3, 3],
        ],

        "Qwen\nPR": [
            [3, 3, 3], [2, 2, 2], [3, 3, 3], [3, 3, 3],
            [1, 1, 1], [2, 2, 2], [1, 1, 1], [1, 1, 1],
            [3, 3, 3], [4, 4, 4], [1, 1, 1], [3, 3, 3],
        ],

        "Qwen\nMPR": [
            [2, 2, 2], [2, 2, 2], [3, 3, 3], [4, 4, 4],
            [1, 1, 1], [1, 1, 1], [4, 4, 4], [1, 1, 1],
            [1, 1, 1], [3, 3, 3], [2, 2, 2], [4, 4, 4],
        ],
    }
}

def majority_label(labels):
    cnt = Counter(labels)
    max_freq = max(cnt.values())
    tied = [k for k, v in cnt.items() if v == max_freq]
    return max(tied)  # pessimistic tie-break: worse label wins

def acceptable(label):
    return 1 if label in (1, 2) else 0

data = []
for field_idx in range(len(FIELDS)):
    row = []
    for col in COLS:
        vals = []
        for dataset_name in DATASETS:
            maj = majority_label(DATASETS[dataset_name][col][field_idx])
            vals.append(acceptable(maj))
        row.append(sum(vals) / len(vals))
    data.append(row)

data = np.array(data, dtype=float)

fig, ax = plt.subplots(figsize=(7.2, 4.8))

masked = np.ma.masked_invalid(data)
cmap = plt.cm.Greys.copy()
cmap.set_bad(color="lightgray")

im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(np.arange(len(COLS)))
ax.set_yticks(np.arange(len(FIELDS)))
ax.set_xticklabels(COLS)
ax.set_yticklabels(FIELDS)

plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
ax.tick_params(axis='both', which='both', length=0)

ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(FIELDS), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=0.6)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Correct Ratio (Cat. 1 or 2)", fontsize=9)

# ax.set_xlabel("Model and Prompting Strategy")
ax.set_ylabel("PG-Sheet Contents")

plt.tight_layout()
plt.savefig("pg_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig("pg_heatmap.pdf", bbox_inches="tight")
plt.show()

