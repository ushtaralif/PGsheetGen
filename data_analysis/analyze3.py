import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from collections import Counter

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

FIELDS = [
    "C1", "C2", "C3", "C4", "C5", "C6",
    "C7", "C8", "C9", "C10", "C11", "C12"
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
            [1, 1, 1], [1, 3, 1], [1, 4, 4], [4, 4, 4],
            [1, 1, 1], [1, 1, 1], [2, 2, 4], [1, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 2, 2], [3, 3, 1],
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
            [2, 2, 2], [1, 1, 1], [1, 1, 1], [2, 2, 2],
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
    return max(tied)  # pessimistic tie-break

def correct_percentage(setting_matrix):
    majority = [majority_label(triple) for triple in setting_matrix]
    good = sum(1 for x in majority if x in (1, 2))
    return good / len(majority) * 100.0

datasets = ["LDBC", "Knows", "DBLP"]

gpt_isr      = np.array([correct_percentage(DATASETS[d]["GPT\nISR"])      for d in datasets], dtype=float)
mistral_isr  = np.array([correct_percentage(DATASETS[d]["Mistral\nISR"])  for d in datasets], dtype=float)
mistral_pr   = np.array([correct_percentage(DATASETS[d]["Mistral\nPR"])   for d in datasets], dtype=float)
mistral_mpr  = np.array([correct_percentage(DATASETS[d]["Mistral\nMPR"])  for d in datasets], dtype=float)
llama_isr    = np.array([correct_percentage(DATASETS[d]["LLaMA\nISR"])    for d in datasets], dtype=float)
llama_pr     = np.array([correct_percentage(DATASETS[d]["LLaMA\nPR"])     for d in datasets], dtype=float)
llama_mpr    = np.array([correct_percentage(DATASETS[d]["LLaMA\nMPR"])    for d in datasets], dtype=float)
qwen_isr     = np.array([correct_percentage(DATASETS[d]["Qwen\nISR"])     for d in datasets], dtype=float)
qwen_pr      = np.array([correct_percentage(DATASETS[d]["Qwen\nPR"])      for d in datasets], dtype=float)
qwen_mpr     = np.array([correct_percentage(DATASETS[d]["Qwen\nMPR"])     for d in datasets], dtype=float)

x = np.arange(len(datasets), dtype=float)
width = 0.085

fig, ax = plt.subplots(figsize=(7.4, 3.6))

ax.bar(
    x - 3*width, gpt_isr, width,
    label="GPT ISR",
    color="0.15",
    edgecolor="black",
    linewidth=0.8
)

ax.bar(
    x - 2*width, mistral_isr, width,
    label="Mistral ISR",
    color="0.35",
    edgecolor="black",
    linewidth=0.8
)

ax.bar(
    x - 1*width, mistral_pr, width,
    label="Mistral PR",
    color="0.55",
    edgecolor="black",
    linewidth=0.8,
    hatch="///"
)

ax.bar(
    x + 0*width, mistral_mpr, width,
    label="Mistral MPR",
    color="0.75",
    edgecolor="black",
    linewidth=0.8,
    hatch="xx"
)

ax.bar(
    x + 1*width, llama_isr, width,
    label="LLaMA ISR",
    color="0.45",
    edgecolor="black",
    linewidth=0.8
)

ax.bar(
    x + 2*width, llama_pr, width,
    label="LLaMA PR",
    color="0.70",
    edgecolor="black",
    linewidth=0.8,
    hatch="..."
)

ax.bar(
    x + 3*width, llama_mpr, width,
    label="LLaMA MPR",
    color="0.90",
    edgecolor="black",
    linewidth=0.8,
    hatch="\\\\"
)
ax.bar(
    x + 4*width, qwen_isr, width,
    label="Qwen ISR",
    color="0.30",
    edgecolor="black",
    linewidth=0.8,
    hatch="--"
)

ax.bar(
    x + 5*width, qwen_pr, width,
    label="Qwen PR",
    color="0.60",
    edgecolor="black",
    linewidth=0.8,
    hatch="||"
)

ax.bar(
    x + 6*width, qwen_mpr, width,
    label="Qwen MPR",
    color="0.85",
    edgecolor="black",
    linewidth=0.8,
    hatch=".."
)

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("Correct (%)")
ax.set_xlabel("Dataset")
ax.set_ylim(0, 100)

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="0.82")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))

plt.tight_layout()
plt.savefig("strategy_accuracy_updated.pdf", bbox_inches="tight")
plt.savefig("strategy_accuracy_updated.png", bbox_inches="tight")
plt.show()

