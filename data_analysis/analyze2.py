import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
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
    return max(tied)

def distribution_for_setting(setting_matrix):

    majority = [majority_label(triple) for triple in setting_matrix]
    cnt = Counter(majority)
    total = len(majority)
    return np.array([
        cnt.get(1, 0) / total * 100.0,  # CC
        cnt.get(2, 0) / total * 100.0,  # PC
        cnt.get(3, 0) / total * 100.0,  # IC
        cnt.get(4, 0) / total * 100.0,  # NA
    ])

def build_strategy_data(strategy_suffix, model_order):

    result = {}
    for dataset_name, entries in DATASETS.items():
        rows = []
        for model in model_order:
            key = f"{model}\n{strategy_suffix}"
            if key in entries:
                rows.append(distribution_for_setting(entries[key]))
        result[dataset_name] = np.array(rows)
    return result

isr_data = build_strategy_data("ISR", ["GPT", "Mistral", "LLaMA", "Qwen"])
pr_data  = build_strategy_data("PR",  ["Mistral", "LLaMA", "Qwen"])
mpr_data = build_strategy_data("MPR", ["Mistral", "LLaMA", "Qwen"])

BAR_WIDTH = 0.42
SPACING = 0.78

C1 = "0.20"  # CC
C2 = "0.45"  # PC
C3 = "0.75"  # IC
C4 = "0.92"  # NA

EDGE = "black"
LW = 0.7

def plot_strategy_panels(data_dict, labels, strategy_name, filename):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), sharey=True)

    for ax, dataset in zip(axes, ["LDBC", "Knows", "DBLP"]):
        vals = data_dict[dataset]
        x = np.arange(len(labels)) * SPACING

        cc, pc, ic, na = vals[:,0], vals[:,1], vals[:,2], vals[:,3]

        ax.bar(x, cc, BAR_WIDTH, color=C1, edgecolor=EDGE, linewidth=LW)
        ax.bar(x, pc, BAR_WIDTH, bottom=cc, color=C2, edgecolor=EDGE, linewidth=LW, hatch="///")
        ax.bar(x, ic, BAR_WIDTH, bottom=cc+pc, color=C3, edgecolor=EDGE, linewidth=LW, hatch="...")
        ax.bar(x, na, BAR_WIDTH, bottom=cc+pc+ic, color=C4, edgecolor=EDGE, linewidth=LW)

        ax.set_title(dataset, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)

        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.4, color="0.85")

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Percentage (%)")

    fig.legend(
        ["CC", "PC", "IC", "NA"],
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"{filename}.png", bbox_inches="tight")
    plt.show()

# ============================================================
# Final plots
# ============================================================
plot_strategy_panels(isr_data, ["GPT", "Mistral", "LLaMA", "Qwen"], "ISR", "pg_isr_updated")
plot_strategy_panels(pr_data,  ["Mistral", "LLaMA", "Qwen"],        "PR",  "pg_pr_updated")
plot_strategy_panels(mpr_data, ["Mistral", "LLaMA", "Qwen"],        "MPR", "pg_mpr_updated")