from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.types import StructType, StructField
from pyspark.sql import functions as sf
from pyspark.sql.functions import col, avg, round
from functools import cmp_to_key


def ranking_based_on_variable(variable: str, top_k: int, descending=True):
    data = pd.read_csv('inference_results_2.csv')

    groups = [group for _, group in data.groupby(variable)]

    sub_group_ranking = dict()
    for group in groups:
        group_type = group[variable].iloc[0]
        sub_group_ranking[group_type] = group['residual'].mean()

    sorted_subgroup_ranking = dict(
        sorted(sub_group_ranking.items(), key=lambda x: x[1], reverse=descending)
    )

    top_k_ranking = dict(list(sorted_subgroup_ranking.items())[:top_k])
    print(top_k_ranking)

    labels = list(top_k_ranking.keys())
    values = list(top_k_ranking.values())
    colors = ['#d9534f' if v < 0 else '#5b9bd5' for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=0.5)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel(variable.capitalize(), fontsize=12)
    ax.set_ylabel('Average Residual', fontsize=12)
    ax.set_title(f'Average Residual by {variable.capitalize()} (Top {top_k})', fontsize=14)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'ranking_{variable}.png', dpi=150)
    plt.show()

    return top_k_ranking


if __name__ == "__main__":
    ranking_based_on_variable("name", 5, False)