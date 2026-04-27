import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np

TRAINING_DATA_DIRECTORY = "training-data"
HEATMAP_DIRECTORY = "heatmap"
BARCHART_DIRECTORY = "barchart"
LINECHART_DIRECTORY = "linechart"
GRADESPAN_DIRECTORY = "gradespan"
MUNICIPALITY_DIRECTORY = "municipality"
SAFETY_DIRECTORY = "safety-comparison"

feature_cols = ["pct_ccr", "ratio", "Violent Crime Total", "Larceny-theft",
                "Motor vehicle theft", "pct_l3", "pct_l4", "pct_l5"]
crime_cols = ["Violent Crime Total", "Larceny-theft", "Motor vehicle theft"]
edu_cols = ["pct_ccr", "pct_l3", "pct_l4", "pct_l5", "ratio"]
required_cols = {"subgroup", "Area Type", "grade", "type", "name"} | set(feature_cols)
gradespan_required = required_cols | {"grade_span", "subject"}

bar_pairs = [
    ("ratio", "pct_ccr"),
    ("ratio", "pct_l3"),
    ("ratio", "Motor vehicle theft"),
    ("pct_l3", "Larceny-theft"),
    ("pct_l3", "Violent Crime Total"),
]


def extract_year(filename):
    match = re.search(r"(20\d{2})", filename)
    return int(match.group(1)) if match else None


def get_filtered_data(file, extra_cols=None):
    df = pd.read_csv(f"{TRAINING_DATA_DIRECTORY}/{file}")
    df.columns = df.columns.str.strip()

    keep_cols = ["name", "subgroup"] + feature_cols
    check_cols = required_cols.copy()

    if extra_cols:
        keep_cols += extra_cols
        check_cols |= set(extra_cols)

    missing = check_cols - set(df.columns)
    if missing:
        print(f"Skipping {file} — missing columns: {missing}")
        return None

    names = df[
        (df["subgroup"] == "ALL") &
        (df["Area Type"] == "Municipality") &
        (df["grade"] == "ALL") &
        (df["type"] == "ALL")
    ][keep_cols].copy()

    if names.empty:
        return None

    for c in feature_cols:
        names[c] = names[c].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        names[c] = pd.to_numeric(names[c], errors="coerce")

    names = names.dropna(subset=feature_cols)
    return names if not names.empty else None


def heatmap():
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    os.makedirs(HEATMAP_DIRECTORY, exist_ok=True)
    relevant_files = data_files[2:]

    for file in relevant_files:
        print(f"\n--- Heatmap: {file} ---")
        try:
            names = get_filtered_data(file)
            if names is None:
                continue
            corr_matrix = names[feature_cols].corr(method="pearson")
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"Pearson Correlation: College Readiness vs Crime ({file})")
            plt.tight_layout()
            plt.savefig(f"{HEATMAP_DIRECTORY}/{file}-heatmap.png")
            plt.close()
            print(f"Saved heatmap for {file}")
        except Exception as e:
            print(f"Failed on {file}: {e}")


def barchart():
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    os.makedirs(BARCHART_DIRECTORY, exist_ok=True)
    relevant_files = data_files[2:]

    for file in relevant_files:
        print(f"\n--- Bar Chart: {file} ---")
        try:
            names = get_filtered_data(file)
            if names is None:
                continue
            corr_matrix = names[feature_cols].corr(method="pearson")
            labels = [f"{a} vs {b}" for a, b in bar_pairs]
            values = [corr_matrix.loc[a, b] for a, b in bar_pairs]
            colors = ["#d73027" if v < 0 else "#4575b4" for v in values]
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values, color=colors)
            plt.axhline(0, color="black", linewidth=0.8)
            plt.ylabel("Pearson Correlation")
            plt.title(f"Key Correlations: Education vs Crime ({file})")
            plt.xticks(rotation=25, ha="right")
            plt.ylim(-1, 1)
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                         f"{val:.2f}", ha="center", fontsize=10)
            plt.tight_layout()
            plt.savefig(f"{BARCHART_DIRECTORY}/{file}-barchart.png")
            plt.close()
            print(f"Saved bar chart for {file}")
        except Exception as e:
            print(f"Failed on {file}: {e}")


def linechart():
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    os.makedirs(LINECHART_DIRECTORY, exist_ok=True)
    relevant_files = data_files[2:]

    yearly_corrs = {}
    for file in relevant_files:
        try:
            year = extract_year(file)
            if year is None:
                continue
            names = get_filtered_data(file)
            if names is None:
                continue
            corr_matrix = names[feature_cols].corr(method="pearson")
            yearly_corrs[year] = {f"{a} vs {b}": corr_matrix.loc[a, b] for a, b in bar_pairs}
        except Exception as e:
            print(f"Failed on {file}: {e}")

    if not yearly_corrs:
        print("No data for line chart")
        return

    corr_df = pd.DataFrame(yearly_corrs).T.sort_index()
    plt.figure(figsize=(12, 6))
    for col in corr_df.columns:
        plt.plot(corr_df.index, corr_df[col], marker="o", label=col)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Year")
    plt.ylabel("Pearson Correlation")
    plt.title("Correlation Trends Over Time: Education vs Crime")
    plt.xticks(corr_df.index, rotation=45)
    plt.ylim(-1, 1)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{LINECHART_DIRECTORY}/correlation-trends.png")
    plt.close()
    print("Saved line chart")


def load_all_data(area_type_filter=None):
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    relevant_files = data_files[2:]
    frames = []

    for file in relevant_files:
        try:
            df = pd.read_csv(f"{TRAINING_DATA_DIRECTORY}/{file}")
            df.columns = df.columns.str.strip()
            missing = gradespan_required - set(df.columns)
            if missing:
                continue

            conditions = (
                (df["Area Type"] == "Municipality") &
                (df["type"] == "ALL")
            )
            if area_type_filter:
                conditions = conditions & (df["Area Type"] == area_type_filter)
            else:
                conditions = conditions & (df["Area Type"] == "Municipality")

            filtered = df[conditions].copy()
            for c in feature_cols:
                filtered[c] = filtered[c].astype(str).str.replace(r"[^0-9.]", "", regex=True)
                filtered[c] = pd.to_numeric(filtered[c], errors="coerce")
            filtered = filtered.dropna(subset=feature_cols)
            if not filtered.empty:
                frames.append(filtered)
        except Exception as e:
            print(f"Failed loading {file}: {e}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all_data_with_area_type():
    """Load all data keeping the Area Type column for cross-area comparison."""
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    relevant_files = data_files[2:]
    frames = []

    for file in relevant_files:
        try:
            df = pd.read_csv(f"{TRAINING_DATA_DIRECTORY}/{file}")
            df.columns = df.columns.str.strip()
            missing = gradespan_required - set(df.columns)
            if missing:
                continue

            filtered = df[
                (df["subgroup"] == "ALL") &
                (df["grade"] == "ALL") &
                (df["type"] == "ALL")
            ].copy()

            for c in feature_cols:
                filtered[c] = filtered[c].astype(str).str.replace(r"[^0-9.]", "", regex=True)
                filtered[c] = pd.to_numeric(filtered[c], errors="coerce")
            filtered = filtered.dropna(subset=feature_cols)
            if not filtered.empty:
                frames.append(filtered)
        except Exception as e:
            print(f"Failed loading {file}: {e}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def get_all_area_types():
    data_files = os.listdir(TRAINING_DATA_DIRECTORY)
    relevant_files = data_files[2:]
    area_types = set()
    for file in relevant_files:
        try:
            df = pd.read_csv(f"{TRAINING_DATA_DIRECTORY}/{file}")
            df.columns = df.columns.str.strip()
            if "Area Type" in df.columns:
                area_types.update(df["Area Type"].dropna().unique())
        except Exception:
            continue
    return list(area_types)


def gradespan_analysis():
    os.makedirs(GRADESPAN_DIRECTORY, exist_ok=True)
    all_data = load_all_data()

    if all_data is None or all_data.empty:
        print("No data available for grade span analysis")
        return

    grade_spans = all_data["grade_span"].dropna().unique()
    subjects = all_data["subject"].dropna().unique()

    print(f"Grade spans found: {list(grade_spans)}")
    print(f"Subjects found: {list(subjects)}")

    for subj in subjects:
        subj_data = all_data[all_data["subject"] == subj]
        if len(subj_data) < 3:
            continue
        rows = []
        for gs in grade_spans:
            gs_data = subj_data[subj_data["grade_span"] == gs]
            if len(gs_data) < 3:
                continue
            corr = gs_data[feature_cols].corr(method="pearson")
            for crime in crime_cols:
                for edu in edu_cols:
                    rows.append({
                        "grade_span": str(gs), "crime": crime,
                        "edu_metric": edu, "correlation": corr.loc[edu, crime]
                    })
        if not rows:
            continue

        pair_df = pd.DataFrame(rows)
        fig, axes = plt.subplots(len(edu_cols), 1, figsize=(14, 4 * len(edu_cols)), sharex=True)
        if len(edu_cols) == 1:
            axes = [axes]
        for ax, edu in zip(axes, edu_cols):
            subset = pair_df[pair_df["edu_metric"] == edu]
            sns.barplot(data=subset, x="grade_span", y="correlation", hue="crime", ax=ax)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"{edu} vs Crime Types")
            ax.set_ylabel("Pearson Correlation")
            ax.set_ylim(-1, 1)
            ax.legend(fontsize=8)
        fig.suptitle(f"Crime Impact by Grade Span — Subject: {subj}", fontsize=14, y=1.01)
        plt.xlabel("Grade Span")
        plt.tight_layout()
        safe_subj = str(subj).replace("/", "-").replace(" ", "_")
        plt.savefig(f"{GRADESPAN_DIRECTORY}/subject_{safe_subj}-by_gradespan.png", bbox_inches="tight")
        plt.close()
        print(f"Saved grade span chart for subject: {subj}")

    for gs in grade_spans:
        gs_data = all_data[all_data["grade_span"] == gs]
        if len(gs_data) < 3:
            continue
        corr = gs_data[feature_cols].corr(method="pearson")
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Aggregated Correlation — Grade Span: {gs}")
        plt.tight_layout()
        safe_gs = str(gs).replace("/", "-")
        plt.savefig(f"{GRADESPAN_DIRECTORY}/aggregated-gs_{safe_gs}-heatmap.png")
        plt.close()

    summary_rows = []
    for gs in grade_spans:
        for subj in subjects:
            subset = all_data[(all_data["grade_span"] == gs) & (all_data["subject"] == subj)]
            if len(subset) < 3:
                continue
            corr = subset[feature_cols].corr(method="pearson")
            for crime in crime_cols:
                avg_corr = corr.loc[edu_cols, crime].abs().mean()
                summary_rows.append({
                    "grade_span": str(gs), "subject": str(subj),
                    "crime": crime, "avg_abs_correlation": avg_corr
                })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        for subj in subjects:
            subj_summary = summary_df[summary_df["subject"] == subj]
            if subj_summary.empty:
                continue
            pivot = subj_summary.pivot_table(index="grade_span", columns="crime", values="avg_abs_correlation")
            plt.figure(figsize=(10, 6))
            pivot.plot(kind="bar", figsize=(10, 6))
            plt.title(f"Avg |Correlation| with Education — Subject: {subj}")
            plt.ylabel("Mean Absolute Pearson Correlation")
            plt.xlabel("Grade Span")
            plt.ylim(0, 1)
            plt.legend(title="Crime Type")
            plt.tight_layout()
            safe_subj = str(subj).replace("/", "-").replace(" ", "_")
            plt.savefig(f"{GRADESPAN_DIRECTORY}/summary_{safe_subj}-crime_impact.png")
            plt.close()
        print("Saved summary charts")
    print("Grade span analysis complete")


def municipality_analysis():
    """Run the full gradespan analysis separately for each Area Type."""
    os.makedirs(MUNICIPALITY_DIRECTORY, exist_ok=True)
    area_types = get_all_area_types()
    print(f"Area types found: {area_types}")

    for area_type in area_types:
        safe_area = str(area_type).replace("/", "-").replace(" ", "_")
        area_dir = f"{MUNICIPALITY_DIRECTORY}/{safe_area}"
        os.makedirs(area_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Area Type: {area_type}")
        print(f"{'='*60}")

        all_data = load_all_data(area_type_filter=area_type)

        if all_data is None or all_data.empty:
            print(f"No data for Area Type: {area_type}")
            continue

        if "grade_span" not in all_data.columns or "subject" not in all_data.columns:
            print(f"Missing grade_span/subject for Area Type: {area_type}")
            continue

        grade_spans = all_data["grade_span"].dropna().unique()
        subjects = all_data["subject"].dropna().unique()

        print(f"  Grade spans: {list(grade_spans)}")
        print(f"  Subjects: {list(subjects)}")

        # --- Charts 1-4 (same as before) ---
        for subj in subjects:
            subj_data = all_data[all_data["subject"] == subj]
            if len(subj_data) < 3:
                continue
            rows = []
            for gs in grade_spans:
                gs_data = subj_data[subj_data["grade_span"] == gs]
                if len(gs_data) < 3:
                    continue
                corr = gs_data[feature_cols].corr(method="pearson")
                for crime in crime_cols:
                    for edu in edu_cols:
                        rows.append({"grade_span": str(gs), "crime": crime,
                                     "edu_metric": edu, "correlation": corr.loc[edu, crime]})
            if not rows:
                continue
            pair_df = pd.DataFrame(rows)
            fig, axes = plt.subplots(len(edu_cols), 1, figsize=(14, 4 * len(edu_cols)), sharex=True)
            if len(edu_cols) == 1:
                axes = [axes]
            for ax, edu in zip(axes, edu_cols):
                subset = pair_df[pair_df["edu_metric"] == edu]
                sns.barplot(data=subset, x="grade_span", y="correlation", hue="crime", ax=ax)
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title(f"{edu} vs Crime Types")
                ax.set_ylabel("Pearson Correlation")
                ax.set_ylim(-1, 1)
                ax.legend(fontsize=8)
            fig.suptitle(f"Crime Impact — {area_type} — Subject: {subj}", fontsize=14, y=1.01)
            plt.xlabel("Grade Span")
            plt.tight_layout()
            safe_subj = str(subj).replace("/", "-").replace(" ", "_")
            plt.savefig(f"{area_dir}/subject_{safe_subj}-by_gradespan.png", bbox_inches="tight")
            plt.close()

        for gs in grade_spans:
            gs_data = all_data[all_data["grade_span"] == gs]
            if len(gs_data) < 3:
                continue
            corr = gs_data[feature_cols].corr(method="pearson")
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"Correlation — {area_type} — Grade Span: {gs}")
            plt.tight_layout()
            safe_gs = str(gs).replace("/", "-")
            plt.savefig(f"{area_dir}/gs_{safe_gs}-heatmap.png")
            plt.close()

        summary_rows = []
        for gs in grade_spans:
            for subj in subjects:
                subset = all_data[(all_data["grade_span"] == gs) & (all_data["subject"] == subj)]
                if len(subset) < 3:
                    continue
                corr = subset[feature_cols].corr(method="pearson")
                for crime in crime_cols:
                    avg_corr = corr.loc[edu_cols, crime].abs().mean()
                    summary_rows.append({"grade_span": str(gs), "subject": str(subj),
                                         "crime": crime, "avg_abs_correlation": avg_corr})
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            for subj in subjects:
                subj_summary = summary_df[summary_df["subject"] == subj]
                if subj_summary.empty:
                    continue
                pivot = subj_summary.pivot_table(index="grade_span", columns="crime", values="avg_abs_correlation")
                plt.figure(figsize=(10, 6))
                pivot.plot(kind="bar", figsize=(10, 6))
                plt.title(f"Avg |Correlation| — {area_type} — Subject: {subj}")
                plt.ylabel("Mean Absolute Pearson Correlation")
                plt.xlabel("Grade Span")
                plt.ylim(0, 1)
                plt.legend(title="Crime Type")
                plt.tight_layout()
                safe_subj = str(subj).replace("/", "-").replace(" ", "_")
                plt.savefig(f"{area_dir}/summary_{safe_subj}-crime_impact.png")
                plt.close()

        compare_rows = []
        for gs in grade_spans:
            for subj in subjects:
                subset = all_data[(all_data["grade_span"] == gs) & (all_data["subject"] == subj)]
                if len(subset) < 3:
                    continue
                corr = subset[feature_cols].corr(method="pearson")
                for crime in crime_cols:
                    for edu in edu_cols:
                        compare_rows.append({"grade_span": str(gs), "subject": str(subj),
                                             "pair": f"{edu} vs {crime}", "correlation": corr.loc[edu, crime]})
        if compare_rows:
            compare_df = pd.DataFrame(compare_rows)
            for gs in grade_spans:
                gs_compare = compare_df[compare_df["grade_span"] == str(gs)]
                if gs_compare.empty:
                    continue
                plt.figure(figsize=(16, 8))
                sns.barplot(data=gs_compare, x="subject", y="correlation", hue="pair")
                plt.axhline(0, color="black", linewidth=0.8)
                plt.title(f"Subject Comparison — {area_type} — Grade Span: {gs}")
                plt.ylabel("Pearson Correlation")
                plt.xlabel("Subject")
                plt.ylim(-1, 1)
                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
                plt.tight_layout()
                safe_gs = str(gs).replace("/", "-")
                plt.savefig(f"{area_dir}/compare_gs_{safe_gs}-by_subject.png", bbox_inches="tight")
                plt.close()

        # --- Chart 5: Avg crime rates + correlation impact per grade span ---
        avg_crime_rows = []
        corr_impact_rows = []
        for gs in grade_spans:
            gs_data = all_data[all_data["grade_span"] == gs]
            if len(gs_data) < 3:
                continue
            for crime in crime_cols:
                avg_crime_rows.append({"grade_span": str(gs), "crime": crime,
                                       "avg_crime_rate": gs_data[crime].mean()})
            corr = gs_data[feature_cols].corr(method="pearson")
            for crime in crime_cols:
                avg_abs = corr.loc[edu_cols, crime].abs().mean()
                corr_impact_rows.append({"grade_span": str(gs), "crime": crime,
                                         "avg_abs_correlation": avg_abs})
        if avg_crime_rows and corr_impact_rows:
            avg_df = pd.DataFrame(avg_crime_rows)
            corr_imp_df = pd.DataFrame(corr_impact_rows)
            for crime in crime_cols:
                crime_avg = avg_df[avg_df["crime"] == crime].set_index("grade_span")
                crime_corr = corr_imp_df[corr_imp_df["crime"] == crime].set_index("grade_span")
                merged = crime_avg.join(crime_corr[["avg_abs_correlation"]], how="inner")
                if merged.empty:
                    continue
                fig, ax1 = plt.subplots(figsize=(12, 6))
                x = np.arange(len(merged))
                ax1.bar(x, merged["avg_crime_rate"], 0.4, color="#4575b4", alpha=0.7, label="Avg Crime Rate")
                ax1.set_ylabel("Average Crime Rate", color="#4575b4")
                ax1.tick_params(axis="y", labelcolor="#4575b4")
                ax1.set_xticks(x)
                ax1.set_xticklabels(merged.index, rotation=30, ha="right")
                ax1.set_xlabel("Grade Span")
                ax2 = ax1.twinx()
                ax2.plot(x, merged["avg_abs_correlation"], color="#d73027", marker="o", linewidth=2,
                         label="Avg |Correlation| with Education")
                ax2.set_ylabel("Mean |Pearson Correlation|", color="#d73027")
                ax2.tick_params(axis="y", labelcolor="#d73027")
                ax2.set_ylim(0, 1)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
                safe_crime = crime.replace(" ", "_")
                plt.title(f"{crime} — Avg Rate & Education Impact — {area_type}")
                plt.tight_layout()
                plt.savefig(f"{area_dir}/crime_rate_vs_impact_{safe_crime}.png")
                plt.close()

        # --- Chart 6: Top 20 municipalities with grade span overlay ---
        if "name" in all_data.columns:
            municipalities = all_data["name"].dropna().unique()
            muni_rows = []
            for muni in municipalities:
                muni_data = all_data[all_data["name"] == muni]
                if len(muni_data) < 3:
                    continue
                for crime in crime_cols:
                    muni_rows.append({"municipality": str(muni), "crime": crime,
                                      "avg_crime_rate": muni_data[crime].mean()})
            if muni_rows:
                muni_df = pd.DataFrame(muni_rows)
                for crime in crime_cols:
                    subset = muni_df[muni_df["crime"] == crime].sort_values(
                        "avg_crime_rate", ascending=False).head(20)
                    if subset.empty:
                        continue
                    muni_corr_rows = []
                    for muni in subset["municipality"].unique():
                        muni_data = all_data[all_data["name"] == muni]
                        for gs in grade_spans:
                            gs_data = muni_data[muni_data["grade_span"] == gs]
                            if len(gs_data) < 3:
                                continue
                            corr = gs_data[feature_cols].corr(method="pearson")
                            avg_abs = corr.loc[edu_cols, crime].abs().mean()
                            muni_corr_rows.append({"municipality": str(muni), "grade_span": str(gs),
                                                    "avg_abs_correlation": avg_abs})
                    fig, ax1 = plt.subplots(figsize=(16, 8))
                    x = np.arange(len(subset))
                    ax1.bar(x, subset["avg_crime_rate"].values, color="#4575b4", alpha=0.7)
                    ax1.set_ylabel("Avg Crime Rate", color="#4575b4")
                    ax1.tick_params(axis="y", labelcolor="#4575b4")
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(subset["municipality"].values, rotation=45, ha="right", fontsize=8)
                    ax1.set_xlabel("Municipality")
                    if muni_corr_rows:
                        muni_corr_df = pd.DataFrame(muni_corr_rows)
                        ax2 = ax1.twinx()
                        for gs in sorted(muni_corr_df["grade_span"].unique()):
                            gs_vals = muni_corr_df[muni_corr_df["grade_span"] == gs]
                            ordered = []
                            for muni in subset["municipality"].values:
                                match = gs_vals[gs_vals["municipality"] == muni]
                                ordered.append(match["avg_abs_correlation"].values[0]
                                               if not match.empty else np.nan)
                            ax2.plot(x, ordered, marker="o", label=f"GS: {gs}", linewidth=1.5)
                        ax2.set_ylabel("Avg |Correlation| with Education", color="#d73027")
                        ax2.tick_params(axis="y", labelcolor="#d73027")
                        ax2.set_ylim(0, 1)
                        ax2.legend(loc="upper right", fontsize=7, title="Grade Span")
                    safe_crime = crime.replace(" ", "_")
                    plt.title(f"Top 20 Municipalities — {crime} Rate & Education Impact\n{area_type}")
                    plt.tight_layout()
                    plt.savefig(f"{area_dir}/top_munis_{safe_crime}-rate_vs_gradespan.png", bbox_inches="tight")
                    plt.close()

        print(f"Saved all charts for Area Type: {area_type}")
    print("\nMunicipality analysis complete")


def safety_comparison():
    """Compare safest vs most dangerous counties and cities, and contrast their education correlations."""
    os.makedirs(SAFETY_DIRECTORY, exist_ok=True)

    all_data = load_all_data_with_area_type()
    if all_data is None or all_data.empty:
        print("No data for safety comparison")
        return

    if "Area Type" not in all_data.columns or "name" not in all_data.columns:
        print("Missing Area Type or name column")
        return

    area_types = all_data["Area Type"].dropna().unique()
    grade_spans = all_data["grade_span"].dropna().unique() if "grade_span" in all_data.columns else []

    TOP_N = 10

    # --- Per area type: safest vs most dangerous ---
    for area_type in area_types:
        at_data = all_data[all_data["Area Type"] == area_type]
        if at_data.empty:
            continue

        safe_area = str(area_type).replace("/", "-").replace(" ", "_")
        at_dir = f"{SAFETY_DIRECTORY}/{safe_area}"
        os.makedirs(at_dir, exist_ok=True)

        # Compute total crime score per name
        name_crime = at_data.groupby("name")[crime_cols].mean().reset_index()
        name_crime["total_crime"] = name_crime[crime_cols].sum(axis=1)
        name_crime = name_crime.sort_values("total_crime")

        safest = name_crime.head(TOP_N).copy()
        most_dangerous = name_crime.tail(TOP_N).copy()

        # --- Chart 1: Side-by-side bar of avg crime rates ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

        safest_melted = safest.melt(id_vars="name", value_vars=crime_cols,
                                     var_name="crime", value_name="avg_rate")
        sns.barplot(data=safest_melted, y="name", x="avg_rate", hue="crime", ax=ax1)
        ax1.set_title(f"Top {TOP_N} Safest")
        ax1.set_xlabel("Avg Crime Rate")
        ax1.set_ylabel("")

        dangerous_melted = most_dangerous.melt(id_vars="name", value_vars=crime_cols,
                                                var_name="crime", value_name="avg_rate")
        sns.barplot(data=dangerous_melted, y="name", x="avg_rate", hue="crime", ax=ax2)
        ax2.set_title(f"Top {TOP_N} Most Dangerous")
        ax2.set_xlabel("Avg Crime Rate")
        ax2.set_ylabel("")

        fig.suptitle(f"Safest vs Most Dangerous — {area_type}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{at_dir}/safest_vs_dangerous_crime_rates.png", bbox_inches="tight")
        plt.close()

        # --- Chart 2: Education metrics comparison (safest vs dangerous) ---
        safest_names = set(safest["name"])
        dangerous_names = set(most_dangerous["name"])

        safest_edu = at_data[at_data["name"].isin(safest_names)][edu_cols].mean()
        dangerous_edu = at_data[at_data["name"].isin(dangerous_names)][edu_cols].mean()

        edu_compare = pd.DataFrame({
            "Safest": safest_edu,
            "Most Dangerous": dangerous_edu
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        edu_compare.plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
        ax.set_title(f"Avg Education Metrics — Safest vs Dangerous — {area_type}")
        ax.set_ylabel("Average Value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"{at_dir}/safest_vs_dangerous_edu_metrics.png")
        plt.close()

        # --- Chart 3: Correlation with education per grade span — safest vs dangerous ---
        if len(grade_spans) > 0:
            for group_label, group_names in [("Safest", safest_names), ("Most Dangerous", dangerous_names)]:
                group_data = at_data[at_data["name"].isin(group_names)]

                rows = []
                for gs in grade_spans:
                    gs_data = group_data[group_data["grade_span"] == gs]
                    if len(gs_data) < 3:
                        continue
                    corr = gs_data[feature_cols].corr(method="pearson")
                    for crime in crime_cols:
                        avg_abs = corr.loc[edu_cols, crime].abs().mean()
                        rows.append({"grade_span": str(gs), "crime": crime,
                                     "avg_abs_correlation": avg_abs})

                if not rows:
                    continue

                gs_df = pd.DataFrame(rows)
                pivot = gs_df.pivot_table(index="grade_span", columns="crime", values="avg_abs_correlation")

                plt.figure(figsize=(10, 6))
                pivot.plot(kind="bar", figsize=(10, 6))
                plt.title(f"Crime-Education |Correlation| by Grade Span\n{group_label} {area_type}s")
                plt.ylabel("Mean |Pearson Correlation|")
                plt.xlabel("Grade Span")
                plt.ylim(0, 1)
                plt.legend(title="Crime Type")
                plt.tight_layout()
                safe_label = group_label.replace(" ", "_").lower()
                plt.savefig(f"{at_dir}/{safe_label}_corr_by_gradespan.png")
                plt.close()

            # --- Chart 4: Direct contrast — same grade span, safest vs dangerous side by side ---
            contrast_rows = []
            for gs in grade_spans:
                for group_label, group_names in [("Safest", safest_names), ("Most Dangerous", dangerous_names)]:
                    gs_data = at_data[(at_data["name"].isin(group_names)) & (at_data["grade_span"] == gs)]
                    if len(gs_data) < 3:
                        continue
                    corr = gs_data[feature_cols].corr(method="pearson")
                    for crime in crime_cols:
                        for edu in edu_cols:
                            contrast_rows.append({
                                "grade_span": str(gs), "group": group_label,
                                "pair": f"{edu} vs {crime}", "correlation": corr.loc[edu, crime]
                            })

            if contrast_rows:
                contrast_df = pd.DataFrame(contrast_rows)

                for gs in grade_spans:
                    gs_contrast = contrast_df[contrast_df["grade_span"] == str(gs)]
                    if gs_contrast.empty:
                        continue

                    plt.figure(figsize=(18, 8))
                    sns.barplot(data=gs_contrast, x="pair", y="correlation", hue="group",
                                palette={"Safest": "#2ca02c", "Most Dangerous": "#d62728"})
                    plt.axhline(0, color="black", linewidth=0.8)
                    plt.title(f"Safest vs Dangerous — Grade Span: {gs} — {area_type}")
                    plt.ylabel("Pearson Correlation")
                    plt.xlabel("")
                    plt.xticks(rotation=35, ha="right", fontsize=8)
                    plt.ylim(-1, 1)
                    plt.legend(title="Group")
                    plt.tight_layout()
                    safe_gs = str(gs).replace("/", "-")
                    plt.savefig(f"{at_dir}/contrast_gs_{safe_gs}.png", bbox_inches="tight")
                    plt.close()

        # --- Chart 5: Heatmap for safest vs dangerous ---
        for group_label, group_names in [("Safest", safest_names), ("Most Dangerous", dangerous_names)]:
            group_data = at_data[at_data["name"].isin(group_names)]
            if len(group_data) < 3:
                continue
            corr = group_data[feature_cols].corr(method="pearson")
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"Correlation — {group_label} {area_type}s")
            plt.tight_layout()
            safe_label = group_label.replace(" ", "_").lower()
            plt.savefig(f"{at_dir}/{safe_label}_heatmap.png")
            plt.close()

        print(f"Saved safety comparison for {area_type}")

    # --- Cross area-type comparison: counties vs cities ---
    county_keywords = ["County", "county"]
    city_keywords = ["Municipality", "City", "city", "municipality"]

    county_types = [at for at in area_types if any(k in str(at) for k in county_keywords)]
    city_types = [at for at in area_types if any(k in str(at) for k in city_keywords)]

    if county_types and city_types:
        county_data = all_data[all_data["Area Type"].isin(county_types)]
        city_data = all_data[all_data["Area Type"].isin(city_types)]

        if not county_data.empty and not city_data.empty:
            county_avg = county_data.groupby("name")[crime_cols].mean()
            county_avg["total_crime"] = county_avg.sum(axis=1)
            city_avg = city_data.groupby("name")[crime_cols].mean()
            city_avg["total_crime"] = city_avg.sum(axis=1)

            # Overall avg comparison
            compare = pd.DataFrame({
                "Counties (avg)": county_data[crime_cols + edu_cols].mean(),
                "Cities (avg)": city_data[crime_cols + edu_cols].mean()
            })

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            compare.loc[crime_cols].plot(kind="bar", ax=ax1, color=["#ff7f0e", "#1f77b4"])
            ax1.set_title("Avg Crime Rates")
            ax1.set_ylabel("Average")
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right")

            compare.loc[edu_cols].plot(kind="bar", ax=ax2, color=["#ff7f0e", "#1f77b4"])
            ax2.set_title("Avg Education Metrics")
            ax2.set_ylabel("Average")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")

            fig.suptitle("Counties vs Cities — Crime & Education Overview", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{SAFETY_DIRECTORY}/counties_vs_cities_overview.png")
            plt.close()

            # Correlation comparison by grade span
            if len(grade_spans) > 0:
                cross_rows = []
                for label, data in [("Counties", county_data), ("Cities", city_data)]:
                    for gs in grade_spans:
                        gs_data = data[data["grade_span"] == gs] if "grade_span" in data.columns else pd.DataFrame()
                        if len(gs_data) < 3:
                            continue
                        corr = gs_data[feature_cols].corr(method="pearson")
                        for crime in crime_cols:
                            avg_abs = corr.loc[edu_cols, crime].abs().mean()
                            cross_rows.append({"area": label, "grade_span": str(gs),
                                               "crime": crime, "avg_abs_correlation": avg_abs})

                if cross_rows:
                    cross_df = pd.DataFrame(cross_rows)

                    for crime in crime_cols:
                        subset = cross_df[cross_df["crime"] == crime]
                        if subset.empty:
                            continue

                        pivot = subset.pivot_table(index="grade_span", columns="area",
                                                   values="avg_abs_correlation")

                        plt.figure(figsize=(10, 6))
                        pivot.plot(kind="bar", color=["#1f77b4", "#ff7f0e"])
                        plt.title(f"Counties vs Cities — |Correlation| {crime} & Education")
                        plt.ylabel("Mean |Pearson Correlation|")
                        plt.xlabel("Grade Span")
                        plt.ylim(0, 1)
                        plt.legend(title="Area")
                        plt.tight_layout()
                        safe_crime = crime.replace(" ", "_")
                        plt.savefig(f"{SAFETY_DIRECTORY}/counties_vs_cities_{safe_crime}_by_gs.png")
                        plt.close()

                print("Saved county vs city comparison charts")

    print("\nSafety comparison complete")


if __name__ == "__main__":

    safety_comparison()
    