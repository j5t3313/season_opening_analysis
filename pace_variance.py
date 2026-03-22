import os

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


CACHE_DIR = "cache"
LOW_LAP_THRESHOLD = 15

SESSIONS = [
    (2026, "Australia", "R"),
    (2026, "China", "R"),
]

PU_MAP = {
    "Mercedes": "Mercedes",
    "McLaren": "Mercedes",
    "Williams": "Mercedes",
    "Ferrari": "Ferrari",
    "Haas": "Ferrari",
    "Cadillac": "Ferrari",
    "Red Bull Racing": "Ford x RBPT",
    "RB": "Ford x RBPT",
    "Racing Bulls": "Ford x RBPT",
    "Aston Martin": "Honda",
    "Alpine": "Mercedes",
    "Audi": "Audi",
    "Kick Sauber": "Audi",
}

PU_COLORS = {
    "Mercedes": "#00d2be",
    "Ferrari": "#dc0000",
    "Ford x RBPT": "#1e41ff",
    "Honda": "#006f62",
    "Audi": "#555555",
}

WATERMARK = "@formulasteele"


def watermark(fig):
    fig.text(
        0.99, 0.01, WATERMARK,
        ha="right", va="bottom", fontsize=8,
        color="#b0b0b0", alpha=0.6, style="italic",
        transform=fig.transFigure,
    )


def resolve_pu(team_name):
    for key, pu in PU_MAP.items():
        if key.lower() in team_name.lower():
            return pu
    return "Unknown"


def load_race(year, gp):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, gp, "R")
    session.load(telemetry=False, laps=True, weather=False)
    return session


def get_clean_laps(session):
    laps = session.laps.copy()
    laps = laps[laps["IsAccurate"] == True].copy()
    laps = laps[~laps["PitInTime"].notna()].copy()
    laps = laps[~laps["PitOutTime"].notna()].copy()

    for col in ["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"]:
        laps[col] = laps[col].dt.total_seconds()

    laps = laps.dropna(subset=["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"])
    return laps


def compute_driver_cv(laps):
    rows = []
    for drv in laps["Driver"].unique():
        drv_laps = laps[laps["Driver"] == drv]
        if len(drv_laps) < 5:
            continue

        team = drv_laps["Team"].iloc[0]
        pu = resolve_pu(str(team))

        sector_cvs = {}
        for sector in ["Sector1Time", "Sector2Time", "Sector3Time"]:
            vals = drv_laps[sector].values
            mean = np.mean(vals)
            if mean > 0:
                sector_cvs[sector] = np.std(vals) / mean
            else:
                sector_cvs[sector] = np.nan

        lap_vals = drv_laps["LapTime"].values
        lap_cv = np.std(lap_vals) / np.mean(lap_vals) if np.mean(lap_vals) > 0 else np.nan

        rows.append({
            "driver": drv,
            "team": team,
            "pu": pu,
            "n_laps": len(drv_laps),
            "cv_s1": sector_cvs["Sector1Time"],
            "cv_s2": sector_cvs["Sector2Time"],
            "cv_s3": sector_cvs["Sector3Time"],
            "cv_lap": lap_cv,
            "cv_mean_sectors": np.nanmean(list(sector_cvs.values())),
            "low_laps": len(drv_laps) < LOW_LAP_THRESHOLD,
        })

    return pd.DataFrame(rows)


def plot_cv_by_driver(cv_df, gp):
    df = cv_df.sort_values("cv_lap").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 7), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    colors = []
    for _, row in df.iterrows():
        c = PU_COLORS.get(row["pu"], "#333333")
        colors.append(c)

    bars = ax.barh(
        range(len(df)),
        df["cv_lap"] * 100,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    for i, (_, row) in enumerate(df.iterrows()):
        if row["low_laps"]:
            bars[i].set_alpha(0.35)
            bars[i].set_hatch("//")

    labels = []
    for _, row in df.iterrows():
        label = f"{row['driver']} ({row['team']})"
        if row["low_laps"]:
            label += f"  [n={row['n_laps']}]"
        labels.append(label)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Lap time CV (%)", fontsize=11, color="#1a1a1a")
    ax.set_title(
        f"Lap Time Consistency (lower = more consistent)\n{gp} 2026 Race",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )
    ax.grid(True, axis="x", color="#e0e0e0", alpha=0.6, linewidth=0.5)
    ax.invert_yaxis()

    has_low = df["low_laps"].any()
    if has_low:
        ax.text(
            0.98, 0.02,
            f"Hatched bars: fewer than {LOW_LAP_THRESHOLD} clean laps (interpret with caution)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#888888", style="italic",
        )

    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.tick_params(colors="#4a4a4a", labelsize=9)

    watermark(fig)
    fig.tight_layout()
    fname = f"cv_by_driver_{gp.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_sector_cv_heatmap(cv_df, gp):
    df = cv_df.sort_values("cv_mean_sectors").reset_index(drop=True)
    matrix = df[["cv_s1", "cv_s2", "cv_s3"]].values * 100

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Sector 1", "Sector 2", "Sector 3"], fontsize=10)
    ax.set_yticks(range(len(df)))

    labels = []
    for _, row in df.iterrows():
        label = row["driver"]
        if row["low_laps"]:
            label += f" [n={row['n_laps']}]*"
        labels.append(label)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(df)):
        for j in range(3):
            val = matrix[i, j]
            color = "#1a1a1a" if val < np.nanmedian(matrix) else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(
        f"Sector Time CV (%) by Driver\n{gp} 2026 Race",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )

    has_low = df["low_laps"].any()
    if has_low:
        ax.text(
            0.5, -0.04,
            f"* fewer than {LOW_LAP_THRESHOLD} clean laps",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#888888", style="italic",
        )

    fig.colorbar(im, ax=ax, label="CV (%)", shrink=0.8)
    watermark(fig)
    fig.tight_layout()
    fname = f"sector_cv_heatmap_{gp.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_cv_vs_pu(combined_df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    medians = combined_df.groupby("pu")["cv_lap"].median() * 100
    pu_order = list(medians.sort_values().index)

    for _, row in combined_df.iterrows():
        pu = row["pu"]
        color = PU_COLORS.get(pu, "#333333")
        x = pu_order.index(pu) + np.random.uniform(-0.15, 0.15)
        alpha = 0.35 if row["low_laps"] else 0.7
        marker = "x" if row["low_laps"] else "o"
        ax.scatter(x, row["cv_lap"] * 100, color=color, s=60, alpha=alpha, marker=marker, zorder=3)

    for i, pu in enumerate(pu_order):
        color = PU_COLORS.get(pu, "#333333")
        ax.hlines(medians[pu], i - 0.3, i + 0.3, color=color, linewidth=2, zorder=4)

    ax.set_xticks(range(len(pu_order)))
    ax.set_xticklabels(pu_order, fontsize=9)
    ax.set_xlabel("PU Manufacturer", fontsize=11, color="#1a1a1a")
    ax.set_ylabel("Lap time CV (%)", fontsize=11, color="#1a1a1a")
    ax.set_title(
        "Pace Consistency by PU Manufacturer\nAustralia + China 2026 Race",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )
    ax.grid(True, axis="y", color="#e0e0e0", alpha=0.6, linewidth=0.5)

    ax.text(
        0.98, 0.02,
        f"x markers: fewer than {LOW_LAP_THRESHOLD} clean laps",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8, color="#888888", style="italic",
    )

    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.tick_params(colors="#4a4a4a", labelsize=9)

    watermark(fig)
    fig.tight_layout()
    fig.savefig("cv_by_pu_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    all_cv = []

    for year, gp, stype in SESSIONS:
        print(f"\nLoading {gp} {year} Race...")
        session = load_race(year, gp)
        laps = get_clean_laps(session)
        print(f"  Clean laps: {len(laps)}")

        cv_df = compute_driver_cv(laps)
        cv_df["gp"] = gp
        all_cv.append(cv_df)

        low = cv_df[cv_df["low_laps"]]
        if len(low) > 0:
            print(f"  Low lap count drivers (<{LOW_LAP_THRESHOLD}): {list(low['driver'])}")

        print(f"\n  === {gp} CV Rankings (lower = more consistent) ===")
        print(cv_df.sort_values("cv_lap")[["driver", "team", "pu", "n_laps", "cv_lap"]].to_string(index=False))

        fname = plot_cv_by_driver(cv_df, gp)
        print(f"  Saved: {fname}")

        fname = plot_sector_cv_heatmap(cv_df, gp)
        print(f"  Saved: {fname}")

    combined = pd.concat(all_cv, ignore_index=True)

    print("\n=== Combined CV by PU Manufacturer ===")
    print(combined.groupby("pu")["cv_lap"].agg(["median", "mean", "count"]).sort_values("median").to_string())

    plot_cv_vs_pu(combined)
    print("\nSaved: cv_by_pu_combined.png")


if __name__ == "__main__":
    main()