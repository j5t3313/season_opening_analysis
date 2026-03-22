import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


PU_MAP = {
    "Mercedes": "Mercedes",
    "McLaren": "Mercedes",
    "Williams": "Mercedes",
    "Ferrari": "Ferrari",
    "Haas": "Ferrari",
    "Cadillac": "Ferrari",
    "Red Bull": "Ford x RBPT",
    "Racing Bulls": "Ford x RBPT",
    "Aston Martin": "Honda",
    "Alpine": "Mercedes",
    "Audi": "Audi",
}

RACES = [
    {
        "gp": "Australia",
        "total_laps": 58,
        "entries": [
            ("Russell", "Mercedes", 58, "Classified"),
            ("Antonelli", "Mercedes", 58, "Classified"),
            ("Leclerc", "Ferrari", 58, "Classified"),
            ("Hamilton", "Ferrari", 58, "Classified"),
            ("Norris", "McLaren", 58, "Classified"),
            ("Verstappen", "Red Bull", 58, "Classified"),
            ("Bearman", "Haas", 57, "Classified"),
            ("Lindblad", "Racing Bulls", 57, "Classified"),
            ("Bortoleto", "Audi", 57, "Classified"),
            ("Gasly", "Alpine", 57, "Classified"),
            ("Ocon", "Haas", 57, "Classified"),
            ("Albon", "Williams", 57, "Classified"),
            ("Lawson", "Racing Bulls", 57, "Classified"),
            ("Colapinto", "Alpine", 56, "Classified"),
            ("Sainz", "Williams", 56, "Classified"),
            ("Perez", "Cadillac", 55, "Classified"),
            ("Stroll", "Aston Martin", 43, "DNF"),
            ("Alonso", "Aston Martin", 21, "DNF"),
            ("Bottas", "Cadillac", 15, "DNF"),
            ("Hadjar", "Red Bull", 10, "DNF"),
            ("Piastri", "McLaren", 0, "DNS"),
            ("Hulkenberg", "Audi", 0, "DNS"),
        ],
    },
    {
        "gp": "China",
        "total_laps": 56,
        "entries": [
            ("Antonelli", "Mercedes", 56, "Classified"),
            ("Russell", "Mercedes", 56, "Classified"),
            ("Hamilton", "Ferrari", 56, "Classified"),
            ("Leclerc", "Ferrari", 56, "Classified"),
            ("Bearman", "Haas", 56, "Classified"),
            ("Gasly", "Alpine", 56, "Classified"),
            ("Lawson", "Racing Bulls", 56, "Classified"),
            ("Hadjar", "Red Bull", 56, "Classified"),
            ("Sainz", "Williams", 55, "Classified"),
            ("Colapinto", "Alpine", 55, "Classified"),
            ("Hulkenberg", "Audi", 55, "Classified"),
            ("Lindblad", "Racing Bulls", 55, "Classified"),
            ("Bottas", "Cadillac", 55, "Classified"),
            ("Ocon", "Haas", 55, "Classified"),
            ("Perez", "Cadillac", 55, "Classified"),
            ("Verstappen", "Red Bull", 45, "DNF"),
            ("Alonso", "Aston Martin", 32, "DNF"),
            ("Stroll", "Aston Martin", 9, "DNF"),
            ("Piastri", "McLaren", 0, "DNS"),
            ("Norris", "McLaren", 0, "DNS"),
            ("Bortoleto", "Audi", 0, "DNS"),
            ("Albon", "Williams", 0, "DNS"),
        ],
    },
]

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


def build_dataframe():
    rows = []
    for race in RACES:
        total = race["total_laps"]
        for driver, team, laps, status in race["entries"]:
            if status == "DNS":
                continue
            rows.append({
                "gp": race["gp"],
                "driver": driver,
                "team": team,
                "pu": PU_MAP[team],
                "laps": laps,
                "pct_complete": laps / total,
                "event": 1 if status == "DNF" else 0,
            })
    return pd.DataFrame(rows)


def build_dns_summary():
    rows = []
    for race in RACES:
        for driver, team, laps, status in race["entries"]:
            if status == "DNS":
                rows.append({
                    "gp": race["gp"],
                    "driver": driver,
                    "team": team,
                    "pu": PU_MAP[team],
                })
    return pd.DataFrame(rows)


def plot_km_by_pu(df):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    kmf = KaplanMeierFitter()

    for pu in sorted(df["pu"].unique()):
        sub = df[df["pu"] == pu]
        kmf.fit(
            sub["pct_complete"],
            event_observed=sub["event"],
            label=f"{pu} (n={len(sub)})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=PU_COLORS.get(pu, "#333333"),
            linewidth=2,
        )

    ax.set_xlabel("Race completion (%)", fontsize=11, color="#1a1a1a")
    ax.set_ylabel("Survival probability", fontsize=11, color="#1a1a1a")
    ax.set_title(
        "Kaplan-Meier Survival by PU Manufacturer\nAustralia + China 2026 (DNS excluded)",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, color="#e0e0e0", alpha=0.6, linewidth=0.5)
    ax.legend(fontsize=10, framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.tick_params(colors="#4a4a4a", labelsize=9)

    watermark(fig)
    fig.tight_layout()
    fig.savefig("km_survival_by_pu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_km_by_team_type(df):
    works = {"Mercedes", "Ferrari", "Red Bull", "Alpine", "Audi", "Aston Martin"}
    df = df.copy()
    df["team_type"] = df["team"].apply(lambda t: "Works" if t in works else "Customer")

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    kmf = KaplanMeierFitter()
    colors = {"Works": "#1e41ff", "Customer": "#dc0000"}

    for tt in ["Works", "Customer"]:
        sub = df[df["team_type"] == tt]
        kmf.fit(
            sub["pct_complete"],
            event_observed=sub["event"],
            label=f"{tt} (n={len(sub)})",
        )
        kmf.plot_survival_function(ax=ax, color=colors[tt], linewidth=2)

    ax.set_xlabel("Race completion (%)", fontsize=11, color="#1a1a1a")
    ax.set_ylabel("Survival probability", fontsize=11, color="#1a1a1a")
    ax.set_title(
        "Kaplan-Meier Survival: Works vs Customer Teams\nAustralia + China 2026 (DNS excluded)",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(True, color="#e0e0e0", alpha=0.6, linewidth=0.5)
    ax.legend(fontsize=10, framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.tick_params(colors="#4a4a4a", labelsize=9)

    watermark(fig)
    fig.tight_layout()
    fig.savefig("km_survival_works_vs_customer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pairwise_logrank(df):
    pus = sorted(df["pu"].unique())
    results = []
    for i in range(len(pus)):
        for j in range(i + 1, len(pus)):
            a = df[df["pu"] == pus[i]]
            b = df[df["pu"] == pus[j]]
            if len(a[a["event"] == 1]) == 0 and len(b[b["event"] == 1]) == 0:
                continue
            lr = logrank_test(
                a["pct_complete"], b["pct_complete"],
                event_observed_A=a["event"], event_observed_B=b["event"],
            )
            results.append({
                "group_a": pus[i],
                "group_b": pus[j],
                "test_statistic": round(lr.test_statistic, 4),
                "p_value": round(lr.p_value, 4),
            })
    return pd.DataFrame(results)


def print_dns_summary(dns_df):
    print("\n=== DNS Summary ===")
    print(f"Total DNS: {len(dns_df)} across 2 races\n")
    for _, row in dns_df.iterrows():
        print(f"  {row['gp']:12s}  {row['driver']:15s}  {row['team']:15s}  ({row['pu']})")
    print()
    print("DNS by PU manufacturer:")
    print(dns_df.groupby("pu").size().sort_values(ascending=False).to_string())


def print_dnf_summary(df):
    dnf = df[df["event"] == 1].sort_values("pct_complete")
    print("\n=== DNF Summary ===")
    print(f"Total DNF: {len(dnf)} across 2 races\n")
    for _, row in dnf.iterrows():
        print(
            f"  {row['gp']:12s}  {row['driver']:15s}  {row['team']:15s}  "
            f"({row['pu']:12s})  Lap exit: {row['pct_complete']:.0%}"
        )


def main():
    df = build_dataframe()
    dns_df = build_dns_summary()

    print_dns_summary(dns_df)
    print_dnf_summary(df)

    print("\n=== Pairwise Log-Rank Tests ===")
    lr = pairwise_logrank(df)
    if len(lr) > 0:
        print(lr.to_string(index=False))
    else:
        print("Insufficient events for pairwise comparison.")

    print(f"\nNote: n={len(df)} car-starts across 2 races. "
          f"DNS (n={len(dns_df)}) excluded from survival analysis. ")

    plot_km_by_pu(df)
    plot_km_by_team_type(df)

    print("\nSaved: km_survival_by_pu.png, km_survival_works_vs_customer.png")


if __name__ == "__main__":
    main()