import os
from collections import defaultdict

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


CACHE_DIR = "cache"
SAVGOL_WINDOW = 13
SAVGOL_POLY = 2
THROTTLE_THRESHOLD = 95
DECEL_THRESHOLD = -0.02
LAP_TIME_PCT = 1.07
MIN_SPEED_LOSS = 30

SESSIONS = [
    (2026, "Australia", "Q"),
    (2026, "China", "Q"),
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

TEAM_COLORS = {
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "Red Bull Racing": "#0f1b3c",
    "McLaren": "#FF8000",
    "Alpine": "#FF87BC",
    "Aston Martin": "#006F62",
    "Haas": "#B6BABD",
    "Racing Bulls": "#6692FF",
    "Williams": "#1868DB",
    "Audi": "#00E701",
    "Cadillac": "#C5A647",
}

DRIVER_STYLES = ["-", "--"]

STRAIGHTS = {
    "Australia": [
        {"name": "T8-T9 back straight", "dist_start": 3400, "dist_end": 4200},
    ],
    "China": [
        {"name": "T13-T14 back straight", "dist_start": 3800, "dist_end": 4700},
        {"name": "Pit straight", "dist_start": 0, "dist_end": 600},
    ],
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


def load_session_telemetry(year, gp, session_type):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, laps=True, weather=False)
    return session


def get_session_fastest(session):
    fastest = session.laps.pick_fastest()
    if fastest is None or pd.isna(fastest["LapTime"]):
        return None
    return fastest["LapTime"]


def get_fastest_lap_telemetry(session, driver_number, session_fastest):
    lap = session.laps.pick_drivers(driver_number).pick_fastest()
    if lap is None or (hasattr(lap, "empty") and lap.empty) or pd.isna(lap["LapTime"]):
        return None, None

    if session_fastest is not None:
        cutoff = session_fastest.total_seconds() * LAP_TIME_PCT
        if lap["LapTime"].total_seconds() > cutoff:
            return None, None

    car = lap.get_car_data().add_distance()

    for col in ["Speed", "Throttle"]:
        car[col] = car[col].interpolate(method="linear")
    car = car.dropna(subset=["Speed", "Throttle", "Distance"]).reset_index(drop=True)

    if len(car) < SAVGOL_WINDOW:
        return None, None

    car["Speed"] = savgol_filter(car["Speed"].values, SAVGOL_WINDOW, SAVGOL_POLY)
    car["dSpeed"] = np.gradient(car["Speed"].values, car["Distance"].values)
    car["is_clipping"] = (
        (car["Throttle"] >= THROTTLE_THRESHOLD) & (car["dSpeed"] < DECEL_THRESHOLD)
    )

    meta = {
        "driver": lap["Driver"],
        "team": lap["Team"],
        "pu": resolve_pu(str(lap["Team"])),
        "lap_time": lap["LapTime"].total_seconds(),
    }

    return car, meta


def extract_straight_metrics(car_df, straight):
    seg = car_df[
        (car_df["Distance"] >= straight["dist_start"])
        & (car_df["Distance"] <= straight["dist_end"])
    ].copy()

    if len(seg) < 5:
        return None

    peak_speed = seg["Speed"].max()
    exit_speed = seg["Speed"].iloc[-1]
    speed_loss = peak_speed - exit_speed

    if speed_loss < MIN_SPEED_LOSS:
        return None

    clip_samples = seg["is_clipping"].sum()
    clip_pct = clip_samples / len(seg) * 100

    clip_onset = None
    if clip_samples > 0:
        first_clip = seg[seg["is_clipping"]].iloc[0]
        clip_onset = first_clip["Distance"] - straight["dist_start"]

    return {
        "peak_speed": round(peak_speed, 1),
        "exit_speed": round(exit_speed, 1),
        "speed_loss": round(speed_loss, 1),
        "clip_pct": round(clip_pct, 1),
        "clip_onset_m": round(clip_onset, 0) if clip_onset is not None else None,
    }


def resolve_team_color(team_name):
    for key, color in TEAM_COLORS.items():
        if key.lower() in team_name.lower():
            return color
    return "#333333"


def plot_straight_overlay(traces, straight, gp):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    team_driver_count = defaultdict(int)

    for trace in traces:
        seg = trace["car"][
            (trace["car"]["Distance"] >= straight["dist_start"])
            & (trace["car"]["Distance"] <= straight["dist_end"])
        ]
        if len(seg) < 5:
            continue

        team = trace["meta"]["team"]
        driver = trace["meta"]["driver"]
        color = resolve_team_color(team)
        idx = team_driver_count[team]
        team_driver_count[team] += 1
        ls = DRIVER_STYLES[idx % len(DRIVER_STYLES)]

        ax.plot(
            seg["Distance"], seg["Speed"],
            color=color, linewidth=2, linestyle=ls,
            label=f"{driver} ({team})",
        )

    ax.set_xlabel("Distance (m)", fontsize=11, color="#1a1a1a")
    ax.set_ylabel("Speed (km/h)", fontsize=11, color="#1a1a1a")
    ax.set_title(
        f"Speed Trace Comparison: {straight['name']}\n{gp} 2026 Qualifying (within 107% of session fastest)",
        fontsize=13, fontweight="bold", color="#1a1a1a",
    )
    ax.grid(True, color="#e0e0e0", alpha=0.6, linewidth=0.5)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)

    for spine in ax.spines.values():
        spine.set_color("#e0e0e0")
    ax.tick_params(colors="#4a4a4a", labelsize=9)

    watermark(fig)
    fig.tight_layout()
    fname = f"straight_overlay_{gp.lower()}_{straight['name'].replace(' ', '_').replace('-', '_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_clipping_summary(all_metrics):
    df = pd.DataFrame(all_metrics)
    df = df.dropna(subset=["speed_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#ffffff")

    for ax in axes:
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#e0e0e0")
        ax.tick_params(colors="#4a4a4a", labelsize=9)
        ax.grid(True, axis="y", color="#e0e0e0", alpha=0.6, linewidth=0.5)

    pu_order = df.groupby("pu")["speed_loss"].median().sort_values(ascending=False).index
    positions = range(len(pu_order))
    for i, pu in enumerate(pu_order):
        sub = df[df["pu"] == pu]
        color = PU_COLORS.get(pu, "#333333")
        jitter = np.random.uniform(-0.15, 0.15, len(sub))
        axes[0].scatter(i + jitter, sub["speed_loss"], color=color, s=50, alpha=0.7, zorder=3)
        axes[0].hlines(sub["speed_loss"].median(), i - 0.25, i + 0.25, color=color, linewidth=2, zorder=4)

    axes[0].set_xticks(list(positions))
    axes[0].set_xticklabels(pu_order, fontsize=9)
    axes[0].set_ylabel("Speed loss on straight (km/h)", fontsize=11, color="#1a1a1a")
    axes[0].set_title("Clipping severity by PU", fontsize=12, fontweight="bold", color="#1a1a1a")

    for i, pu in enumerate(pu_order):
        sub = df[df["pu"] == pu]
        color = PU_COLORS.get(pu, "#333333")
        jitter = np.random.uniform(-0.15, 0.15, len(sub))
        axes[1].scatter(i + jitter, sub["clip_pct"], color=color, s=50, alpha=0.7, zorder=3)
        axes[1].hlines(sub["clip_pct"].median(), i - 0.25, i + 0.25, color=color, linewidth=2, zorder=4)

    axes[1].set_xticks(list(positions))
    axes[1].set_xticklabels(pu_order, fontsize=9)
    axes[1].set_ylabel("Straight spent clipping (%)", fontsize=11, color="#1a1a1a")
    axes[1].set_title("Clipping frequency by PU", fontsize=12, fontweight="bold", color="#1a1a1a")

    fig.suptitle(
        "Super-Clipping Comparison by PU Manufacturer\nAustralia + China 2026 Qualifying",
        fontsize=14, fontweight="bold", color="#1a1a1a", y=1.02,
    )
    watermark(fig)
    fig.tight_layout()
    fig.savefig("clipping_summary_by_pu.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    all_metrics = []
    all_traces = {}

    for year, gp, stype in SESSIONS:
        print(f"\nLoading {gp} {year} {stype}...")
        session = load_session_telemetry(year, gp, stype)
        session_fastest = get_session_fastest(session)
        if session_fastest is not None:
            print(f"  Session fastest: {session_fastest.total_seconds():.3f}s")
            print(f"  107% cutoff: {session_fastest.total_seconds() * LAP_TIME_PCT:.3f}s")

        drivers = session.laps["DriverNumber"].unique()
        traces = []
        excluded = []
        for drv in drivers:
            car, meta = get_fastest_lap_telemetry(session, drv, session_fastest)
            if car is None:
                excluded.append(drv)
                continue
            traces.append({"car": car, "meta": meta})

            if gp in STRAIGHTS:
                for straight in STRAIGHTS[gp]:
                    metrics = extract_straight_metrics(car, straight)
                    if metrics is not None:
                        metrics["driver"] = meta["driver"]
                        metrics["team"] = meta["team"]
                        metrics["pu"] = meta["pu"]
                        metrics["gp"] = gp
                        metrics["straight"] = straight["name"]
                        all_metrics.append(metrics)

        if excluded:
            print(f"  Excluded (no valid lap or outside 107%): {excluded}")
        print(f"  Included: {len(traces)} drivers")

        all_traces[gp] = traces

        if gp in STRAIGHTS:
            for straight in STRAIGHTS[gp]:
                fname = plot_straight_overlay(traces, straight, gp)
                print(f"  Saved: {fname}")

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print("\n=== Straight Metrics Summary ===")
        print(metrics_df.to_string(index=False))

        print("\n=== Median Speed Loss by PU ===")
        print(metrics_df.groupby("pu")["speed_loss"].agg(["median", "mean", "count"]).to_string())

        plot_clipping_summary(all_metrics)
        print("\nSaved: clipping_summary_by_pu.png")


if __name__ == "__main__":
    main()