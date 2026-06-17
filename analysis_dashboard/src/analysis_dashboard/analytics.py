from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger("analysis_dashboard.analytics")

# Use a clean seaborn theme
sns.set_theme(style="darkgrid", palette="muted")

# Resolve charts/ relative to the repo root, not the process cwd.
# __file__ = <repo_root>/analysis_dashboard/src/analysis_dashboard/analytics.py
# parents[3] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_env_chart_dir = os.getenv("CHART_OUTPUT_DIR", "")
CHART_OUTPUT_DIR = Path(_env_chart_dir) if _env_chart_dir else (_REPO_ROOT / "charts")


def _ensure_chart_dir() -> Path:
    CHART_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return CHART_OUTPUT_DIR


def _fig_to_base64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64 PNG string for Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _save_fig(fig: plt.Figure, save_path: str | None, default_name: str) -> str | None:
    """Save figure to disk if save_path is provided. Returns final path."""
    if save_path:
        out = Path(save_path)
    else:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    logger.info("Chart saved to %s", out)
    return str(out)


# ─────────────────────────────────────────────────────────────
# Analytics functions (called by @tool wrappers in agent.py)
# ─────────────────────────────────────────────────────────────

def compute_latency_trend(
    log_entries: list[dict],
    window: int = 5,
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute per-tool latency moving averages over time.

    Parameters
    ----------
    log_entries : Raw rows from HLPLogStore.get_all()
    window      : Rolling window size for moving average
    save_path   : Optional path to save the chart PNG

    Returns
    -------
    dict with keys: summary (text), chart_b64 (base64 PNG), saved_path
    """
    df = pd.DataFrame(log_entries)
    if df.empty or "latency_ms" not in df.columns:
        return {"summary": "No latency data available.", "chart_b64": None, "saved_path": None}

    df = df[df["latency_ms"].notna()].copy()
    if df.empty:
        return {"summary": "No latency records with measured values.", "chart_b64": None, "saved_path": None}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")
    df["tool_name"] = df["tool_name"].fillna("unknown")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Tool Latency Trend Analysis", fontsize=14, fontweight="bold")

    # Panel 1: per-tool latency scatter + moving average
    ax1 = axes[0]
    tools = df["tool_name"].unique()
    palette = sns.color_palette("tab10", len(tools))

    for i, tool in enumerate(tools):
        tool_df = df[df["tool_name"] == tool].reset_index(drop=True)
        if len(tool_df) < 2:
            continue
        xs = range(len(tool_df))
        ax1.scatter(xs, tool_df["latency_ms"], alpha=0.4, s=20, color=palette[i], label=f"{tool} (raw)")
        if len(tool_df) >= window:
            ma = tool_df["latency_ms"].rolling(window, min_periods=1).mean()
            ax1.plot(xs, ma, color=palette[i], linewidth=2, label=f"{tool} (MA-{window})")

    ax1.set_xlabel("Event Index")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-Tool Latency with Moving Average")
    ax1.legend(fontsize=8)

    # Panel 2: box plot of latency distribution per tool
    ax2 = axes[1]
    latency_data = [
        df[df["tool_name"] == t]["latency_ms"].dropna().values
        for t in tools
        if len(df[df["tool_name"] == t]["latency_ms"].dropna()) > 0
    ]
    valid_tools = [
        t for t in tools
        if len(df[df["tool_name"] == t]["latency_ms"].dropna()) > 0
    ]

    if latency_data:
        bp = ax2.boxplot(latency_data, labels=valid_tools, patch_artist=True)
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax2.set_xlabel("Tool")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Distribution per Tool")
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    chart_b64 = _fig_to_base64(fig)
    if save_path is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_path = str(_ensure_chart_dir() / f"latency_trend_{ts_str}.png")
    saved     = _save_fig(fig, save_path, "latency_trend.png")
    plt.close(fig)

    # Compute summary stats
    stats = df.groupby("tool_name")["latency_ms"].agg(["mean", "median", "max", "count"]).round(1)
    summary_lines = ["Latency Summary (ms):"]
    for tool, row in stats.iterrows():
        summary_lines.append(
            f"  {tool}: mean={row['mean']:.1f}  median={row['median']:.1f}  "
            f"max={row['max']:.1f}  n={int(row['count'])}"
        )
    return {
        "summary":    "\n".join(summary_lines),
        "chart_b64":  chart_b64,
        "saved_path": saved,
        "stats":      stats.to_dict(),
    }


def compute_token_metrics(
    log_entries: list[dict],
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Analyse token consumption patterns grouped by interaction_type and session.
    """
    df = pd.DataFrame(log_entries)
    if df.empty or "token_count" not in df.columns:
        return {"summary": "No token data available.", "chart_b64": None, "saved_path": None}

    df = df[df["token_count"].notna()].copy()
    if df.empty:
        return {"summary": "No entries with token counts.", "chart_b64": None, "saved_path": None}

    df["token_count"] = pd.to_numeric(df["token_count"], errors="coerce").fillna(0).astype(int)
    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Token Consumption Analysis", fontsize=14, fontweight="bold")

    # Panel 1: stacked bar by interaction_type
    ax1 = axes[0]
    type_totals = df.groupby("interaction_type")["token_count"].sum().sort_values(ascending=False)
    if not type_totals.empty:
        bars = ax1.bar(type_totals.index, type_totals.values,
                       color=sns.color_palette("muted", len(type_totals)))
        ax1.bar_label(bars, padding=2, fontsize=8)
    ax1.set_xlabel("Interaction Type")
    ax1.set_ylabel("Total Tokens")
    ax1.set_title("Token Usage by Interaction Type")
    ax1.tick_params(axis="x", rotation=25)

    # Panel 2: cumulative token usage over time
    ax2 = axes[1]
    df["cumulative_tokens"] = df["token_count"].cumsum()
    ax2.plot(range(len(df)), df["cumulative_tokens"], color="steelblue", linewidth=2)
    ax2.fill_between(range(len(df)), df["cumulative_tokens"], alpha=0.3, color="steelblue")
    ax2.set_xlabel("Event Index")
    ax2.set_ylabel("Cumulative Tokens")
    ax2.set_title("Cumulative Token Consumption Over Time")

    plt.tight_layout()
    chart_b64 = _fig_to_base64(fig)
    if save_path is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_path = str(_ensure_chart_dir() / f"token_metrics_{ts_str}.png")
    saved     = _save_fig(fig, save_path, "token_metrics.png")
    plt.close(fig)

    total   = df["token_count"].sum()
    by_type = df.groupby("interaction_type")["token_count"].sum().to_dict()
    summary = (
        f"Total tokens consumed: {total:,}\n"
        + "\n".join(f"  {k}: {v:,}" for k, v in by_type.items())
    )
    return {
        "summary":    summary,
        "chart_b64":  chart_b64,
        "saved_path": saved,
        "total_tokens": int(total),
        "by_type":    by_type,
    }


def compute_error_frequency(
    log_entries: list[dict],
    window_minutes: int = 5,
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute rolling error frequency counts over time windows.
    """
    df = pd.DataFrame(log_entries)
    if df.empty:
        return {"summary": "No log entries.", "chart_b64": None, "saved_path": None}

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    errors_df = df[df["interaction_type"] == "error"].copy()
    total_errors = len(errors_df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error Frequency Analysis", fontsize=14, fontweight="bold")

    # Panel 1: error vs non-error timeline
    ax1 = axes[0]
    df["is_error"] = (df["interaction_type"] == "error").astype(int)
    ax1.fill_between(range(len(df)), df["is_error"], alpha=0.5, color="crimson", label="Error")
    normal = (df["interaction_type"] != "error").astype(int)
    ax1.fill_between(range(len(df)), normal, alpha=0.2, color="steelblue", label="Normal")
    ax1.set_xlabel("Event Index")
    ax1.set_ylabel("Event Type (1=True)")
    ax1.set_title("Error vs Normal Events Timeline")
    ax1.legend()

    # Panel 2: error rate by interaction type
    ax2 = axes[1]
    type_counts = df.groupby("interaction_type").size()
    error_count = type_counts.get("error", 0)
    colors = ["crimson" if t == "error" else "steelblue" for t in type_counts.index]
    bars = ax2.bar(type_counts.index, type_counts.values, color=colors)
    ax2.bar_label(bars, padding=2, fontsize=8)
    ax2.set_xlabel("Interaction Type")
    ax2.set_ylabel("Count")
    ax2.set_title("Event Distribution by Type")
    ax2.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    chart_b64 = _fig_to_base64(fig)
    if save_path is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_path = str(_ensure_chart_dir() / f"error_frequency_{ts_str}.png")
    saved     = _save_fig(fig, save_path, "error_frequency.png")
    plt.close(fig)

    summary = (
        f"Total errors: {total_errors} / {len(df)} events "
        f"({100*total_errors/max(len(df),1):.1f}% error rate)\n"
    )
    if not errors_df.empty:
        summary += "Error namespaces:\n"
        ns_counts = errors_df["namespace"].value_counts()
        for ns, cnt in ns_counts.items():
            summary += f"  {ns}: {cnt}\n"

    return {
        "summary":      summary,
        "chart_b64":    chart_b64,
        "saved_path":   saved,
        "total_errors": total_errors,
        "total_events": len(df),
        "error_rate":   total_errors / max(len(df), 1),
    }


def generate_dashboard_chart(
    log_entries: list[dict],
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Generate a comprehensive 4-panel system health dashboard chart.
    Always saves to CHART_OUTPUT_DIR unless save_path overrides.
    """
    df = pd.DataFrame(log_entries)
    if df.empty:
        return {"summary": "No data for dashboard chart.", "chart_b64": None, "saved_path": None}

    df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["token_count"] = pd.to_numeric(df.get("token_count", 0), errors="coerce").fillna(0)
    df["latency_ms"]  = pd.to_numeric(df.get("latency_ms", None), errors="coerce")
    df["is_error"]    = (df.get("interaction_type", "") == "error").astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("HLP System Health Dashboard", fontsize=16, fontweight="bold")

    # Panel 1: Events by interaction_type
    ax1 = axes[0, 0]
    type_counts = df["interaction_type"].value_counts()
    wedges, texts, autotexts = ax1.pie(
        type_counts.values,
        labels=type_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("muted", len(type_counts)),
    )
    ax1.set_title("Event Distribution")

    # Panel 2: Latency over time (if available)
    ax2 = axes[0, 1]
    lat_df = df[df["latency_ms"].notna()]
    if not lat_df.empty:
        ax2.scatter(lat_df.index, lat_df["latency_ms"], alpha=0.5, s=15, color="steelblue")
        ma = lat_df["latency_ms"].rolling(5, min_periods=1).mean()
        ax2.plot(lat_df.index, ma, color="crimson", linewidth=2, label="MA-5")
        ax2.set_xlabel("Event Index")
        ax2.set_ylabel("Latency (ms)")
        ax2.legend()
    ax2.set_title("Latency Over Time")

    # Panel 3: Cumulative tokens
    ax3 = axes[1, 0]
    df["cum_tokens"] = df["token_count"].cumsum()
    ax3.plot(df.index, df["cum_tokens"], color="darkorange", linewidth=2)
    ax3.fill_between(df.index, df["cum_tokens"], alpha=0.3, color="darkorange")
    ax3.set_xlabel("Event Index")
    ax3.set_ylabel("Cumulative Tokens")
    ax3.set_title("Cumulative Token Consumption")

    # Panel 4: Error rate by namespace
    ax4 = axes[1, 1]
    ns_errors = df.groupby("namespace")["is_error"].agg(["sum", "count"])
    ns_errors["error_rate"] = ns_errors["sum"] / ns_errors["count"].clip(lower=1)
    ns_errors = ns_errors.sort_values("error_rate", ascending=False).head(8)
    if not ns_errors.empty:
        colors = ["crimson" if r > 0.1 else "steelblue" for r in ns_errors["error_rate"]]
        bars = ax4.barh(ns_errors.index, ns_errors["error_rate"] * 100, color=colors)
        ax4.set_xlabel("Error Rate (%)")
        ax4.set_title("Error Rate by Namespace")
        ax4.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    chart_b64 = _fig_to_base64(fig)

    # Auto-save to chart output dir
    if save_path is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        save_path = str(_ensure_chart_dir() / f"dashboard_{ts_str}.png")
    saved = _save_fig(fig, save_path, "dashboard.png")
    plt.close(fig)

    summary = (
        f"Dashboard chart generated.\n"
        f"  Total events: {len(df)}\n"
        f"  Sessions: {df['session_id'].nunique()}\n"
        f"  Error rate: {df['is_error'].mean()*100:.1f}%\n"
        f"  Avg latency: {lat_df['latency_ms'].mean():.1f} ms"
        if not lat_df.empty
        else f"Dashboard chart generated. Total events: {len(df)}"
    )
    return {
        "summary":    summary,
        "chart_b64":  chart_b64,
        "saved_path": saved,
    }