"""Progress callbacks for training loops.

Separated from the training loops so scripts stay thin.
"""
from __future__ import annotations

from trickster.train_utils import _GAME_PTS_MAX
from trickster.training.bidding_loop import (
    BiddingTrainConfig,
    BiddingTrainStats,
    DISPLAY_ORDER,
)
from trickster.training.model_io import DK_LABELS


def fmt_time(seconds: float) -> str:
    """Format seconds as human-readable time (e.g. '1h 23m', '5m 30s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def fmt_pts(norm_sum: float, n: int) -> str:
    """Format normalized value-head sum as average game points."""
    if n == 0:
        return "  —  "
    return f"{(norm_sum / n) * _GAME_PTS_MAX / 2:+5.1f}"


# ---------------------------------------------------------------------------
#  Bidding progress callbacks
# ---------------------------------------------------------------------------

def bidding_progress_bar(cfg: BiddingTrainConfig):
    """Single updating line with progress bar and ETA."""
    bar_width = 30

    def on_progress(stats: BiddingTrainStats) -> None:
        step = stats.step
        frac = step / stats.total_steps
        filled = int(bar_width * frac)
        bar = "█" * filled + "░" * (bar_width - filled)

        elapsed = stats.train_time_s
        eta = elapsed / frac * (1 - frac) if frac > 0 else 0

        total_deals = stats.total_games + stats.total_passes
        deals_s = total_deals / max(0.1, elapsed)

        print(
            f"\r  {bar} {frac*100:5.1f}%  "
            f"{step}/{stats.total_steps}  "
            f"games={stats.total_games:,}  "
            f"{deals_s:.1f} deals/s  "
            f"elapsed {fmt_time(elapsed)}  "
            f"ETA {fmt_time(eta)}   ",
            end="", flush=True,
        )

        if step == stats.total_steps:
            print()

    return on_progress


def bidding_progress_verbose(cfg: BiddingTrainConfig):
    """Verbose progress: full per-contract breakdown every 5%."""
    interval = max(1, cfg.steps // 20)

    prev: dict = {
        "games": {},
        "pts": {},
        "wins": {},
        "total_games": 0,
        "total_passes": 0,
    }

    def on_progress(stats: BiddingTrainStats) -> None:
        step = stats.step
        if step % interval != 0 and step != 1 and step != stats.total_steps:
            return

        w_total_games = stats.total_games - prev["total_games"]
        w_total_passes = stats.total_passes - prev["total_passes"]
        w_total_deals = w_total_games + w_total_passes
        w_pass_pct = w_total_passes / max(1, w_total_deals) * 100

        print(
            f"  step {step:>5d}/{stats.total_steps} "
            f"({step / stats.total_steps * 100:4.0f}%)  "
            f"games={stats.total_games:>5d}  "
            f"pass={w_pass_pct:.0f}%  "
            f"lr={stats.lr:.1e}  [{stats.train_time_s:.0f}s]",
            flush=True,
        )

        print(f"    {'contract':<10} {'games':>5} {'avg_pts':>7} {'win%':>5} {'bid%':>5}")

        for dk in DISPLAY_ORDER:
            cum_g = stats.cumulative_games.get(dk, 0)
            cum_pts = stats.cumulative_pts.get(dk, 0.0)
            cum_wins = stats.cumulative_wins.get(dk, 0)

            w_g = cum_g - prev["games"].get(dk, 0)
            w_pts = cum_pts - prev["pts"].get(dk, 0.0)
            w_wins = cum_wins - prev["wins"].get(dk, 0)

            if w_g == 0:
                continue

            label = DK_LABELS.get(dk, dk)
            print(
                f"    {label:<10} {w_g:>5} "
                f"{fmt_pts(w_pts, w_g):>7} "
                f"{w_wins / w_g * 100:4.0f}% "
                f"{w_g / max(1, w_total_deals) * 100:4.0f}%"
            )

        parts = []
        for mkey in cfg.contract_keys:
            vl = stats.model_vloss.get(mkey, 0.0)
            pl = stats.model_ploss.get(mkey, 0.0)
            if vl > 0 or pl > 0:
                parts.append(f"{mkey} v={vl:.4f} p={pl:.4f}")
        if parts:
            print(f"    loss: {' | '.join(parts)}")
        print(flush=True)

        prev["games"] = dict(stats.cumulative_games)
        prev["pts"] = dict(stats.cumulative_pts)
        prev["wins"] = dict(stats.cumulative_wins)
        prev["total_games"] = stats.total_games
        prev["total_passes"] = stats.total_passes

    return on_progress
