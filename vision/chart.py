"""Candlestick chart generator with a pure matplotlib implementation."""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import CHART_DIR, SYMBOL, TIMEFRAME


class ChartGenerator:
    def create(
        self,
        df,
        filename: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> str:
        if df is None or len(df) < 2:
            raise ValueError("not enough bars to draw a chart")
        work = df.reset_index(drop=True)
        chart_symbol = symbol or SYMBOL
        chart_timeframe = timeframe or TIMEFRAME
        filename = filename or os.path.join(
            CHART_DIR,
            f"{chart_symbol}_{chart_timeframe}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        )

        fig, (ax_price, ax_vol) = plt.subplots(
            2,
            1,
            figsize=(10, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        x = list(range(len(work)))
        for i, row in work.iterrows():
            color = "#16a34a" if row["close"] >= row["open"] else "#dc2626"
            ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
            body_low = min(row["open"], row["close"])
            body_high = max(row["open"], row["close"])
            ax_price.add_patch(
                Rectangle(
                    (i - 0.3, body_low),
                    0.6,
                    max(body_high - body_low, 1e-6),
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.9,
                )
            )

        close = work["close"]
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        ax_price.plot(x, sma20, label="SMA20", color="#2563eb", linewidth=1.2)
        ax_price.plot(x, sma50, label="SMA50", color="#ea580c", linewidth=1.2)
        ax_price.set_title(
            f"{chart_symbol} {chart_timeframe} candlestick chart"
        )
        ax_price.legend(loc="upper left")
        ax_price.grid(alpha=0.25)
        ax_vol.bar(x, work["tick_volume"], color="#94a3b8", alpha=0.65)
        ax_vol.set_ylabel("Volume")
        fig.tight_layout()
        fig.savefig(filename, dpi=110)
        plt.close(fig)
        return filename
