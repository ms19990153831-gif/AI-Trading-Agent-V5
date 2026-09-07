import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = r"C:\Users\Administrator\Documents\Codex\2026-08-24\cc-switch-local-proxy-failed-while\outputs"

combos = [
    {"name": "E_stop2_rr3", "rr": "3.0", "stop": "2.0", "min": "2"},
    {"name": "F_stop25_rr25", "rr": "2.5", "stop": "2.5", "min": "2"},
    {"name": "G_stop2_min3", "rr": "2.5", "stop": "2.0", "min": "3"},
    {"name": "H_stop175_rr25", "rr": "2.5", "stop": "1.75", "min": "2"},
]


def start_one(combo):
    env = os.environ.copy()
    for key in (
        "FEATURE_MASK", "FEATURE_RR", "FEATURE_STOP_ATR",
        "FEATURE_MIN_OPTIONAL", "BACKTEST_LIGHT_CONTEXT",
        "BACKTEST_HISTORY_CSV", "BACKTEST_DB_PATH",
        "BACKTEST_REPORT_PATH", "BACKTEST_CHART_PATH",
    ):
        env.pop(key, None)
    env["FEATURE_MASK"] = "base,rsi,cci,macd,ict"
    env["FEATURE_RR"] = combo["rr"]
    env["FEATURE_STOP_ATR"] = combo["stop"]
    env["FEATURE_MIN_OPTIONAL"] = combo["min"]
    env["INITIAL_BALANCE"] = "1000"
    for key in (
        "ENABLE_VISION_GUARD", "ENABLE_EXTREME_GUARD",
        "ENABLE_TREND_EXHAUSTION_GUARD", "ENABLE_TREND_CONFLICT_GUARD",
        "ENABLE_SIDE_LOSS_PAUSE", "ENABLE_BOUNDARY_GUARD",
        "ENABLE_QUALITY_GATE", "ENABLE_AI_CLOSE", "ENABLE_REVERSAL_SIGNAL",
        "ENABLE_MACRO_GATE",
    ):
        env[key] = "false"
    env["BACKTEST_DB_PATH"] = os.path.join(OUT, f"bt_{combo['name']}.db")
    env["BACKTEST_REPORT_PATH"] = os.path.join(OUT, f"bt_{combo['name']}.json")
    env["BACKTEST_CHART_PATH"] = os.path.join(OUT, f"bt_{combo['name']}.png")
    code = (
        "from backtest import run_backtest;"
        "r=run_backtest(bars=60000,decision_start='2025-01-01',"
        "end='2026-09-02 00:00:00');"
        "print('RESULT',r['closed_trades'],r['wins'],round(r['win_rate'],4),"
        "round(r['profit_factor'],3),round(r['net_pnl'],2),"
        "round(r['max_drawdown'],4))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


if __name__ == "__main__":
    procs = []
    for combo in combos:
        procs.append((combo, start_one(combo)))
    for combo, proc in procs:
        stdout, stderr = proc.communicate(timeout=1800)
        with open(os.path.join(OUT, f"bt_{combo['name']}.out"), "w", encoding="utf-8") as f:
            f.write(stdout + "\n" + stderr)
        line = stdout.strip().splitlines()[-1] if stdout.strip() else "ERR"
        print(combo["name"], line)
