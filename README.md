# AI Trading Agent (V1-V5)

把 ChatGPT 对话里的 AI 自主交易系统从架构设计实现为可运行的软件。系统不是固定策略 EA，而是一个由 AI 决策、风险防火墙保护、带记忆和复盘学习的多 Agent 交易员。

English version: [README_en.md](README_en.md)

## Public repository note

本仓库只公开源码、测试和部署文件，不提交 `.env`、行情 CSV、SQLite 数据库、日志或生成的图表。

## Rules & Contributing

- 规则总览：[docs/RULES.md](docs/RULES.md)
- 机器可读规则清单：[rules/manifest.json](rules/manifest.json)
- 贡献指南：[CONTRIBUTING.md](CONTRIBUTING.md)
- 建议新规则：[Rule suggestion issue](.github/ISSUE_TEMPLATE/rule_suggestion.md)

运行前请执行：

```bash
copy .env.example .env
```

然后填入 LLM Key；历史行情 CSV 由 MT5 导出后放入 `data/history_{SYMBOL}_{TIMEFRAME}.csv`，或直接连接 MT5 获取。

## V1-V5 对应实现

| 阶段 | 构思 | 实现位置 |
| --- | --- | --- |
| V1 | AI + MT5 行情 + 手动确认交易闭环 | `main.py --mode v1`，`agent/`、`mt5/`、`memory/` |
| V2 | 自动交易执行层：仓位计算、止损止盈、移动止损、保本、分批止盈 | `main.py --mode v2`，`execution/order_manager.py`、`execution/paper_broker.py` |
| V3 | AI 看 K 线：生成图片 + Vision 分析 | `main.py --mode v3`，`vision/chart.py`、`vision/analyzer.py` |
| V4 | AI 每日复盘：找错误、更新交易经验 | `main.py --mode v4`，`agent/reflection_agent.py`、`brain/memory.py` |
| V5 | 24 小时多 Agent 自主交易 + 心跳监控 + Telegram 报警 + Docker 部署 | `main.py --mode v5`，`agent/ceo.py`、`monitor/`、`deployment/` |

## 快速开始（连接你本机 MT5 真实行情，模拟执行，不下真单）

项目自带 `.env`，默认配置：

```env
SIMULATE=false        # 使用 MT5 真实行情
USE_MOCK_LLM=true     # 未填 Key 时用离线 AI，填 Key 后改为 false
EXECUTION_MODE=paper  # 订单只在本机模拟撮合，不发送到 MT5
```

直接运行：

```bash
python main.py --mode demo --cycles 3 --fresh
```

系统会读取你 MT5 里的 XAUUSD H1 真实数据，跑完整 V1-V5 流程，但所有成交都在本地纸面账户，不会动 MT5 持仓。AI 没有明显优势时会输出 `WAIT`，这是 V1 的纪律设计。

想完全离线体验（不依赖 MT5），把 `.env` 里 `SIMULATE=true` 后再运行。

当你准备好真实交易，把 `EXECUTION_MODE=mt5` 并填写 MT5 登录信息后再运行；建议先在模拟账户上验证。

默认 `SIMULATE=true` 和 `USE_MOCK_LLM=true`，使用离线模拟行情和确定性 Mock AI，完整跑一遍：

```text
市场读取 -> 多Agent分析 -> AI决策 -> 风控 -> 自动执行 -> 移动止损/分批止盈 -> 复盘学习 -> 心跳
```

数据写入 `data/trading.db`，K 线图输出到 `data/charts/`，心跳写入 `data/heartbeat.txt`。

## 各模式

```bash
# V1：单轮闭环，下单前手动确认
python main.py --mode v1 --manual

# V2：自动模拟交易，观察仓位/止损/移动止损
python main.py --mode v2 --cycles 10

# V3：生成 K 线图并用 Vision 分析
python main.py --mode v3 --cycles 3

# V4：只跑每日复盘
python main.py --mode v4

# 只读分析：真实行情下跑全部 Agent，不执行任何订单
python main.py --mode analyze

# V5：完整多 Agent 自主循环 + 复盘
python main.py --mode v5 --cycles 5 --interval 2

# 历史训练/自我进化：模拟多种策略人格并选出最佳
python main.py --mode train --environments 20 --bars 2500

# 历史回测：逐根K线回放完整Agent流程（默认本地AI，不消耗Token）
python main.py --mode backtest --bars 3000

# 回测想用真实 DeepSeek 决策（会消耗 Token）
python main.py --mode backtest --bars 3000 --real-llm
```

MT5 演示账户的历史往往不全。想按完整年份（跨牛熊）回测时，先在本机 MT5 的“历史数据中心”下载 XAUUSD H1 全部历史，导出为 CSV（列：`time,open,high,low,close,tick_volume`），放到：

```text
data/history_XAUUSD_H1.csv
```

回测会自动优先读取该 CSV，报告会输出 `bars_per_year`，方便确认每年覆盖了多少根 K 线。

回测已加入执行真实性：决策用上一根收盘，下一根开盘成交（避免未来函数）；入场按 `SPREAD`（默认 0.25）计点差并支持 `COMMISSION_PER_LOT` 佣金；报告会输出 `min_lot_risk_percent` 和 `sparse_years`，提醒小账户最小手数风险与数据覆盖不足。

V5 循环中每 5 轮会额外运行一次 `研究AI`，扫描 XAUUSD/EURUSD/GBPUSD/USDJPY 并输出机会评分。K 线视觉在没有视觉 API 时会自动使用本地结构分析（Higher High/Lower Low、回调/突破结构），不再只是“不可用”。

## 接入真实环境

复制 `.env.example` 为 `.env`，填写：

```env
SIMULATE=false
USE_MOCK_LLM=false
DEEPSEEK_KEY=sk-xxx
MT5_LOGIN=123456
MT5_PASSWORD=xxx
MT5_SERVER=xxx
```

然后运行：

```bash
python main.py --mode v5
```

`SIMULATE=false` 时会连接 MT5 并调用 DeepSeek；`MT5_LOGIN=0` 时 MT5 只做行情读取。

小账户保护：开仓前会按止损距离重新估算实际风险。若最小 0.01 手仍让单笔风险超过
`MAX_RISK_PERCENT` 的容差上限，机器人会拒绝该订单并记录原因，避免小账户被最小仓
放大亏损。可用 `ENABLE_MIN_LOT_RISK_GUARD=false` 关闭（不建议）。

## 生产部署

```bash
cd deployment
docker compose up -d
```

容器默认每 300 秒跑一个完整循环，`restart: always` 自动拉起。可以配置 Telegram 报警：

```env
TELEGRAM_TOKEN=123456:ABC
TELEGRAM_CHAT_ID=123456789
```

## V5 24 小时运行（本机）

双击 [run_v5.bat](run_v5.bat) 或执行：

```bash
python main.py --mode v5 --cycles 288 --interval 300
```

每 300 秒读取一次 MT5 真实行情并跑完整 Agent 循环；`EXECUTION_MODE=paper` 时不会向 MT5 下真实订单。填好 `DEEPSEEK_KEY` 并设置 `USE_MOCK_LLM=false` 后，同一命令会自动换成真实 AI 决策。

V5 也支持多品种/多周期组合：`.env` 的 `TRADING_UNIVERSE` 用
`SYMBOL:TIMEFRAME` 条目定义，例如
`XAUUSD:H1;XAUUSD:M15;EURUSD:H1;GBPUSD:H1;USDJPY:H1`。每个组合拥有独立的
决策状态；同一品种同时只允许一个持仓方向，不同品种最多并行
`MAX_OPEN_POSITIONS` 个仓位。

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖 Mock AI 决策、风控防火墙、仓位计算、模拟撮合的保本/移动止损/分批止盈，以及行情指标计算。

## 降低 Token 消耗

- `RESEARCH_INTERVAL`：研究 AI 扫描频率（默认每 10 轮；改小会更及时但更耗 Token）
- `CONTEXT_CLOSES`：传给模型的历史收盘数组数量（默认 16，减少输入 Token）
- 循环间隔 `--interval 600` 或 `900` 能显著降耗，但会降低决策频率
- DeepSeek 对重复前缀有自动上下文缓存，保持提示词稳定会进一步降低成本

## 项目结构

```text
AI_TRADING_AGENT/
├── main.py
├── config.py
├── agent/          # CEO/Market/Vision/Macro/Trader/Risk/Reflection
├── brain/          # LLM 客户端、提示词、记忆
├── execution/      # 订单构建 + 模拟撮合/止损管理
├── risk/           # 风控防火墙
├── mt5/            # MT5 连接、行情、账户、执行
├── vision/         # K 线图生成 + 视觉分析
├── memory/         # SQLite 决策/交易/经验/复盘/心跳
├── monitor/        # 心跳 + Telegram
├── deployment/     # Dockerfile + docker-compose
└── tools/          # 仓位与止损计算
```

## 免责声明

这是交易研究/自动化框架，不构成投资建议。真实交易前请用模拟账户充分验证；AI 不能保证盈利，风控和人工监督始终优先。
