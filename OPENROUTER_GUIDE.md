# OpenRouter API 使用指南

## 什么是 OpenRouter

OpenRouter（https://openrouter.ai）是一个 LLM API 聚合代理，兼容 OpenAI SDK。一个 API key 可以调用 300+ 模型（GPT-4o、Claude、Gemini、DeepSeek、MiniMax、Qwen 等），按量计费，不需要分别注册各家平台。

---

## 1. 注册 & 获取 API Key

1. 访问 https://openrouter.ai/keys
2. 用 Google / GitHub 登录
3. 点击 **Create Key**，复制得到 `sk-or-v1-xxxx...`
4. 充值：Settings → Credits，充 $5-10 足够日常使用

---

## 2. 基本用法（Python）

安装 OpenAI SDK（OpenRouter 完全兼容）：

```bash
pip install openai
```

调用示例：

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-or-v1-你的key",
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="openai/gpt-4o-mini",   # 选你想用的模型
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
)

print(response.choices[0].message.content)
```

**和直接用 OpenAI 的唯一区别：** 多传一个 `base_url`，key 换成 OpenRouter 的。其他代码完全一样。

---

## 3. 可用模型 & 价格

完整列表：https://openrouter.ai/models

### 常用模型速查

| 模型 | model ID | Input $/M | Output $/M | 特点 |
|------|----------|-----------|------------|------|
| GPT-4o | `openai/gpt-4o` | $2.50 | $10.00 | OpenAI 旗舰 |
| GPT-4o-mini | `openai/gpt-4o-mini` | $0.15 | $0.60 | 性价比之王 |
| Claude Sonnet 4 | `anthropic/claude-sonnet-4` | $3.00 | $15.00 | Anthropic 旗舰 |
| Claude Haiku 3.5 | `anthropic/claude-3.5-haiku` | $0.80 | $4.00 | 快且便宜 |
| Gemini 2.5 Flash | `google/gemini-2.5-flash-preview` | $0.15 | $0.60 | Google 快速模型 |
| DeepSeek Chat (V3) | `deepseek/deepseek-chat` | $0.14 | $0.28 | 极便宜，中文强 |
| DeepSeek R1 | `deepseek/deepseek-r1` | $0.55 | $2.19 | 推理模型 |
| MiniMax M2.7 | `minimax/minimax-m2.7` | $0.30 | $1.20 | 新一代 agentic |
| MiniMax M2.5 免费 | `minimax/minimax-m2.5:free` | **$0** | **$0** | 完全免费 |
| Qwen3-235B | `qwen/qwen3-235b` | $0.40 | $2.20 | 中文最强开源 |
| Qwen3-30B-A3B | `qwen/qwen3-30b-a3b` | $0.10 | $0.40 | 轻量高效 |

> 价格单位：美元 / 百万 tokens。一般对话每轮约 500-2000 tokens。

---

## 4. 环境变量方式（推荐）

不要把 key 硬编码在代码里，用环境变量：

```bash
# .env 文件
OPENAI_API_KEY=sk-or-v1-你的key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),
)

response = client.chat.completions.create(
    model=os.environ.get("LLM_MODEL", "gpt-4o"),
    messages=[{"role": "user", "content": "你好"}],
)
```

这样切模型只改 `.env`，不改代码。

---

## 5. JSON 结构化输出

大多数模型支持 `response_format`，强制返回 JSON：

```python
response = client.chat.completions.create(
    model="openai/gpt-4o-mini",
    messages=[
        {"role": "system", "content": "输出 JSON 格式"},
        {"role": "user", "content": "分析这句话的情感：'今天天气真好'"},
    ],
    response_format={"type": "json_object"},
)
```

---

## 6. 如果之前用的是 OpenAI 官方 API

迁移到 OpenRouter 只需改两行：

```python
# 之前（OpenAI 官方）
client = OpenAI(api_key="sk-xxx")

# 之后（OpenRouter）
client = OpenAI(
    api_key="sk-or-v1-xxx",                    # 换 key
    base_url="https://openrouter.ai/api/v1",   # 加这行
)
```

model 参数保持不变（`gpt-4o` 等短名自动映射），或者用完整的 `openai/gpt-4o` 格式。

---

## 7. 查看用量

- Dashboard：https://openrouter.ai/activity
- 查余额：https://openrouter.ai/settings/credits
