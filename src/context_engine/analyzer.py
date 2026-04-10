"""LLM Context Engine — analyze game dialogue and produce emotion annotations."""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import APIError, APITimeoutError, OpenAI
from pydantic import ValidationError

from .models import EmotionAnnotation, EmotionCategory, EmotionState

_SYSTEM_PROMPT = """\
你是一位专业的游戏情感导演（AI Emotional Director）。你的任务是分析游戏对白的上下文，为角色"秦彻"的每一句台词生成情感标注。

## 角色简介
秦彻是一位表面冷淡、内心温柔的角色。他极度自律，说话简洁有力，关心他人时往往故作轻松。

## 情感分类体系（6 类）
- tender（温柔）：关心、安慰、心疼、温柔低语
- calm（沉稳）：日常对话、理性分析、平静陈述
- playful（俏皮）：调侃、轻松打趣、偶尔的幽默
- intense（激烈）：战斗、愤怒、紧张、严厉警告
- cold（冷淡）：疏离、威严、拒人于千里之外
- intimate（亲密）：私密场景、深情告白、低语呢喃

## 输出格式
严格输出以下 JSON（不要输出其他内容）：
{
  "emotion": "细粒度情感标签，如 gentle_reassuring",
  "intensity": 0.0-1.0 的情感强度,
  "pace": "slow / normal / fast",
  "style": "TTS 语音指令，必须是祈使句格式，如'用温柔且故作轻松的语气说'。要求简短（10-20字），以'用…的语气说'或'用…的口吻'结尾",
  "ref_emotion_category": "tender / calm / playful / intense / cold / intimate",
  "fish_audio_tags": "[tag]文本 格式，用于 Fish Audio 情感控制"
}
"""

_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "场景：战斗结束后，秦彻受伤但安慰女主\n"
            "台词：慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很\n"
            "近期情感历史：[intense(0.8), intense(0.6)]"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "emotion": "gentle_reassuring",
                "intensity": 0.6,
                "pace": "slow",
                "style": "用温柔且故作轻松的语气说，略带疲惫感",
                "ref_emotion_category": "tender",
                "fish_audio_tags": "[温柔][轻松]慌什么，肩上的伤早就愈合了，听我的声音，这不是好得很",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            "场景：日常办公室，秦彻处理文件\n"
            "台词：这份报告的数据有误，重新核实一遍\n"
            "近期情感历史：[calm(0.4)]"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "emotion": "matter_of_fact",
                "intensity": 0.3,
                "pace": "normal",
                "style": "用平静简洁的语气说",
                "ref_emotion_category": "calm",
                "fish_audio_tags": "[平静]这份报告的数据有误，重新核实一遍",
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            "场景：深夜，秦彻打电话给女主\n"
            "台词：有个很久没见的人想在N109约会，除了这个，还能是因为什么\n"
            "近期情感历史：[calm(0.5)]"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "emotion": "teasing_flirty",
                "intensity": 0.5,
                "pace": "normal",
                "style": "用带有调侃意味的轻松口吻说",
                "ref_emotion_category": "playful",
                "fish_audio_tags": "[调侃][轻松]有个很久没见的人想在N109约会，除了这个，还能是因为什么",
            },
            ensure_ascii=False,
        ),
    },
]

MAX_RETRIES = 3


class EmotionAnalyzer:
    """Analyze a dialogue line in context and produce an EmotionAnnotation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or os.environ.get("LLM_BASE_URL"),
        )

    def analyze(
        self,
        line: str,
        scene_description: str,
        emotion_history: list[EmotionState],
        character_context: list[str],
    ) -> EmotionAnnotation:
        """Call LLM to produce structured emotion annotation for *line*."""

        history_str = ", ".join(
            f"{s.emotion}({s.intensity})" for s in emotion_history
        ) or "（无）"

        context_str = "\n".join(
            f"- {ctx}" for ctx in character_context
        ) or "（无额外角色上下文）"

        user_message = (
            f"场景：{scene_description}\n"
            f"台词：{line}\n"
            f"近期情感历史：[{history_str}]\n"
            f"角色上下文：\n{context_str}"
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_message},
        ]

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                return EmotionAnnotation(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                continue
            except (APITimeoutError, APIError) as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue

        raise ValueError(
            f"Failed after {MAX_RETRIES} retries: {last_error}"
        )
