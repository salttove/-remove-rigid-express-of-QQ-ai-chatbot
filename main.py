"""AstrBot AI-yubagu filter.

Two layers:
  - before the model: a short expression guard in the system prompt;
  - after the model: remove DS-style yubagu / AI filler from outgoing text.

The heavy lifting lives in filter_core.py (no AstrBot imports).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .filter_core import YubaguFilter, load_lexicon
except ImportError:
    from filter_core import YubaguFilter, load_lexicon

logger = logging.getLogger("astrbot")

DEFAULT_GUARD = (
    "【表达禁忌】直接说事，别用鱼八股：不要只回一个字/两个字（好、行、行吧、你定、我不理）；"
    "不要认错官腔（这条我认、我记下了、我记住了）；"
    "不要先分点再总结再拔高（讲的其实是、本质上、说到底）；"
    "不要把工具状态或备选回复发出来（已记录、记进待办、以下是几个版本）；"
    "不要固定句式（不是……而是……、总的来说、值得一提的是、某种程度上）；"
    "不要给对方的发言打分，不要宣告“我在/我守着”，不要问“你认吗/行不行/你说呢”。"
    "短句，口语，别每句都收尾。"
)


@register(
    "astrbot_plugin_ai_yubagu_filter",
    "local",
    "AI 八股过滤器",
    "1.0.0",
    "拦截 DS/鱼八股和 AI 套话，清理模型输出里的固定口癖。",
)
class AIYubaguFilterPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.cfg = config
        lexicon = load_lexicon(Path(__file__).with_name("lexicon.json"))
        self._filter = self._build_filter(lexicon)
        self.stats: dict[str, Any] = {
            "checked": 0,
            "changed": 0,
            "hits": 0,
            "last": [],
        }

    def _build_filter(self, lexicon: dict[str, Any]) -> YubaguFilter:
        return YubaguFilter(
            lexicon,
            enable_ds=bool(self.cfg.get("enable_ds_yubagu", True)),
            enable_transitions=bool(self.cfg.get("enable_ai_transitions", True)),
            enable_short_cut=bool(self.cfg.get("enable_short_cut", True)),
            enable_fiction=bool(self.cfg.get("enable_fiction_cliches", False)),
            enable_rhetoric=bool(self.cfg.get("enable_rhetoric_patterns", False)),
            fallback=str(self.cfg.get("fallback_reply") or "嗯，我在听。"),
            extra_phrases=list(self.cfg.get("extra_phrases") or []),
        )

    # -- hooks -----------------------------------------------------------

    @filter.on_llm_request(priority=-900)
    async def on_llm_request(self, event: AstrMessageEvent, req: Any):
        if not self.cfg.get("enabled", True):
            return
        if not self.cfg.get("prompt_guard_enabled", True):
            return
        guard = str(self.cfg.get("guard_prompt") or DEFAULT_GUARD).strip()
        if guard:
            req.system_prompt = (req.system_prompt or "") + "\n\n" + guard

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: Any):
        if not self.cfg.get("enabled", True):
            return
        if not self.cfg.get("clean_llm_response", True):
            return
        text = getattr(resp, "completion_text", None)
        if isinstance(text, str) and text:
            cleaned, hits = self._clean(text)
            if hits:
                resp.completion_text = cleaned

    @filter.on_decorating_result(priority=-100)
    async def on_decorating_result(self, event: AstrMessageEvent):
        if not self.cfg.get("enabled", True):
            return
        if not self.cfg.get("clean_result_chain", True):
            return
        result = self._get_result(event)
        if result is None:
            return
        if not self._is_llm_result(result):
            return
        chain = getattr(result, "chain", None)
        if not isinstance(chain, list) or not chain:
            return
        for component in chain:
            if type(component).__name__ != "Plain":
                continue
            text = getattr(component, "text", None)
            if not isinstance(text, str) or not text:
                continue
            cleaned, hits = self._clean(text)
            if hits:
                component.text = cleaned

    # -- cleaning --------------------------------------------------------

    def _clean(self, text: str) -> tuple[str, list[dict[str, str]]]:
        self.stats["checked"] += 1
        cleaned, hits = self._filter.clean(text)
        if hits:
            self.stats["changed"] += 1
            self.stats["hits"] += len(hits)
            self.stats["last"] = hits[-5:]
            if self.cfg.get("log_hits", False):
                logger.info("[yubagu] hits=%s", hits[:3])
        return cleaned, hits

    @staticmethod
    def _get_result(event: AstrMessageEvent) -> Any:
        try:
            fn = getattr(event, "get_result", None)
            if callable(fn):
                return fn()
        except Exception:
            pass
        return getattr(event, "_result", None)

    @staticmethod
    def _is_llm_result(result: Any) -> bool:
        fn = getattr(result, "is_llm_result", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass
        typ = getattr(result, "result_content_type", None)
        name = str(getattr(typ, "name", typ) or "")
        if not name or name == "None":
            return True
        return "LLM" in name.upper()

    # -- commands --------------------------------------------------------

    @filter.command("八股", alias={"/八股", "yubagu", "/yubagu"})
    async def cmd_yubagu(self, event: AstrMessageEvent):
        text = (event.message_str or "").strip()
        text = re.sub(r"^[/!]?(?:八股|yubagu)\b", "", text, flags=re.I).strip()

        if not text or text in {"状态", "status"}:
            last = "、".join(f"{h['phrase']}({h['category']})" for h in self.stats["last"])
            yield event.plain_result(
                f"AI 八股过滤器：{'开' if self.cfg.get('enabled', True) else '关'} | "
                f"检查 {self.stats['checked']} 条 | 改动 {self.stats['changed']} 条 | "
                f"命中 {self.stats['hits']} 处"
                + (f"\n最近：{last}" if last else "")
            )
            return

        if text.startswith("测试"):
            sample = text[2:].strip()
            if not sample:
                yield event.plain_result("用法：/八股 测试 <一段文本>")
                return
            cleaned, hits = self._filter.clean(sample)
            hit_text = "、".join(f"{h['phrase']}({h['category']})" for h in hits) or "无"
            yield event.plain_result(
                f"命中：{hit_text}\n清理后：{cleaned[:800]}"
            )
            return

        if text in {"开", "关", "on", "off"}:
            if not event.is_admin():
                yield event.plain_result("只有管理员可以开关。")
                return
            enabled = text in {"开", "on"}
            self.cfg["enabled"] = enabled
            yield event.plain_result(f"AI 八股过滤器已{'开启' if enabled else '关闭'}。")
            return

        yield event.plain_result(
            "用法：/八股 状态 | /八股 测试 <文本> | /八股 开 | /八股 关"
        )
