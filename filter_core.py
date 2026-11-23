"""Pure text filtering for the AstrBot AI-yubagu plugin.

No AstrBot imports here, so this module can be unit-tested on its own.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DEFAULT_FALLBACK = "嗯，我在听。"

_SENTENCE_RE = re.compile(r"(?<=[。！？!?…\n])")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_URL_RE = re.compile(r"https?://[^\s]+")
_PLACEHOLDER_RE = re.compile("\u0000(\\d+)\u0000")
_PUNCT_EDGE = "，,、；;：:。！？!?… \t\r\n"


def normalize_for_compare(text: str) -> str:
    return re.sub(r"[\s。！？!?…，,、；;：:]+", "", text or "")


class YubaguFilter:
    def __init__(
        self,
        lexicon: dict[str, Any],
        *,
        enable_ds: bool = True,
        enable_transitions: bool = True,
        enable_short_cut: bool = True,
        enable_fiction: bool = False,
        enable_rhetoric: bool = False,
        fallback: str = DEFAULT_FALLBACK,
        extra_phrases: list[str] | None = None,
    ) -> None:
        self.hard = list(lexicon.get("hard") or [])
        self.soft = list(lexicon.get("soft") or [])
        self.short_cut = list(lexicon.get("short_cut") or [])
        self.opener_replace = list(lexicon.get("opener_replace") or [])
        self.fiction = list(lexicon.get("fiction") or [])
        for phrase in extra_phrases or []:
            phrase = str(phrase).strip()
            if phrase:
                self.hard.append(phrase)
        self.enable_ds = enable_ds
        self.enable_transitions = enable_transitions
        self.enable_short_cut = enable_short_cut
        self.enable_fiction = enable_fiction
        self.enable_rhetoric = enable_rhetoric
        self.fallback = fallback or DEFAULT_FALLBACK

        transition = str(lexicon.get("transition_prefix") or "")
        filler = str(lexicon.get("filler") or "")
        era = str(lexicon.get("era_prefix") or "")
        let_us = str(lexicon.get("let_us") or "")
        not_but = str(lexicon.get("not_but") or "")

        self.ai_self_re = re.compile(
            r"^(?:作为一个?(?:AI|人工智能|语言模型))[，,、：:\s]*",
            re.I,
        )
        self.transition_re = (
            re.compile(rf"^(?:{transition})[，,、：:\s]*", re.I)
            if transition
            else None
        )
        self.filler_re = (
            re.compile(rf"^(?:{filler})[^。！？\n]*[。！？]?$", re.I)
            if filler
            else None
        )
        self.era_re = re.compile(rf"^(?:{era})[，,、：:\s]*", re.I) if era else None
        self.let_us_re = re.compile(rf"^(?:{let_us})[。！？]?$", re.I) if let_us else None
        self.not_but_re = re.compile(not_but, re.I) if not_but else None
        self.soft_set = {normalize_for_compare(x) for x in self.soft}
        self.short_cut_set = {normalize_for_compare(x) for x in self.short_cut}

    def clean(self, text: str) -> tuple[str, list[dict[str, str]]]:
        if not text or not text.strip():
            return text, []
        protected, stash = _protect(text)
        sentences = [s for s in _SENTENCE_RE.split(protected) if s != ""]
        out: list[str] = []
        hits: list[dict[str, str]] = []
        for sentence in sentences:
            cleaned, sentence_hits = self._clean_sentence(sentence)
            hits.extend(sentence_hits)
            if cleaned.strip():
                out.append(cleaned)
        result = "".join(out)
        result = _restore(result, stash)
        result = re.sub(r"\n{3,}", "\n\n", result).strip()
        if not result:
            return self.fallback, hits
        return result, hits

    def _clean_sentence(self, sentence: str) -> tuple[str, list[dict[str, str]]]:
        hits: list[dict[str, str]] = []
        s = sentence.strip()
        if not s:
            return "", hits

        if self.enable_short_cut:
            for opener in self.opener_replace:
                m = re.match(rf"^{re.escape(opener)}[，,、：:\s]+", s)
                if m:
                    s = s[m.end() :].strip()
                    hits.append({"category": "opener_replace", "phrase": opener})
                    if not s:
                        return "", hits
                    break

        if self.enable_transitions:
            m = self.ai_self_re.match(s)
            if m and m.end() > 0:
                s = s[m.end() :].lstrip()
            while True:
                changed = False
                if self.era_re:
                    m = self.era_re.match(s)
                    if m and m.end() > 0:
                        s = s[m.end() :].lstrip()
                        changed = True
                if self.transition_re:
                    m = self.transition_re.match(s)
                    if m and m.end() > 0:
                        s = s[m.end() :].lstrip()
                        changed = True
                if not changed:
                    break
            if not s:
                return "", hits
            if self.filler_re and self.filler_re.fullmatch(s):
                return "", [{"category": "filler", "phrase": s[:40]}]
            if self.let_us_re and self.let_us_re.fullmatch(s):
                return "", [{"category": "let_us", "phrase": s[:40]}]

        if self.enable_rhetoric and self.not_but_re and self.not_but_re.search(s):
            return "", [{"category": "not_but", "phrase": s[:60]}]

        if self.enable_ds:
            for phrase in self.hard:
                if phrase and phrase in s:
                    s = _remove_phrase(s, phrase)
                    hits.append({"category": "ds_hard", "phrase": phrase})
                    if not s.strip():
                        return "", hits
                    break

        if self.enable_fiction and self.fiction:
            for phrase in self.fiction:
                if phrase and phrase in s:
                    hits.append({"category": "fiction", "phrase": phrase})
                    return "", hits

        normalized = normalize_for_compare(s)
        if self.enable_short_cut and normalized and normalized in self.short_cut_set:
            hits.append({"category": "short_cut", "phrase": s[:20]})
            return "", hits

        if self.enable_ds:
            if normalized and normalized in self.soft_set:
                hits.append({"category": "ds_soft", "phrase": s[:40]})
                return "", hits

        return s.strip(), hits


def _remove_phrase(sentence: str, phrase: str) -> str:
    """Remove one phrase; keep the rest when it is a leading/trailing clause."""
    idx = sentence.find(phrase)
    if idx < 0:
        return sentence
    before = sentence[:idx]
    after = sentence[idx + len(phrase) :]
    if not before.strip():
        return after.lstrip(_PUNCT_EDGE)
    if not after.strip():
        return before.rstrip(_PUNCT_EDGE)
    return ""


def _protect(text: str) -> tuple[str, list[str]]:
    stash: list[str] = []

    def repl(match: re.Match[str]) -> str:
        stash.append(match.group(0))
        return f"\u0000{len(stash) - 1}\u0000"

    text = _CODE_FENCE_RE.sub(repl, text)
    text = _INLINE_CODE_RE.sub(repl, text)
    text = _URL_RE.sub(repl, text)
    return text, stash


def _restore(text: str, stash: list[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return stash[index] if 0 <= index < len(stash) else ""

    return _PLACEHOLDER_RE.sub(repl, text)


def load_lexicon(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}
