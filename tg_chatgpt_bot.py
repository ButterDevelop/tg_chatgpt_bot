from __future__ import annotations
import base64
import json
import logging
import os
import sqlite3
import textwrap
import atexit
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from itertools import cycle
from urllib.parse import urlparse, parse_qs, unquote
from uuid import uuid4
from functools import partial

import re
import io
import asyncio
import httpx
import requests
import threading
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from trafilatura import extract as trafi_extract

from contextlib import asynccontextmanager
from telegram.constants import ChatAction
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, LabeledPrice
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    PreCheckoutQueryHandler,
    ContextTypes,
    filters,
)

from openai import OpenAI

# ==============================
# Configuration
# ==============================
load_dotenv()

# Regular expressions for code/math protection when converting Markdown to Telegram
_CODE_BLOCK_RE = re.compile(r"```(?:.|\n)*?```", re.DOTALL)  # triple backticks ```...```
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")                   # inline `code`
_MATH_BLOCK_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)     # LaTeX blocks of the form \[ ... \]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")
DEFAULT_ONE_SHOT = os.getenv("DEFAULT_ONE_SHOT", "false").lower() in {"1", "true", "yes"}

ALLOWLIST = set(int(x.strip()) for x in os.getenv("ALLOWLIST", "").split(",") if x.strip())
ADMIN_IDS = set(int(x.strip()) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip())
if not ALLOWLIST and ADMIN_IDS:
    ALLOWLIST = set(ADMIN_IDS)

# Allowlist restriction is optional (by default, the bot is accessible to everyone)
RESTRICT_TO_ALLOWLIST = os.getenv("RESTRICT_TO_ALLOWLIST", "false").lower() in {"1", "true", "yes"}

# How many stars does one message cost for regular users?
MESSAGE_PRICE_STARS = int(os.getenv("MESSAGE_PRICE_STARS", "2"))

# Available top-up packages
TOPUP_OPTIONS = [10, 20, 50, 100, 200, 500, 1000]

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_KEYS_RAW = os.getenv("TAVILY_API_KEYS", "")
TAVILY_API_KEYS = [k.strip() for k in TAVILY_API_KEYS_RAW.split(",") if k.strip()]
if not TAVILY_API_KEYS and TAVILY_API_KEY:
    TAVILY_API_KEYS = [TAVILY_API_KEY]

_tavily_lock = threading.Lock()
_tavily_cycle = cycle(TAVILY_API_KEYS) if TAVILY_API_KEYS else None

def _next_tavily_key() -> Optional[str]:
    if _tavily_cycle is None:
        return None
    with _tavily_lock:
        return next(_tavily_cycle)

HTTP_USER_AGENT = os.getenv("HTTP_USER_AGENT", "Mozilla/5.0 (compatible; ChatGPT-TG-Bot/1.0)")

def get_lang(update: Update) -> str:
    code = ""
    if update.effective_user and update.effective_user.language_code:
        code = update.effective_user.language_code.lower()

    # Everyone who knows russian is considered russian language
    if code.startswith(("ru", "uk", "be")):
        return "ru"
    return "en"

# Optional proxy for OpenAI + reuse for requests
OPENAI_PROXY = os.getenv("OPENAI_PROXY", "").strip()
OPENAI_HTTP_PROXY = os.getenv("HTTP_PROXY", os.getenv("http_proxy", "")).strip()
OPENAI_HTTPS_PROXY = os.getenv("HTTPS_PROXY", os.getenv("https_proxy", "")).strip()

DB_PATH = os.getenv("DB_PATH", "chatgpt_tg.db")
MAX_HISTORY_MSGS = int(os.getenv("MAX_HISTORY_MSGS", "16"))
MAX_TOOL_ITER = int(os.getenv("MAX_TOOL_ITER", "3"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("tg-bot")

CONTEXT_CHAR_BUDGET  = int(os.getenv("CONTEXT_CHAR_BUDGET", "120000"))
MAX_TEXT_PARTS_TOTAL = int(os.getenv("MAX_TEXT_PARTS_TOTAL", "18"))
MAX_TEXT_PART_LEN    = int(os.getenv("MAX_TEXT_PART_LEN", "8000"))

ALBUM_GROUP_DELAY = float(os.getenv("ALBUM_GROUP_DELAY", "0.7"))
TEXT_ATTACH_DELAY = float(os.getenv("TEXT_ATTACH_DELAY", "1.2"))
LOOSE_GROUP_DELAY = float(os.getenv("LOOSE_GROUP_DELAY", "0.8"))

REQUESTS_TIMEOUT = float(os.getenv("REQUESTS_TIMEOUT", "30"))

MESSAGES = {
    "thinking": {
        "en": "🤖 Thinking…",
        "ru": "🤖 Думаю…",
    },
    "error_generic": {
        "en": "⚠️ Error: {error}",
        "ru": "⚠️ Ошибка: {error}",
    },
    "export_empty": {
        "en": "Nothing to export yet - the history is empty.",
        "ru": "Пока нечего экспортировать - история пуста.",
    },
    "export_pdf_caption": {
        "en": "Conversation exported as PDF.",
        "ru": "Экспорт диалога в PDF.",
    },
    "export_fail": {
        "en": "Failed to generate PDF: {error}",
        "ru": "Не удалось собрать PDF: {error}",
    },
    "long_reply_pdf_caption": {
        "en": "The reply is long - sending it as PDF.",
        "ru": "Ответ длинный - отправляю как PDF.",
    },
    "access_denied": {
        "en": "Access denied. Contact the bot owner.",
        "ru": "Доступ запрещён. Свяжитесь с владельцем бота.",
    },
    "first_message_free": {
        "en": (
            "This is your free message 🎁\n"
            "After that, each message will subtract {price}⭐ from your balance.\n"
            "Commands: /balance - balance, /topup - top-up."
        ),
        "ru": (
            "Это ваше бесплатное сообщение 🎁\n"
            "Дальше каждое сообщение будет списывать {price}⭐ с баланса.\n"
            "Команды: /balance - баланс, /topup - пополнение."
        ),
    },
    "choose_model": {
        "en": "Choose a model or use /setmodel <id>:",
        "ru": "Выберите модель или используйте /setmodel <id>:",
    },
    "model_updated": {
        "en": "Model updated",
        "ru": "Модель обновлена",
    },
    "model_set": {
        "en": "Model set to `{model}`",
        "ru": "Модель установлена: `{model}`",
    },
    "setmodel_usage": {
        "en": "Usage: /setmodel <model-id>",
        "ru": "Использование: /setmodel <model-id>",
    },
    "dialog_mode_one_shot": {
        "en": "Dialog mode is now: `one-shot (new dialog per message)`",
        "ru": "Режим диалога теперь: `one-shot (новый диалог на каждое сообщение)`",
    },
    "dialog_mode_continuous": {
        "en": "Dialog mode is now: `continuous (keeps history)`",
        "ru": "Режим диалога теперь: `continuous (помню историю)`",
    },
    "history_cleared": {
        "en": "History cleared.",
        "ru": "История очищена.",
    },
    "balance_admin_info": {
        "en": (
            "You are an administrator; stars are not charged for you.\n"
            "Regular user price: {price}⭐ per message."
        ),
        "ru": (
            "Вы администратор, с вас звёзды не списываются.\n"
            "Тариф для обычных пользователей: {price}⭐ за сообщение."
        ),
    },
    "balance_info": {
        "en": "Your balance: {balance}⭐.\nPrice per message: {price}⭐.",
        "ru": "Ваш баланс: {balance}⭐.\nСтоимость одного сообщения: {price}⭐.",
    },
    "balance_free_left": {
        "en": "You still have 1 free message.",
        "ru": "У вас ещё есть 1 бесплатное сообщение.",
    },
    "topup_choose": {
        "en": "Choose an amount to top up your balance:",
        "ru": "Выберите, на сколько пополнить баланс:",
    },
    "topup_invalid_amount": {
        "en": "Invalid amount",
        "ru": "Неверная сумма",
    },
    "topup_success": {
        "en": (
            "Top-up successful 🎉\n"
            "{stars}⭐ have been added to your balance.\n"
            "Check your balance: /balance"
        ),
        "ru": (
            "Пополнение успешно 🎉\n"
            "На ваш баланс зачислено {stars}⭐.\n"
            "Посмотреть баланс: /balance"
        ),
    },
}

def tr(update: Optional[Update], key: str, **kwargs) -> str:
    lang = get_lang(update) if update else "en"
    variants = MESSAGES.get(key, {})
    template = variants.get(lang) or variants.get("en") or ""
    return template.format(**kwargs)

# ==============================
# DB
# ==============================
SCHEMA_SQL = r"""
CREATE TABLE IF NOT EXISTS chat_settings (
  chat_id INTEGER PRIMARY KEY,
  model TEXT NOT NULL,
  one_shot INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chat_id INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
  content TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users (
  user_id INTEGER PRIMARY KEY,
  balance_stars INTEGER NOT NULL DEFAULT 0,
  free_message_used INTEGER NOT NULL DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS payments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  stars_amount INTEGER NOT NULL,
  telegram_payment_charge_id TEXT NOT NULL,
  provider_payment_charge_id TEXT,
  invoice_payload TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def db_init():
    with db() as con:
        con.executescript(SCHEMA_SQL)

# ==============================
# OpenAI client (with optional proxy)
# ==============================
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY is required")

_httpx_client: Optional[httpx.Client] = None
proxies: Any = None
if OPENAI_PROXY:
    proxies = OPENAI_PROXY
elif OPENAI_HTTP_PROXY or OPENAI_HTTPS_PROXY:
    proxies = {}
    if OPENAI_HTTP_PROXY:
        proxies["http://"] = OPENAI_HTTP_PROXY
    if OPENAI_HTTPS_PROXY:
        proxies["https://"] = OPENAI_HTTPS_PROXY

if proxies:
    # Try modern and legacy kw names for httpx
    try:
        _httpx_client = httpx.Client(proxies=proxies, timeout=60.0)
    except TypeError:
        _httpx_client = httpx.Client(proxy=proxies, timeout=60.0)  # older httpx
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=_httpx_client)
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

@atexit.register
def _close_httpx():
    try:
        if _httpx_client is not None:
            _httpx_client.close()
    except Exception:
        pass

# ==============================
# requests.Session with the SAME proxy map (for Tavily/DDG/pages)
# ==============================
_requests_session: Optional[requests.Session] = None
_requests_proxies: Optional[Dict[str, str]] = None


def _redact_proxy(url: str) -> str:
    try:
        p = urlparse(url)
        auth = "***:***@" if (p.username or p.password) else ""
        host = p.hostname or ""
        port = f":{p.port}" if p.port else ""
        return f"{p.scheme}://{auth}{host}{port}"
    except Exception:
        return "<redacted>"


def _redact_proxy_map(m: Dict[str, str]) -> Dict[str, str]:
    return {k: _redact_proxy(v) for k, v in m.items()}


def _build_requests_session():
    global _requests_session, _requests_proxies
    s = requests.Session()
    s.headers.update({"User-Agent": HTTP_USER_AGENT})

    # Convert httpx-style proxies into requests-style
    req_map: Optional[Dict[str, str]] = None
    if proxies:
        if isinstance(proxies, str):
            req_map = {"http": proxies, "https": proxies}
        elif isinstance(proxies, dict):
            req_map = {}
            for k, v in proxies.items():
                key = k.split(":", 1)[0] if "://" in k else k
                if key in {"http", "http", "https"}:
                    req_map[key] = v
                elif key in {"http", "https"}:  # already ok
                    req_map[key] = v
        # Apply
        if req_map:
            s.proxies.update(req_map)
            log.info("HTTP proxies enabled for requests (Tavily/web): %s", _redact_proxy_map(req_map))
    _requests_session = s
    _requests_proxies = req_map


_build_requests_session()

# Small helpers to use the shared session

def _r_get(url: str, **kwargs):
    kwargs.setdefault("timeout", REQUESTS_TIMEOUT)
    return _requests_session.get(url, **kwargs) if _requests_session else requests.get(url, **kwargs)


def _r_post(url: str, **kwargs):
    kwargs.setdefault("timeout", REQUESTS_TIMEOUT)
    return _requests_session.post(url, **kwargs) if _requests_session else requests.post(url, **kwargs)

# ==============================
# Browsing tools
# ==============================
class SearchResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class WebSearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)

class OpenUrlResponse(BaseModel):
    url: str
    title: Optional[str] = None
    text: str


def _unwrap_duckduckgo(url: str) -> str:
    try:
        if url.startswith("/"):
            url = "https://duckduckgo.com" + url
        p = urlparse(url)
        if p.netloc.endswith("duckduckgo.com") and p.path.startswith("/l/"):
            qs = parse_qs(p.query)
            target = qs.get("uddg", [""])[0]
            return unquote(target) or url
        return url
    except Exception:
        return url


def tavily_search(query: str, max_results: int = 5) -> WebSearchResponse:
    # If no keys -> DuckDuckGo immediately
    if not TAVILY_API_KEYS:
        log.warning("TAVILY_API_KEYS not set, attempting DuckDuckGo fallback")
        return _duckduckgo_search(query, max_results)

    attempts = min(len(TAVILY_API_KEYS), max(1, len(TAVILY_API_KEYS)))
    last_errors: List[str] = []
    for _ in range(attempts):
        api_key = _next_tavily_key()
        if not api_key:
            break
        try:
            r = _r_post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": HTTP_USER_AGENT,
                },
                json={
                    "query": query,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                    "search_depth": "advanced",
                    "include_domains": [],
                    "exclude_domains": [],
                },
            )

            if r.status_code in (401, 403, 429, 500, 502, 503, 504):
                last_errors.append(f"{r.status_code}:{r.text[:200]}")
                continue
            r.raise_for_status()
            data = r.json()
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                ))
            return WebSearchResponse(results=results)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "HTTPError"
            body = e.response.text[:200] if getattr(e, "response", None) is not None else str(e)
            last_errors.append(f"{status}:{body}")
            continue
        except Exception as e:
            last_errors.append(str(e))
            continue

    log.warning("Tavily: all keys failed (%s). Falling back to DuckDuckGo.", "; ".join(last_errors))
    return _duckduckgo_search(query, max_results)


def _duckduckgo_search(query: str, max_results: int) -> WebSearchResponse:
    r = _r_get(
        "https://duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": HTTP_USER_AGENT},
        allow_redirects=True,
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    out: List[SearchResult] = []

    # Try classic selector first
    links = soup.select("a.result__a")
    if not links:
        # Fallback selectors/heuristics
        links = [a for a in soup.select("a[href]") if "/l/?" in a.get("href", "")]
    for a in links[:max_results]:
        title = a.get_text(strip=True)
        url = _unwrap_duckduckgo(a.get("href", ""))
        if title and url:
            out.append(SearchResult(title=title, url=url))
    return WebSearchResponse(results=out)


def fetch_and_extract(url: str, max_chars: int = 20_000) -> OpenUrlResponse:
    r = _r_get(url, headers={"User-Agent": HTTP_USER_AGENT})
    r.raise_for_status()
    html = r.text
    text = trafi_extract(html, include_tables=False, include_links=False) or ""
    title = None
    try:
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        title = t.get_text(strip=True)[:200] if t else None
    except Exception:
        pass
    text = (text or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return OpenUrlResponse(url=url, title=title, text=text or "(no extractable text)")

# ==============================
# Model tool schema (for function calling)
# ==============================
TOOLS = [
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for recent information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "type": "function",
        "name": "open_url",
        "description": "Open a URL and extract main text for the model to read.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
]

# ==============================
# Helpers: allowlist, settings, history
# ==============================
@dataclass
class ChatSettings:
    model: str
    one_shot: bool


def db_get_history(chat_id: int) -> List[Dict[str, Any]]:
    with db() as con:
        cur = con.execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id ASC",
            (chat_id,),
        )
        return [{"role": r["role"], "content": r["content"]} for r in cur.fetchall()]


def ensure_chat_settings(chat_id: int) -> ChatSettings:
    with db() as con:
        row = con.execute("SELECT model, one_shot FROM chat_settings WHERE chat_id=\n?", (chat_id,)).fetchone()
        if row:
            return ChatSettings(model=row["model"], one_shot=bool(row["one_shot"]))
        con.execute(
            "INSERT INTO chat_settings(chat_id, model, one_shot) VALUES (?,?,?)",
            (chat_id, DEFAULT_MODEL, int(DEFAULT_ONE_SHOT)),
        )
        return ChatSettings(model=DEFAULT_MODEL, one_shot=DEFAULT_ONE_SHOT)


def set_model(chat_id: int, model: str):
    with db() as con:
        con.execute("UPDATE chat_settings SET model=? WHERE chat_id=\n?", (model, chat_id))


def toggle_mode(chat_id: int) -> bool:
    with db() as con:
        row = con.execute("SELECT one_shot FROM chat_settings WHERE chat_id=\n?", (chat_id,)).fetchone()
        current = bool(row["one_shot"]) if row else DEFAULT_ONE_SHOT
        new_val = 0 if current else 1
        con.execute("UPDATE chat_settings SET one_shot=? WHERE chat_id=\n?", (new_val, chat_id))
        return bool(new_val)


def clear_history(chat_id: int):
    with db() as con:
        con.execute("DELETE FROM messages WHERE chat_id=\n?",(chat_id,))


def add_message(chat_id: int, role: str, content: str):
    with db() as con:
        con.execute(
            "INSERT INTO messages(chat_id, role, content) VALUES (?,?,?)",
            (chat_id, role, content),
        )
        rows = con.execute(
            "SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, MAX_HISTORY_MSGS * 2),
        ).fetchall()
        if rows:
            last_keep_id = rows[-1]["id"]
            con.execute("DELETE FROM messages WHERE chat_id=? AND id<?", (chat_id, last_keep_id))

# ==============================
# OpenAI call with tool loop (supports tool_use + function_call)
# ==============================

def to_data_uri_image(jpeg_bytes: bytes) -> str:
    b64 = base64.b64encode(jpeg_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


def apply_entities_as_markdown(text: str, entities: Optional[Sequence]) -> str:
    """
    Converts Telegram + entities text to simple Markdown,
    which then goes to ChatGPT.

    Supports:
    - bold        -> **text**
    - italic      -> *text*
    - code        -> `text`
    - pre         -> ```text```
    - text_link   -> [text](url)
    Other types are simply ignored (leave the text as is).
    """

    if not text:
        return ""

    if not entities:
        return text

    # We work in descending order of offset so as not to shift the indices for more
    # unprocessed entities.
    # In entities from python-telegram-bot offset/length are already normal indices.
    sorted_entities = sorted(entities, key=lambda e: getattr(e, "offset", 0), reverse=True)

    result = text

    for ent in sorted_entities:
        try:
            offset = int(getattr(ent, "offset", 0))
            length = int(getattr(ent, "length", 0))
        except (TypeError, ValueError):
            continue

        if length <= 0:
            continue

        # Protection against going beyond the boundaries
        if offset < 0 or offset >= len(result):
            continue

        end = offset + length
        if end > len(result):
            end = len(result)

        segment = result[offset:end]
        etype = getattr(ent, "type", "") or ""

        # Simple formatting for Markdown
        if etype == "bold":
            formatted = f"**{segment}**"
        elif etype == "italic":
            formatted = f"*{segment}*"
        elif etype == "code":
            formatted = f"`{segment}`"
        elif etype == "pre":
            # If multiline – take triple quotes
            if "\n" in segment:
                formatted = f"```\n{segment}\n```"
            else:
                formatted = f"```{segment}```"
        elif etype == "text_link":
            url = getattr(ent, "url", None)
            if url:
                formatted = f"[{segment}]({url})"
            else:
                formatted = segment
        else:
            # Unknown type – leave as is
            formatted = segment

        # Insert the formatted segment back
        result = result[:offset] + formatted + result[end:]

    return result


def build_user_content(text: Optional[str], image_bytes_list: List[bytes]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"type": "input_text", "text": text})
    for b in image_bytes_list:
        parts.append({"type": "input_image", "image_url": to_data_uri_image(b)})
    return parts or [{"type": "input_text", "text": "(no text)"}]


def _history_block(role: str, text: str) -> Dict[str, Any]:
    return {"type": "output_text" if role == "assistant" else "input_text", "text": text}


def _flatten_response(resp) -> List[Dict[str, Any]]:
    raw = resp.model_dump()
    blocks: List[Dict[str, Any]] = []
    for item in raw.get("output", []):
        t = item.get("type")
        if t == "message":
            for c in item.get("content", []) or []:
                blocks.append(c)
        else:
            blocks.append(item)
    return blocks


def _estimate_chars(msgs: List[Dict[str, Any]]) -> int:
    total = 0
    for m in msgs:
        for c in m.get("content", []):
            t = c.get("type")
            if t in ("input_text", "output_text", "refusal", "summary_text"):
                total += len(c.get("text", ""))
    return total


def _prune_history_to_budget(msgs: List[Dict[str, Any]], hist_start_idx: int, budget: int) -> List[Dict[str, Any]]:
    pruned = msgs[:]
    while _estimate_chars(pruned) > budget:
        if hist_start_idx >= len(pruned) - 1:
            break
        del pruned[hist_start_idx]
    return pruned


def run_model_with_tools(
    model: str,
    system_prompt: str,
    history: List[Dict[str, Any]],
    user_parts: List[Dict[str, Any]],
):
    """
    The only point that goes to OpenAI and returns:
    - reply_text: text to send to the user
    - all_blocks: "raw" blocks of the answer (for the future, if you want to do something with them)
    """

    MAX_HISTORY_MSGS = globals().get("MAX_HISTORY_MSGS", 16)
    MAX_OUTPUT_TOKENS = globals().get("MAX_OUTPUT_TOKENS", 1200)
    CONTEXT_CHAR_BUDGET = globals().get("CONTEXT_CHAR_BUDGET", 120_000)

    messages: List[Dict[str, Any]] = []

    # 1) system prompt
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [
                {"type": "input_text", "text": system_prompt},
            ],
        })

    # 2) history (without the current message - you're passing it as is)
    # trim on the safe side
    trimmed_history = history[-(MAX_HISTORY_MSGS * 2):]

    for h in trimmed_history:
        role = h.get("role", "user")
        text = h.get("content", "") or ""
        if not isinstance(text, str):
            text = str(text)

        if role not in ("user", "assistant"):
            role = "user"

        messages.append({
            "role": role,
            "content": [
                _history_block(role, text),
            ],
        })

    # index from which the exact history starts (for CONTEXT_CHAR_BUDGET trimming)
    hist_start_idx = 1 if messages and messages[0].get("role") == "system" else 0

    # 3) current user message (text + images + files)
    messages.append({
        "role": "user",
        "content": user_parts,
    })

    # 4) trim history by budget to avoid context limit
    messages = _prune_history_to_budget(messages, hist_start_idx, CONTEXT_CHAR_BUDGET)

    # 5) call OpenAI Responses API
    # Important: tools are not passed now to avoid tool-calling loop.
    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        # If you want to enable your tools (web_search / open_url),
        # add here: tools=TOOLS and write the tool_call processing loop.
    )

    # 6) try to extract ready text
    reply_text = getattr(resp, "output_text", None)
    all_blocks: List[Dict[str, Any]] = []

    if reply_text is None:
        # if output_text is missing for some reason, we extract "raw" blocks ourselves
        try:
            all_blocks = _flatten_response(resp)
            parts: List[str] = []
            for b in all_blocks:
                t = b.get("type")
                if t in ("output_text", "input_text", "refusal", "summary_text"):
                    txt = b.get("text", "")
                    if txt:
                        parts.append(txt)
            reply_text = "\n".join(parts).strip()
        except Exception:
            reply_text = ""

    if not reply_text:
        reply_text = "(no response)"

    return reply_text, all_blocks


PROGRESS_MODE = os.getenv("PROGRESS_MODE", "delete").lower()  # "delete" | "edit"
PROGRESS_TEXT = os.getenv("PROGRESS_TEXT", "🤖 Думаю…")
PROGRESS_ANIMATE = os.getenv("PROGRESS_ANIMATE", "true").lower() in {"1","true","yes"}

class _StatusMsg:
    def __init__(self, msg):
        self.msg = msg
        self.delete_on_done = True

@asynccontextmanager
async def temp_status(update, context, text=PROGRESS_TEXT, animate=PROGRESS_ANIMATE):
    chat_id = update.effective_chat.id
    m = await update.effective_message.reply_text(text)
    stop = asyncio.Event()
    tasks = []

    # typing loop (Telegram holds the action for ~5 seconds)
    async def _typing():
        while not stop.is_set():
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                pass
            await asyncio.sleep(4.5)

    tasks.append(context.application.create_task(_typing()))

    # Dot animation / text rotation
    if animate:
        frames = [text, text.replace("…","."), text.replace("…",".."), text.replace("…","...")]
        async def _animate():
            i = 0
            while not stop.is_set():
                await asyncio.sleep(1.2)
                i = (i + 1) % len(frames)
                try:
                    await m.edit_text(frames[i])
                except Exception:
                    pass
        tasks.append(context.application.create_task(_animate()))

    holder = _StatusMsg(m)
    try:
        yield holder
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        if holder.delete_on_done:
            try:
                await m.delete()
            except Exception:
                pass

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        return

    prev = context.chat_data.get("_pending_text")
    if prev and not prev.get("consumed"):
        task = prev.get("task")
        if task and not task.done():
            task.cancel()
        old_update = prev["update"]
        context.chat_data.pop("_pending_text", None)
        await handle_message(old_update, context)

    # Apply markdown entities to pending text
    raw_txt = msg.text or msg.caption or ""
    ents = msg.entities or msg.caption_entities
    final_txt = apply_entities_as_markdown(raw_txt, ents)

    pend = {"update": update, "text": final_txt, "consumed": False}
    context.chat_data["_pending_text"] = pend

    async def _flush():
        try:
            await asyncio.sleep(TEXT_ATTACH_DELAY)
        except asyncio.CancelledError:
            return
        p = context.chat_data.get("_pending_text")
        if p is pend and not p.get("consumed"):
            context.chat_data.pop("_pending_text", None)
            await handle_message(update, context)

    task = context.application.create_task(_flush())
    pend["task"] = task


async def router_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.effective_message
    if msg is None:
        return

    chat_id = update.effective_chat.id
    albums = context.chat_data.setdefault("_albums", {})
    mgid = getattr(msg, "media_group_id", None)

    def _consume_pending_text() -> str:
        pend = context.chat_data.pop("_pending_text", None)
        if pend and not pend.get("consumed"):
            task = pend.get("task")
            if task and not task.done():
                task.cancel()
            pend["consumed"] = True
            return pend.get("text") or ""
        return ""

    if mgid:
        key = (chat_id, mgid)
        bucket = albums.get(key)
        if bucket is None:
            albums[key] = [msg]
            await asyncio.sleep(ALBUM_GROUP_DELAY)
            messages = albums.pop(key, [])
            extra_text = _consume_pending_text()
            await handle_media_group(messages, update, context, extra_text=extra_text)
        else:
            bucket.append(msg)
        return

    if msg.photo or msg.document:
        key = (chat_id, "_loose")
        bucket = albums.get(key)
        if bucket is None:
            albums[key] = [msg]
            await asyncio.sleep(LOOSE_GROUP_DELAY)
            messages = albums.pop(key, [])
            extra_text = _consume_pending_text()
            await handle_media_group(messages, update, context, extra_text=extra_text)
        else:
            bucket.append(msg)
        return


async def handle_media_group(messages, update: Update, context: ContextTypes.DEFAULT_TYPE, extra_text: Optional[str] = None):
    # 0) access + биллинг
    if not update.effective_user or not update.effective_chat:
        return

    uid = update.effective_user.id
    if not is_allowed_user(uid):
        if update.effective_message:
            await update.effective_message.reply_text(tr(update, "access_denied"))
        return

    should_charge = False
    if not is_admin(uid):
        allowed, should_charge = await ensure_balance_for_message(update, context)
        if not allowed:
            return

    text_lines: List[str] = []
    image_bytes_list: List[bytes] = []
    text_files: List[Tuple[str, str]] = []
    pdf_files:  List[Tuple[bytes, str]] = []
    other_files: List[str] = []
    photo_count = 0
    image_doc_names: List[str] = []

    if extra_text:
        text_lines.append(extra_text)

    for m in sorted(messages, key=lambda x: x.message_id):
        # Extract caption/text with entities
        if m.caption:
            cap = apply_entities_as_markdown(m.caption, m.caption_entities)
        elif m.text:
            cap = apply_entities_as_markdown(m.text, m.entities)
        else:
            cap = None

        if cap:
            text_lines.append(cap)
    
        if m.photo:
            ph = m.photo[-1]
            f = await ph.get_file()
            b = await f.download_as_bytearray()
            image_bytes_list.append(bytes(b))
            photo_count += 1
    
        if m.document:
            doc = m.document
            f = await doc.get_file()
            b = await f.download_as_bytearray()
            mime = doc.mime_type or ""
            fname = doc.file_name or "file.bin"
    
            if mime.startswith("image/"):
                image_bytes_list.append(bytes(b))
                if fname:
                    image_doc_names.append(fname)
    
            elif mime == "application/pdf" or fname.lower().endswith(".pdf"):
                pdf_files.append((bytes(b), fname))
    
            else:
                try:
                    text_data = bytes(b).decode("utf-8", errors="replace")
                    text_files.append((fname, text_data))
                except Exception:
                    other_files.append(fname)

    text = "\n\n".join(text_lines).strip() or None
    user_parts = build_user_content(text, image_bytes_list)

    MAX_TEXT_PARTS_TOTAL = globals().get("MAX_TEXT_PARTS_TOTAL", 18)

    def chunk_text(s: str, max_len: int = globals().get("MAX_TEXT_PART_LEN", 8000)) -> List[str]:
        return [s[i:i+max_len] for i in range(0, len(s), max_len)]

    used_parts = 0
    truncated_any = False
    for fname, text_data in text_files:
        if used_parts >= MAX_TEXT_PARTS_TOTAL:
            truncated_any = True
            break
        parts = chunk_text(text_data)
        parts = parts[:max(0, MAX_TEXT_PARTS_TOTAL - used_parts)]
        for i, part in enumerate(parts, start=1):
            header = f"[{fname} - part {i}/{len(parts)}]"
            user_parts.append({"type": "input_text", "text": f"{header}\n{part}"})
            used_parts += 1
    if truncated_any:
        user_parts.append({"type": "input_text", "text": "(Note: extra text parts were omitted.)"})

    from builtins import bytes as _b
    for fb, fname in pdf_files:
        try:
            bio = io.BytesIO(fb); bio.name = fname
            uploaded = client.files.create(file=bio, purpose="assistants")
            user_parts.append({"type": "input_file", "file_id": uploaded.id})
        except Exception as e:
            log.exception("OpenAI file upload failed for %s", fname)
            user_parts.append({"type": "input_text", "text": f"[PDF '{fname}' upload failed: {e}]"})
    if other_files:
        warn = ", ".join(other_files[:5])
        user_parts.append({"type": "input_text", "text": f"Note: unsupported file formats for context stuffing. Please resend as PDF or paste text: {warn}"})

    chat_id = update.effective_chat.id
    st = ensure_chat_settings(chat_id)
    if st.one_shot:
        with db() as con:
            con.execute("DELETE FROM messages WHERE chat_id=\n?", (chat_id,))

    hist = db_get_history(chat_id)

    attach_note = _compose_attachment_note(
        text_files=text_files,
        pdf_files=pdf_files,
        image_doc_names=image_doc_names,
        photo_count=photo_count,
        other_files=other_files,
    )
    if attach_note:
        user_parts.append({"type": "input_text", "text": attach_note})

    labels = []
    if image_bytes_list: labels.append("images")
    if text_files:       labels.append("text-files")
    if pdf_files:        labels.append("pdfs")
    if other_files:      labels.append("other-files")
    base_text = (text or "").strip()
    store_text = (base_text + attach_note).strip() or "[Attachment]"
    add_message(chat_id, "user", store_text)

    # "status" message + single model call
    async with temp_status(update, context, text=tr(update, "thinking")) as stmsg:
        try:
            reply_text, _msgs = run_model_with_tools(
                model=st.model,
                system_prompt=SYSTEM_PROMPT,
                history=hist,
                user_parts=user_parts,
            )
        except Exception as e:
            log.exception("OpenAI error")
            await update.effective_message.reply_text(tr(update, "error_generic", error=e))
            return

    if reply_text and reply_text != "(no response)":
        if not is_admin(uid) and should_charge:
            consume_message_credit(uid, MESSAGE_PRICE_STARS)
        add_message(chat_id, "assistant", reply_text[:20_000])

    # Use the album-collected 'text' as the user prompt to show in the PDF
    await send_text_or_pdf(update, reply_text or "", user_text_for_pdf=(text or attach_note or "[Attachment]"))


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS


def is_allowed_user(user_id: int) -> bool:
    # Admins can always access the bot
    if is_admin(user_id):
        return True
    # If strict allowlist is enabled:
    if RESTRICT_TO_ALLOWLIST:
        return user_id in ALLOWLIST
    # By default, the bot is accessible to everyone; access is regulated by balance
    return True


def ensure_user(user_id: int) -> sqlite3.Row:
    with db() as con:
        row = con.execute(
            "SELECT user_id, balance_stars, free_message_used FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()
        if row:
            return row
        con.execute(
            "INSERT INTO users(user_id, balance_stars, free_message_used) VALUES (?,?,?)",
            (user_id, 0, 0),
        )
        return con.execute(
            "SELECT user_id, balance_stars, free_message_used FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()


def get_user_balance(user_id: int) -> int:
    row = ensure_user(user_id)
    return int(row["balance_stars"])


def add_user_balance(user_id: int, delta: int) -> None:
    if delta <= 0:
        return
    with db() as con:
        row = con.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,)).fetchone()
        if row:
            con.execute(
                "UPDATE users SET balance_stars = balance_stars + ? WHERE user_id=?",
                (delta, user_id),
            )
        else:
            con.execute(
                "INSERT INTO users(user_id, balance_stars, free_message_used) VALUES (?,?,?)",
                (user_id, delta, 0),
            )


async def ensure_balance_for_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> Tuple[bool, bool]:
    """
    Returns (allowed, should_charge).

    allowed = False -> handler should not proceed (we already showed message/keyboard).
    should_charge = True -> AFTER successful response need to deduct MESSAGE_PRICE_STARS.
    """
    if not update.effective_user or not update.effective_message:
        return False, False

    uid = update.effective_user.id

    # Admins don't pay
    user = ensure_user(uid)
    balance = int(user["balance_stars"])
    free_used = int(user["free_message_used"] or 0)

    # First message is free
    if not free_used:
        mark_free_message_used(uid)
        await update.effective_message.reply_text(
            tr(update, "first_message_free", price=MESSAGE_PRICE_STARS)
        )
        # If this is a free message - nothing to deduct (should_charge=False)
        return True, False

    # Further on - only for the stars
    if balance < MESSAGE_PRICE_STARS:
        await send_topup_keyboard(
            update,
            context,
            prefix_text=(
                "У вас недостаточно звёзд на балансе.\n"
                f"Стоимость одного сообщения: {MESSAGE_PRICE_STARS}⭐.\n"
                "Пополните баланс, пожалуйста:"
            ),
        )
        return False, False

    # Balance is enough – NOTHING to deduct yet, just mark that we need to deduct later
    return True, True


def mark_free_message_used(user_id: int) -> None:
    with db() as con:
        con.execute(
            "UPDATE users SET free_message_used=1 WHERE user_id=?",
            (user_id,),
        )


def consume_message_credit(user_id: int, amount: int) -> bool:
    """
    Try to deduct amount stars. Return True if successful.
    """
    if amount <= 0:
        return True
    with db() as con:
        cur = con.execute(
            "UPDATE users SET balance_stars = balance_stars - ? "
            "WHERE user_id=? AND balance_stars >= ?",
            (amount, user_id, amount),
        )
        return cur.rowcount > 0


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user:
        return
    uid = update.effective_user.id
    if not is_allowed_user(uid):
        await update.effective_message.reply_text(tr(update, "access_denied"))
        return

    st = ensure_chat_settings(update.effective_chat.id)
    lang = get_lang(update)

    if is_admin(uid):
        tariff_line = "Вы администратор – сообщения для вас бесплатны."
    else:
        tariff_line = (
            f"Первое сообщение для каждого пользователя - бесплатно.\n"
            f"Дальше каждое сообщение стоит {MESSAGE_PRICE_STARS}⭐ с внутреннего баланса.\n"
            "Команды: /balance - баланс, /topup - пополнение."
        )

    if lang == "ru":
        txt = textwrap.dedent(f"""
        Привет, {update.effective_user.first_name}! 👋

        Я ассистент на базе ChatGPT.

        • Текущая модель: `{st.model}`
        • Режим диалога: `{ 'one-shot (новый диалог на каждое сообщение)' if st.one_shot else 'continuous (помню историю)' }`

        Что я умею:
        • Отвечать на вопросы и просто общаться
        • Переключать модели: /model или /setmodel <id>
        • Искать информацию в интернете
        • Читать и использовать картинки, PDF и текстовые файлы
        • Экспортировать диалог в PDF: /export
        • Очищать историю: /new
        • Переключать режим диалога: /mode

        Просто отправь сообщение, картинку или файл, чтобы начать.
        """)
    else:
        txt = textwrap.dedent(f"""
        Hello, {update.effective_user.first_name}! 👋

        I am a ChatGPT-based assistant.

        • Current model: `{st.model}`
        • Dialog mode: `{ 'one-shot (new dialog per message)' if st.one_shot else 'continuous (keeps history)' }`

        What I can do:
        • Answer questions and chat in natural language
        • Switch between multiple models: /model or /setmodel <id>
        • Use web search when needed
        • Read and use images, PDFs and text files in replies
        • Export our conversation to a PDF file: /export
        • Clear history: /new
        • Toggle dialog mode: /mode

        Just send a message, image, file or voice note to start.
        """)
    
    await update.effective_message.reply_text(txt, parse_mode=ParseMode.MARKDOWN)


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_allowed_user(update.effective_user.id):
        return
    choices = [
        [InlineKeyboardButton("gpt-5.1",     callback_data="model:gpt-5.1")],
        [InlineKeyboardButton("gpt-5",       callback_data="model:gpt-5")],
        [InlineKeyboardButton("gpt-5-mini",  callback_data="model:gpt-5-mini")],
        [InlineKeyboardButton("gpt-5-nano",  callback_data="model:gpt-5-nano")],
        [InlineKeyboardButton("gpt-4o",      callback_data="model:gpt-4o")],
        [InlineKeyboardButton("gpt-4o-mini", callback_data="model:gpt-4o-mini")],
        [InlineKeyboardButton("o4-mini",     callback_data="model:o4-mini")],
        [InlineKeyboardButton("gpt-4.1",     callback_data="model:gpt-4.1")],
    ]
    await update.effective_message.reply_text(
        tr(update, "choose_model"),
        reply_markup=InlineKeyboardMarkup(choices),
    )


async def cb_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    if not q or not q.data or not update.effective_user:
        return
    if not is_allowed_user(update.effective_user.id):
        await q.answer(tr(update, "access_denied"))
        return
    if q.data.startswith("model:"):
        model = q.data.split(":", 1)[1]
        set_model(update.effective_chat.id, model)
        await q.answer(tr(update, "model_updated"))
        await q.edit_message_text(
            tr(update, "model_set", model=model),
            parse_mode=ParseMode.MARKDOWN,
        )


async def cmd_setmodel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_allowed_user(update.effective_user.id):
        return
    if not context.args:
        await update.effective_message.reply_text(tr(update, "setmodel_usage"))
        return
    model = context.args[0]
    set_model(update.effective_chat.id, model)
    await update.effective_message.reply_text(
        tr(update, "model_set", model=model),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_allowed_user(update.effective_user.id):
        return
    new_val = toggle_mode(update.effective_chat.id)
    key = "dialog_mode_one_shot" if new_val else "dialog_mode_continuous"
    await update.effective_message.reply_text(
        tr(update, key),
        parse_mode=ParseMode.MARKDOWN,
    )


async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_allowed_user(update.effective_user.id):
        return
    clear_history(update.effective_chat.id)
    await update.effective_message.reply_text(tr(update, "history_cleared"))


async def cmd_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user:
        return
    uid = update.effective_user.id
    if not is_allowed_user(uid):
        await update.effective_message.reply_text(tr(update, "access_denied"))
        return

    if is_admin(uid):
        await update.effective_message.reply_text(
            tr(update, "balance_admin_info", price=MESSAGE_PRICE_STARS)
        )
        return

    user = ensure_user(uid)
    balance = int(user["balance_stars"])
    free_left = 0 if user["free_message_used"] else 1
    txt = tr(update, "balance_info", balance=balance, price=MESSAGE_PRICE_STARS)
    if free_left:
        txt += "\n" + tr(update, "balance_free_left")
    await update.effective_message.reply_text(txt)


async def send_topup_keyboard(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    prefix_text: str | None = None,
):
    if not update.effective_message:
        return

    rows = [
        [InlineKeyboardButton(f"{a}⭐", callback_data=f"topup:{a}")
         for a in TOPUP_OPTIONS[i:i + 3]]
        for i in range(0, len(TOPUP_OPTIONS), 3)
    ]
    text = prefix_text or tr(update, "topup_choose")
    await update.effective_message.reply_text(
        text,
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def cmd_topup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user:
        return
    uid = update.effective_user.id
    if not is_allowed_user(uid):
        await update.effective_message.reply_text(tr(update, "access_denied"))
        return
    await send_topup_keyboard(update, context)


async def send_stars_invoice(
    chat_id: int,
    user_id: int,
    stars_amount: int,
    context: ContextTypes.DEFAULT_TYPE,
):
    if stars_amount <= 0:
        return

    payload = f"topup:{user_id}:{stars_amount}:{uuid4().hex}"
    prices = [LabeledPrice(label=f"{stars_amount}⭐ баланс", amount=stars_amount)]

    await context.bot.send_invoice(
        chat_id=chat_id,
        title=f"Пополнение баланса на {stars_amount}⭐",
        description="Оплата доступа к боту. Звёзды будут зачислены на ваш внутренний баланс.",
        payload=payload,
        provider_token="",  # Telegram Stars doesn't require a provider.
        currency="XTR",
        prices=prices,
    )


async def cb_topup(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not update.effective_user:
        return

    uid = update.effective_user.id
    if not is_allowed_user(uid):
        await query.answer(tr(update, "access_denied"))
        return

    data = query.data or ""
    if not data.startswith("topup:"):
        await query.answer()
        return

    try:
        amount = int(data.split(":", 1)[1])
    except ValueError:
        await query.answer(tr(update, "topup_invalid_amount"))
        return

    await query.answer()
    chat_id = query.message.chat_id  # type: ignore[attr-defined]
    await send_stars_invoice(chat_id=chat_id, user_id=uid, stars_amount=amount, context=context)


async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.pre_checkout_query
    if not query:
        return
    try:
        if query.currency != "XTR":
            await query.answer(ok=False, error_message="Неподдерживаемая валюта")
            return
        await query.answer(ok=True)
    except Exception as e:
        log.exception("Precheckout error: %s", e)
        try:
            await query.answer(ok=False, error_message="Ошибка при обработке платежа")
        except Exception:
            pass


async def successful_payment_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not update.effective_message:
        return

    sp = update.effective_message.successful_payment
    if not sp:
        return

    user_id = update.effective_user.id
    stars = sp.total_amount  # for XTR: number of units = number of stars
    payload = sp.invoice_payload or ""

    add_user_balance(user_id, stars)
    with db() as con:
        con.execute(
            "INSERT INTO payments(user_id, stars_amount, telegram_payment_charge_id, "
            "provider_payment_charge_id, invoice_payload) "
            "VALUES (?,?,?,?,?)",
            (
                user_id,
                stars,
                sp.telegram_payment_charge_id,
                sp.provider_payment_charge_id,
                payload,
            ),
        )

    await update.effective_message.reply_text(
        tr(update, "topup_success", stars=stars)
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # 0) access
    if not update.effective_user or not update.effective_chat:
        return

    uid = update.effective_user.id
    if not is_allowed_user(uid):
        if update.effective_message:
            await update.effective_message.reply_text(tr(update, "access_denied"))
        return

    # 0.5) биллинг: проверяем, можно ли отвечать, НО пока ничего не списываем
    should_charge = False  # флаг, нужно ли потом списать за это сообщение
    if not is_admin(uid):
        allowed, should_charge = await ensure_balance_for_message(update, context)
        if not allowed:
            # либо это было первое бесплатное (мы просто продолжаем, но не будем списывать),
            # либо мы показали клавиатуру пополнения и дальше отвечать нельзя
            return

    chat_id = update.effective_chat.id
    st = ensure_chat_settings(chat_id)

    # 1) Extract text and attachments from a single message
    msg = update.effective_message
    if msg.text:
        text = apply_entities_as_markdown(msg.text, msg.entities)
    elif msg.caption:
        text = apply_entities_as_markdown(msg.caption, msg.caption_entities)
    else:
        text = None

    image_bytes_list: List[bytes] = []
    text_files: List[Tuple[str, str]] = []
    pdf_files:  List[Tuple[bytes, str]] = []
    other_files: List[str] = []
    photo_count = 0
    image_doc_names: List[str] = []

    if update.effective_message.photo:
        photo = update.effective_message.photo[-1]
        file = await photo.get_file()
        fbytes = await file.download_as_bytearray()
        image_bytes_list.append(bytes(fbytes))
        photo_count += 1

    if update.effective_message.document:
        doc = update.effective_message.document
        file = await doc.get_file()
        fbytes = await file.download_as_bytearray()
        mime = (doc.mime_type or "")
        fname = doc.file_name or "file.bin"
        if mime.startswith("image/"):
            image_bytes_list.append(bytes(fbytes))
            if fname:
                image_doc_names.append(fname)
        elif mime == "application/pdf" or fname.lower().endswith(".pdf"):
            pdf_files.append((bytes(fbytes), fname))
        else:
            try:
                text_data = bytes(fbytes).decode("utf-8", errors="replace")
                text_files.append((fname, text_data))
            except Exception:
                other_files.append(fname)

    # 2) collecting user_parts (text + images + text attachments)
    user_parts = build_user_content(text, image_bytes_list)

    used_parts = 0
    truncated_any = False
    for fname, text_data in text_files:
        if used_parts >= MAX_TEXT_PARTS_TOTAL:
            truncated_any = True
            break
        parts = chunk_text(text_data)
        parts = parts[:max(0, MAX_TEXT_PARTS_TOTAL - used_parts)]
        for i, part in enumerate(parts, start=1):
            header = f"[{fname} - part {i}/{len(parts)}]"
            user_parts.append({"type": "input_text", "text": f"{header}\n{part}"})
            used_parts += 1

    if truncated_any:
        user_parts.append({
            "type": "input_text",
            "text": "(Note: extra text parts were omitted to fit the model's context window.)"
        })

    for fb, fname in pdf_files:
        try:
            bio = io.BytesIO(fb); bio.name = fname
            uploaded = client.files.create(file=bio, purpose="assistants")
            user_parts.append({"type": "input_file", "file_id": uploaded.id})
        except Exception as e:
            log.exception("OpenAI file upload failed for %s", fname)
            user_parts.append({"type": "input_text", "text": f"[PDF '{fname}' upload failed: {e}]"})

    if other_files:
        warn = ", ".join(other_files[:5])
        user_parts.append({
            "type": "input_text",
            "text": f"Note: the following files are in unsupported formats for context stuffing. Please resend as PDF or paste text: {warn}"
        })

    # 3) Let's add a short note about attachments (both in the prompt and in the history)
    attach_note = _compose_attachment_note(
        text_files=text_files,
        pdf_files=pdf_files,
        image_doc_names=image_doc_names,
        photo_count=photo_count,
        other_files=other_files,
    )
    if attach_note:
        user_parts.append({"type": "input_text", "text": attach_note})

    # 4) One-shot mode - clear the history BEFORE reading the history
    if st.one_shot:
        with db() as con:
            con.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))

    # We take the history WITHOUT the current user message
    hist = db_get_history(chat_id)

    # 5) we save the user's CURRENT message in the database (so that it appears in /export, etc.)
    base_text = (text or "").strip()
    store_text = (base_text + attach_note).strip() or "[Attachment]"
    add_message(chat_id, "user", store_text)

    # 6) "status" message + single model call
    async with temp_status(update, context, text=tr(update, "thinking")) as stmsg:
        try:
            reply_text, _msgs = run_model_with_tools(
                model=st.model,
                system_prompt=SYSTEM_PROMPT,
                history=hist,          # history without current user
                user_parts=user_parts, # current user message (text/attachments)
            )
        except Exception as e:
            # We show the error and leave the status message as final.
            stmsg.delete_on_done = False
            try:
                await stmsg.msg.edit_text(tr(update, "error_generic", error=e))
            except Exception:
                pass
            return

        # 7) Successful response: either edit the status or delete and send it in the usual way
        if (PROGRESS_MODE == "edit"
            and not _env_true("ALWAYS_PDF", "false")
            and len(reply_text or "") <= _get_long_msg_threshold()):
            stmsg.delete_on_done = False  # do not delete in finally
            safe_html = prettify_telegram_html(reply_text or "")
            try:
                await stmsg.msg.edit_text(safe_html, parse_mode=ParseMode.HTML)
            except Exception:
                # If the edit fails, delete it and send it in the usual way.
                try: await stmsg.msg.delete()
                except Exception: pass
                await send_text_or_pdf(update, reply_text or "", user_text_for_pdf=text or "")
        else:
            # Long answer/ALWAYS_PDF - remove the status and send
            try: await stmsg.msg.delete()
            except Exception: pass
            await send_text_or_pdf(update, reply_text or "", user_text_for_pdf=text or "")

    # 8) save the assistant's response to history + списание звёзд
    if reply_text and reply_text != "(no response)":
        # Если не админ и при проверке решили, что это платное сообщение - списываем СЕЙЧАС
        if not is_admin(uid) and should_charge:
            consume_message_credit(uid, MESSAGE_PRICE_STARS)
        add_message(chat_id, "assistant", reply_text[:20_000])


    # 1) Protect bullet starters "* " at the beginning of a line.
    text = re.sub(r"(?m)^\* ", "{MD_BULLET} ", text)

    # 2) Combined emphasis first: ***text*** -> _*text*_
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"_*\1*_", text)

    # 3) Bold: **text** -> placeholders so it won't be converted by the italic step.
    text = re.sub(r"\*\*(.+?)\*\*", r"{B_OPEN}\1{B_CLOSE}", text)

    # 4) Italic: single *text* (not part of **...**) -> _text_
    # No leading/trailing spaces inside, avoid crossing lines.
    text = re.sub(r"(?<!\*)\*(?!\*)(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)", r"_\1_", text)

    # 5) Restore bold placeholders to Telegram bold: *text*
    text = text.replace("{B_OPEN}", "*").replace("{B_CLOSE}", "*")

    # 6) Restore bullet markers.
    text = text.replace("{MD_BULLET} ", "* ")

    return text


def chunk_text(s: str, max_len: int = MAX_TEXT_PART_LEN) -> List[str]:
    return [s[i:i+max_len] for i in range(0, len(s), max_len)]


def split_telegram_message(text: str, limit: int = 4000) -> List[str]:
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        if cur_len + len(line) > limit:
            parts.append("".join(cur))
            cur = [line]
            cur_len = len(line)
        else:
            cur.append(line)
            cur_len += len(line)
    if cur:
        parts.append("".join(cur))
    return parts


# ==============================
# PDF rendering helpers (HTML -> PDF with graceful fallback)
# ==============================
import datetime
from pathlib import Path
from html import escape as _html_escape

PDF_AUTHOR = os.getenv("PDF_AUTHOR", "by ButterDevelop")

def _env_true(name: str, default: str = "false") -> bool:
    """Parse boolean-like env var."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

def _html_escape_preserve(s: str) -> str:
    """HTML-escape text safely."""
    return _html_escape(s or "")

def _format_title(update: Optional["Update"] = None) -> str:
    """Build a dynamic PDF title using PDF_TITLE_PATTERN and placeholders."""
    now = datetime.datetime.now()
    ts = now.strftime("%Y-%m-%d %H.%M")
    pattern = os.getenv("PDF_TITLE_PATTERN", "ChatGPT × Telegram - Reply {ts}")

    chat_id = ""
    username = ""
    try:
        if update and update.effective_user:
            username = update.effective_user.username or update.effective_user.first_name or ""
        if update and update.effective_chat:
            chat_id = str(update.effective_chat.id)
    except Exception:
        pass

    return pattern.format(
        ts=ts,
        date=now.strftime("%Y-%m-%d"),
        time=now.strftime("%H:%M"),
        chat_id=chat_id,
        username=username,
    )

def _text_to_html(s: str) -> str:
    """Turn plain text with Markdown-like code into HTML with explicit <br>.
    - ```code``` fenced blocks -> <pre><code>…</code></pre>
    - `inline` -> <code>…</code>
    - Remaining newlines -> <br>
    """
    if not s:
        return ""
    # Protect fenced code first
    fences = []
    def _grab_fence(m):
        code = m.group(2)
        fences.append(_html_escape_preserve(code))
        return f"{{{{FENCE_{len(fences)-1}}}}}"
    s = re.sub(r"```([a-zA-Z0-9_-]+)?\s*\n(.*?)```", _grab_fence, s, flags=re.DOTALL)

    # Escape HTML
    s = _html_escape_preserve(s)

    # Inline `code`
    s = re.sub(r"`([^`]+)`", lambda m: f"<code>{_html_escape_preserve(m.group(1))}</code>", s)

    # Newlines -> <br>
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")

    # Restore fenced blocks as <pre><code>
    for i, code in enumerate(fences):
        s = s.replace(f"{{{{FENCE_{i}}}}}", f"<pre><code>{code}</code></pre>")
    return s

def _find_font_path() -> Optional[str]:
    """Locate a TTF with Cyrillic support.
    Priority: explicit env -> ./fonts next to this file -> OS paths.
    """
    # 1) Explicit env path
    p = os.getenv("PDF_FONT_PATH", "").strip()
    if p and os.path.isfile(p):
        return p

    # 2) ./fonts next to this Python file (or custom PDF_FONT_BUNDLE_DIR)
    bundle_dir_name = os.getenv("PDF_FONT_BUNDLE_DIR", "fonts").strip()
    bundle_dir = (_project_root() / bundle_dir_name).resolve()
    candidates = _candidate_fonts()

    def _bundle_variants(name: str) -> list[Path]:
        base = name.replace(" ", "")
        return [
            bundle_dir / f"{name}-Regular.ttf",
            bundle_dir / f"{base}-Regular.ttf",
            bundle_dir / f"{name}.ttf",
            bundle_dir / f"{base}.ttf",
        ]

    if bundle_dir.is_dir():
        for name in candidates:
            for fp in _bundle_variants(name):
                if fp.is_file():
                    return str(fp)

    # 3) Well-known OS locations (best effort)
    os_paths: list[str] = []
    os_paths += [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    win = "C:/Windows/Fonts"
    os_paths += [
        f"{win}/NotoSans-Regular.ttf",
        f"{win}/DejaVuSans.ttf",
        f"{win}/Roboto-Regular.ttf",
        f"{win}/arial.ttf",
        f"{win}/segoeui.ttf",
    ]
    for fp in os_paths:
        if os.path.isfile(fp):
            return fp

    return None

def _font_css() -> str:
    """@font-face for the discovered TTF + robust CSS fallback stack."""
    fp = _find_font_path()
    fallback_stack = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Noto Sans', 'DejaVu Sans', Arial, sans-serif"
    if not fp:
        return f"""
/* No embeddable TTF found; relying on system fallback */
body, .bubble, .label, header h1, header .meta, .footer {{ font-family: {fallback_stack}; }}
code, pre {{ font-family: {fallback_stack}, 'Courier New', monospace; }}
"""
    url = "file://" + Path(fp).as_posix()
    return f"""
@font-face {{
  font-family: 'PDFMain';
  src: url('{url}') format('truetype');
  font-weight: normal; font-style: normal;
}}
body, .bubble, .label, header h1, header .meta, .footer {{
  font-family: 'PDFMain', {fallback_stack};
}}
code, pre {{
  font-family: 'PDFMain', 'Courier New', monospace;
}}
"""

def _candidate_fonts() -> list[str]:
    """Prioritized font family base names to try."""
    raw = os.getenv("PDF_FONT_STACK", "").strip()
    if raw:
        return [x.strip() for x in raw.split("|") if x.strip()]
    return ["Noto Sans", "DejaVuSans", "Roboto", "Liberation Sans", "Arial"]

def _project_root() -> Path:
    """Return directory of this file (works with scripts, modules, and PyInstaller)."""
    import sys
    # PyInstaller support
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent

def _bundle_dir() -> Path:
    """Resolve the 'fonts' directory next to this Python file (unless an absolute path is given)."""
    cfg = os.getenv("PDF_FONT_BUNDLE_DIR", "fonts")
    p = Path(cfg)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    return p

def _bundle_variants(name: str, bundle: Path) -> list[Path]:
    """Possible file name variants for the given font name."""
    base = name.replace(" ", "")
    return [
        bundle / f"{name}-Regular.ttf",
        bundle / f"{base}-Regular.ttf",
        bundle / f"{name}.ttf",
        bundle / f"{base}.ttf",
        bundle / f"{name}.otf",
        bundle / f"{base}.otf",
    ]

def build_dialog_html(user_text: str, assistant_text: str, title: str) -> str:
    """Return HTML with running footer (left link + right page counter) for WeasyPrint."""
    import os
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    u = _text_to_html(user_text)
    a = _text_to_html(assistant_text)
    font_css = _font_css()

    link_text  = os.getenv("PDF_FOOTER_LINK_TEXT", "").strip()
    link_url   = os.getenv("PDF_FOOTER_LINK_URL", "").strip()
    page_label = os.getenv("PDF_PAGE_LABEL", "Page")
    link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")

    safe_href = _iri_to_uri(link_url) if link_text and link_url else ""
    left_footer_html = (
        f'<a href="{_html_escape_preserve(safe_href)}" '
        f'style="text-decoration: underline; color: {link_color};">{_html_escape_preserve(link_text)}</a>'
        if safe_href else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_html_escape_preserve(title)}</title>
  <meta name="author" content="{_html_escape_preserve(PDF_AUTHOR)}"/>
  <style>
    {font_css}

    /* Running footer for WeasyPrint */
    @page {{
      size: A4;
      margin: 20mm;
      @bottom-left  {{
        content: element(pdf-footer-left);
      }}
      @bottom-right {{
        content: "{_html_escape_preserve(page_label)} " counter(page) " / " counter(pages);
        color: #6B7280;
        font-size: 10px;
      }}
    }}
    .pdf-footer-left {{
      position: running(pdf-footer-left);
      font-size: 10px;
      color: #374151;
    }}

    :root {{
      --bg: #f6f7fb; --ink: #111; --muted: #555;
      --user-bg: #e8f0ff; --bot-bg: #eef7ec;
      --bubble-radius: 14px; --card-radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 24px; font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Arial, sans-serif; color: var(--ink); background: white; }}
    .wrap {{ max-width: 800px; margin: 0 auto; padding: 24px; background: var(--bg); border-radius: var(--card-radius); border: 1px solid #e5e7ef; }}
    header {{ margin-bottom: 12px; padding-bottom: 8px; border-bottom: 1px solid #e5e7ef; }}
    header h1 {{ font-size: 18px; margin: 0 0 2px 0; }}
    header .meta {{ color: var(--muted); font-size: 12px; }}

    .bubble {{ padding: 10px 12px; border-radius: var(--bubble-radius); margin: 8px 0; border: 1px solid rgba(0,0,0,0.05); overflow-wrap: anywhere; word-break: break-word; hyphens: manual; }}
    .from-user {{ background: var(--user-bg); border-top-left-radius: 4px; }}
    .from-bot  {{ background: var(--bot-bg);  border-top-right-radius: 4px; }}
    .label {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .section {{ margin: 12px 0; }}
    .footer {{ margin-top: 14px; font-size: 11px; color: var(--muted); text-align: right; }}
    pre {{ margin: 8px 0; padding: 8px; background: #fff; border: 1px solid #e5e7ef; border-radius: 8px; overflow-wrap: anywhere; }}
    code {{ font-size: 12px; }}
    
    pre, pre code, code {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
      hyphens: manual;
    }}
  </style>
</head>
<body>
  <!-- Running footer (left box) -->
  <div class="pdf-footer-left" id="pdf-footer-left">{left_footer_html}</div>

  <div class="wrap">
    <header>
      <h1>{_html_escape_preserve(title)}</h1>
      <div class="meta">Generated: {now}</div>
    </header>

    <div class="section">
      <div class="label">User</div>
      <div class="bubble from-user">{u}</div>
    </div>

    <div class="section">
      <div class="label">Assistant</div>
      <div class="bubble from-bot">{a}</div>
    </div>

    <div class="footer">{_html_escape_preserve(PDF_AUTHOR)}</div>
  </div>
</body>
</html>"""

def _render_pdf_with_weasyprint(html: str) -> Optional[bytes]:
    """Try rendering PDF using WeasyPrint (best HTML/CSS support)."""
    try:
        from weasyprint import HTML
        return HTML(string=html).write_pdf()
    except Exception:
        return None

def _render_pdf_with_xhtml2pdf(html: str) -> Optional[bytes]:
    """Fallback: render PDF using xhtml2pdf (pure Python)."""
    try:
        from xhtml2pdf import pisa
        import io as _io
        out = _io.BytesIO()
        pisa_status = pisa.CreatePDF(_io.BytesIO(html.encode("utf-8")), dest=out, encoding="utf-8")
        if pisa_status.err:
            return None
        return out.getvalue()
    except Exception:
        return None

# ==============================
# Pretty ReportLab renderer helpers (mini-Markdown + bubbles)
# ==============================

def _register_reportlab_fonts() -> dict:
    """Register Regular/Bold/Italic/Mono TTF if available. Return font name map.
    Tries many filename variants (Bold, SemiBold, Medium, Italic, Oblique, Mono).
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
    except Exception:
        # Fallback to built-ins (no guaranteed Cyrillic)
        return {"main": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique", "mono": "Courier"}

    base_fp = _find_font_path()
    fonts = {"main": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique", "mono": "Courier"}

    def _try_reg(name: str, path: str) -> bool:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return True
        except Exception:
            return False

    found_bold = False
    found_italic = False
    found_mono = False

    if base_fp and os.path.isfile(base_fp):
        base_dir = Path(base_fp).parent
        stem = Path(base_fp).stem  # e.g., Roboto-Regular or Roboto

        # Regular
        if _try_reg("PDFMain", base_fp):
            fonts["main"] = "PDFMain"

            # Bold-ish candidates (prefer Bold, then SemiBold/DemiBold/Medium as pseudo-bold)
            bold_candidates = [
                f"{stem.replace('Regular','Bold')}.ttf",
                f"{stem}-Bold.ttf",
                f"{stem.replace('Regular','SemiBold')}.ttf",
                f"{stem}-SemiBold.ttf",
                f"{stem.replace('Regular','DemiBold')}.ttf",
                f"{stem}-DemiBold.ttf",
                f"{stem.replace('Regular','Medium')}.ttf",
                f"{stem}-Medium.ttf",
                "NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf", "Roboto-Bold.ttf",
                "LiberationSans-Bold.ttf"
            ]

            # Italic/Oblique candidates
            italic_candidates = [
                f"{stem.replace('Regular','Italic')}.ttf",
                f"{stem}-Italic.ttf",
                f"{stem.replace('Regular','Oblique')}.ttf",
                f"{stem}-Oblique.ttf",
                "NotoSans-Italic.ttf", "DejaVuSans-Oblique.ttf", "Roboto-Italic.ttf",
                "LiberationSans-Italic.ttf"
            ]

            # Monospace candidates
            mono_candidates = [
                "NotoSansMono-Regular.ttf", "DejaVuSansMono.ttf", "RobotoMono-Regular.ttf",
                "Cousine-Regular.ttf", "LiberationMono-Regular.ttf", "LucidaConsoleMono-Regular.ttf"
            ]

            for fn in bold_candidates:
                p = (base_dir / fn)
                if p.is_file() and _try_reg("PDFMain-bold", str(p)):
                    fonts["bold"] = "PDFMain-bold"; found_bold = True; break

            for fn in italic_candidates:
                p = (base_dir / fn)
                if p.is_file() and _try_reg("PDFMain-italic", str(p)):
                    fonts["italic"] = "PDFMain-italic"; found_italic = True; break

            for fn in mono_candidates:
                p = (base_dir / fn)
                if p.is_file() and _try_reg("PDFMain-mono", str(p)):
                    fonts["mono"] = "PDFMain-mono"; found_mono = True; break

            # Family mapping so <b>/<i> in Paragraph switch within the same family
            try:
                registerFontFamily(
                    "PDFMain",
                    normal=fonts["main"],
                    bold=fonts.get("bold", fonts["main"]),
                    italic=fonts.get("italic", fonts["main"]),
                    boldItalic=fonts.get("bold", fonts["main"]),
                )
            except Exception:
                pass

    # Logs to make it obvious what's missing
    if log:
        log.info("ReportLab fonts map: %s", fonts)
        if fonts["main"] == "PDFMain" and not (fonts["bold"].startswith("PDFMain") and found_bold):
            log.warning("PDFMain bold face not found - **bold** will look like regular until you add a Bold/SemiBold TTF to ./fonts")
        if fonts["main"] == "PDFMain" and not (fonts["italic"].startswith("PDFMain") and found_italic):
            log.warning("PDFMain italic face not found - *italic* will look like regular until you add an Italic/Oblique TTF to ./fonts")
        if not found_mono and fonts["mono"] == "Courier":
            log.warning("Mono font not found - inline code will use Courier (ASCII ok, but no Cyrillic).")

    return fonts


def _md_inline_to_para_html(s: str, mono_font_name: str) -> str:
    """
    Convert a tiny subset of Markdown to ReportLab Paragraph inline markup.

    Supports:
      - **bold**, *italic*
      - Links: [text](url) and *bare* URLs (https://..., http://..., www....)
      - Inline code via `...` and LaTeX \\( ... \\) / \\[ ... \\]
    Notes:
      - Bare URLs are underlined and colored (PDF_LINK_COLOR), and trailing
        punctuation like . , ; : ! ? … is *not* included in the link.
      - Closing ), ], } at the very end are trimmed only if unbalanced.
    """
    import re
    from xml.sax.saxutils import escape as _xml_escape
    from urllib.parse import urlsplit, urlunsplit, quote

    if not s:
        return ""

    # ---------- helpers ----------
    def _to_href(url: str) -> str:
        """Percent-encode path/query/fragment; add https:// for 'www.'; keep IDN host."""
        if not url:
            return url
        if url.startswith("www."):
            url = "https://" + url
        try:
            p = urlsplit(url)
            path  = quote(p.path,     safe="/:@-._~!$&'()*+,;=")
            query = quote(p.query,    safe="=&:@-._~!$'()*+,;/?")
            frag  = quote(p.fragment, safe=":@-._~!$&'()*+,;=")
            return urlunsplit((p.scheme, p.netloc, path, query, frag))
        except Exception:
            return url

    def _strip_trailing_punct(text: str) -> tuple[str, str]:
        """
        Strip trailing sentence punctuation from 'text'.
        Keep closing ), ], } only if balanced with an opening counterpart.
        Returns (clean, stripped_tail).
        """
        tail = ""
        # strip common sentence punctuation
        while text and text[-1] in ".,;:!?…":
            tail = text[-1] + tail
            text = text[:-1]
        # handle unmatched closing brackets
        pairs = {')': '(', ']': '[', '}': '{'}
        while text and text[-1] in pairs and text.count(pairs[text[-1]]) < text.count(text[-1]):
            tail = text[-1] + tail
            text = text[:-1]
        return text, tail

    # ---------- 0) Stash inline code and LaTeX math ----------
    code_ph: list[str] = []

    def _put_code(txt: str) -> str:
        code_ph.append(txt)
        return f"{{{{INLINECODE_{len(code_ph)-1}}}}}"

    s = re.sub(r"(?<!`)`([^`\n]+)`", lambda m: _put_code(m.group(1)), s)
    s = re.sub(r"\\\(([^)\n]+)\\\)",   lambda m: _put_code(m.group(1)), s)  # \( ... )
    s = re.sub(r"\\\[([^\]\n]+)\\\]",  lambda m: _put_code(m.group(1)), s)  # \[ ... ]

    # ---------- 0.5) Stash Markdown links [text](url) ----------
    link_ph: list[tuple[str, str]] = []

    def _put_link(m):
        link_ph.append((m.group(1), m.group(2)))
        return f"{{{{INLINELINK_{len(link_ph)-1}}}}}"

    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", _put_link, s)

    # ---------- 0.6) Stash *bare* URLs (http(s)://... or www....) ----------
    auto_ph: list[tuple[str, str]] = []
    url_re = re.compile(r"(?<![\w/])((?:https?://|www\.)[^\s<>()\[\]{}\"'\u00A0]+)")

    def _put_auto(m):
        raw = m.group(1)
        clean, _ = _strip_trailing_punct(raw)
        href = _to_href(clean)
        auto_ph.append((clean, href))
        return f"{{{{AUTOLINK_{len(auto_ph)-1}}}}}"

    s = url_re.sub(_put_auto, s)

    # ---------- 1) Escape to safe XML ----------
    s = _xml_escape(s)

    # ---------- 2) Bold then italic ----------
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)

    # ---------- 2.5) Restore explicit [text](url) links ----------
    link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")
    for i, (txt, href) in enumerate(link_ph):
        s = s.replace(
            f"{{{{INLINELINK_{i}}}}}",
            f"<a href=\"{_xml_escape(_to_href(href))}\" color=\"{link_color}\"><u>{_xml_escape(txt)}</u></a>"
        )

    # ---------- 2.6) Restore auto-links ----------
    for i, (label, href) in enumerate(auto_ph):
        s = s.replace(
            f"{{{{AUTOLINK_{i}}}}}",
            f"<a href=\"{_xml_escape(href)}\" color=\"{link_color}\"><u>{_xml_escape(label)}</u></a>"
        )

    # ---------- 3) Restore inline code as a 'pill' ----------
    ic_bg = os.getenv("PDF_INLINE_CODE_BG", "#F3F4F6")
    ic_fg = os.getenv("PDF_INLINE_CODE_TEXT", "#111111")
    pad_nbsp = "&nbsp;" * int(os.getenv("PDF_INLINE_CODE_PAD_NBSP", "1"))
    for i, txt in enumerate(code_ph):
        inner = _xml_escape(txt)
        s = s.replace(
            f"{{{{INLINECODE_{i}}}}}",
            f"<font backColor=\"{ic_bg}\" face=\"{mono_font_name}\" color=\"{ic_fg}\">{pad_nbsp}{inner}{pad_nbsp}</font>"
        )

    # ---------- 4) Newlines → <br/> ----------
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br/>")
    return s


def _md_to_flowables(md: str, styles: dict, fonts: dict):
    """Turn Markdown-ish text into Flowables with splitting support."""
    from reportlab.platypus import Paragraph, ListFlowable, ListItem, Spacer, HRFlowable
    from reportlab.lib import colors
    import re

    if not md:
        return [Paragraph("&nbsp;", styles["base"])]

    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i, n = 0, len(lines)
    out = []

    def para(text: str, style) -> Paragraph:
        return Paragraph(_md_inline_to_para_html(text, fonts["mono"]), style)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # --- Horizontal rule ---
        if re.fullmatch(r"\s*([-*_])\1\1+\s*", line):
            out.append(HRFlowable(width="100%", color=colors.HexColor("#b4b4b4"), thickness=1, spaceBefore=4, spaceAfter=6))
            i += 1
            continue

        # --- Fenced code ---
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            if i < n and lines[i].strip().startswith("```"):
                i += 1
            from xml.sax.saxutils import escape as _xml_escape
            code_html = _xml_escape("\n".join(buf)).replace("\n", "<br/>")
            out.append(Paragraph(code_html, styles["code"]))
            out.append(Spacer(0, 4))
            continue

        # --- Headings #...###### ---
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            style = styles.get(f"h{level}", styles["h3"])
            out.append(para(text, style))
            out.append(Spacer(0, 1))
            i += 1
            continue

        # --- Blockquote > ---
        if stripped.startswith(">"):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            quote_text = "\n".join(quote_lines)
            out.append(para(quote_text, styles["quote"]))
            out.append(Spacer(0, 4))
            continue

        # --- Ordered list 1. 2. / 1) 2) ---  (now only if >= 2 in a row)
        num_item_re = re.compile(r"^\s*\d+[.)]\s+")
        if num_item_re.match(line):
            # look-ahead: we collect consecutive numbered lines
            j = i
            items = []
            while j < n and num_item_re.match(lines[j]):
                item_txt = num_item_re.sub("", lines[j]).rstrip()
                items.append(ListItem(para(item_txt, styles["base"])))
                j += 1
    
            if len(items) >= 2:
                out.append(ListFlowable(items, bulletType="1", start="1", leftIndent=18))
                out.append(Spacer(0, 4))
                i = j
                continue

        # --- Bullet list - / * ---
        if stripped.startswith(("- ", "* ")):
            items = []
            while i < n and lines[i].strip().startswith(("- ", "* ")):
                item_txt = lines[i].strip()[2:].rstrip()
                items.append(ListItem(para(item_txt, styles["base"])))
                i += 1
            out.append(ListFlowable(items, bulletType="bullet", leftIndent=16, bulletFontName=fonts["main"]))
            out.append(Spacer(0, 4))
            continue

        # --- Regular paragraph (collect until blank/special) ---
        buf = [line]; i += 1
        while i < n and lines[i].strip() and not re.match(r"^\s*(```|#|>|[-*_]{3,}|\d+\.\s+|[-*]\s+)", lines[i].strip()):
            buf.append(lines[i]); i += 1
        while i < n and not lines[i].strip():
            i += 1

        out.append(para("\n".join(buf), styles["base"]))
        out.append(Spacer(0, 4))

    return out


def _bubble_split(flowables: list, width: float, bg, border, padding=6):
    """A single-column table that can split across pages, with cell background & padding."""
    from reportlab.platypus import Table, TableStyle
    rows = [[f] for f in flowables] if flowables else [[""]]
    tbl = Table(rows, colWidths=[width], repeatRows=0)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("BOX",          (0,0), (-1,-1), 0.5, border),
        ("LEFTPADDING",  (0,0), (-1,-1), padding),
        ("RIGHTPADDING", (0,0), (-1,-1), padding),
        ("TOPPADDING",   (0,0), (-1,-1), padding),
        ("BOTTOMPADDING",(0,0), (-1,-1), padding),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    return tbl


def _ensure_abs_url(u: str) -> str:
    """
    Make sure the footer URL is absolute and IRI-safe:
    - '@username' -> 'https://t.me/username'
    - 't.me/...' or 'example.com' -> add 'https://'
    - percent-encode path/query/fragment + IDNA host (via _iri_to_uri).
    """
    import re
    u = (u or "").strip()
    if not u:
        return ""
    if u.startswith("@"):
        u = "https://t.me/" + u.lstrip("@")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", u):
        u = "https://" + u
    return _iri_to_uri(u)


# ---------- HISTORY -> HTML (for WeasyPrint / xhtml2pdf) ----------
def build_history_html(history: List[Dict[str, str]], title: str) -> str:
    """Full chat HTML with bubbles, author line and big centered link."""
    import os, datetime
    font_css = _font_css()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # env-driven link (same as in ReportLab renderer)
    link_text  = os.getenv("PDF_FOOTER_LINK_TEXT", "").strip()
    link_url   = os.getenv("PDF_FOOTER_LINK_URL", "").strip()
    safe_href  = _ensure_abs_url(link_url or link_text)
    link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")

    # big center link (optional)
    big_link_html = ""
    if safe_href:
        from html import escape as _esc
        big_link_html = (
            f'<div style="text-align:center;margin:8px 0 12px 0;">'
            f'  <a href="{_esc(safe_href)}" '
            f'     style="text-decoration:none;color:{link_color};'
            f'            font-weight:600;font-size:15px;">{_esc(link_text or safe_href)}</a>'
            f'</div>'
        )

    # build message sections
    parts = []
    for m in history:
        role = (m.get("role") or "").lower()
        raw  = m.get("content") or ""
        html = _text_to_html(raw)
        if role == "assistant":
            parts.append(f"""
            <div class="section">
              <div class="label">Assistant</div>
              <div class="bubble from-bot">{html}</div>
            </div>""")
        elif role == "user":
            parts.append(f"""
            <div class="section">
              <div class="label">User</div>
              <div class="bubble from-user">{html}</div>
            </div>""")
        elif role == "system":
            parts.append(f"""
            <div class="section">
              <div class="label">System</div>
              <div class="bubble" style="background:#f3f4f6">{html}</div>
            </div>""")

    # left footer link (WeasyPrint running element)
    left_footer_html = ""
    if safe_href:
        from html import escape as _esc
        left_footer_html = (
            f'<a href="{_esc(safe_href)}" '
            f'style="text-decoration:underline;color:{link_color};">{_esc(link_text or safe_href)}</a>'
        )

    page_label = os.getenv("PDF_PAGE_LABEL", "Page")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{_html_escape_preserve(title)}</title>
  <meta name="author" content="{_html_escape_preserve(PDF_AUTHOR)}"/>
  <style>
    {font_css}
    @page {{
      size: A4;
      margin: 20mm;
      @bottom-left  {{ content: element(pdf-footer-left); }}
      @bottom-right {{ content: "{_html_escape_preserve(page_label)} " counter(page) " / " counter(pages); color:#6B7280; font-size:10px; }}
    }}
    .pdf-footer-left {{ position: running(pdf-footer-left); font-size:10px; color:#374151; }}

    :root {{ --bg:#f6f7fb; --ink:#111; --muted:#555; --user-bg:#e8f0ff; --bot-bg:#eef7ec; --bubble-radius:14px; --card-radius:18px; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; padding:24px; font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial,sans-serif; color:var(--ink); background:white; }}
    .wrap {{ max-width:800px; margin:0 auto; padding:24px; background:var(--bg); border-radius:var(--card-radius); border:1px solid #e5e7ef; }}
    header {{ margin-bottom:10px; padding-bottom:6px; border-bottom:1px solid #e5e7ef; }}
    header h1 {{ font-size:18px; margin:0 0 2px 0; }}
    header .meta {{ color:var(--muted); font-size:12px; }}

    .bubble {{ padding:10px 12px; border-radius:var(--bubble-radius); margin:8px 0; border:1px solid rgba(0,0,0,0.05); background:#fff; }}
    .from-user {{ background:var(--user-bg); border-top-left-radius:4px; }}
    .from-bot  {{ background:var(--bot-bg);  border-top-right-radius:4px; }}
    .label {{ font-size:12px; color:var(--muted); margin-bottom:4px; }}
    .section {{ margin:12px 0; }}
    .footer {{ margin-top:14px; font-size:11px; color:var(--muted); text-align:right; }}
    pre {{ margin:8px 0; padding:8px; background:#fff; border:1px solid #e5e7ef; border-radius:8px; overflow-wrap:anywhere; }}
    code {{ font-size:12px; }}
  </style>
</head>
<body>
  <div class="pdf-footer-left" id="pdf-footer-left">{left_footer_html}</div>

  <div class="wrap">
    <header>
      <h1>{_html_escape_preserve(title)}</h1>
      <div class="meta">Generated: {now}</div>
      <div class="meta">Author: {_html_escape_preserve(PDF_AUTHOR)}</div>
    </header>

    {big_link_html}

    {''.join(parts)}

    <div class="footer">{_html_escape_preserve(PDF_AUTHOR)}</div>
  </div>
</body>
</html>"""
# ---------- END HTML builder ----------


def _render_pdf_history_with_reportlab(history: List[Dict[str, str]], title: str) -> Optional[bytes]:
    """
    ReportLab conversation renderer:
    - Author (plain text) + big centered link on top.
    - Bubbles for each message (User/Assistant/System).
    - Footer with a robust /Link annotation on every page.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfgen import canvas as _rl_canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFName, PDFString
        from functools import partial
        from xml.sax.saxutils import escape as _xml_escape
        import io as _io, os, datetime
        from uuid import uuid4

        fonts = _register_reportlab_fonts()
        width, height = A4
        margin = 17 * mm
        content_w = width - 2 * margin

        # env link config
        link_text_env = os.getenv("PDF_FOOTER_LINK_TEXT", "").strip()
        link_url_env  = os.getenv("PDF_FOOTER_LINK_URL", "").strip()
        href          = _ensure_abs_url(link_url_env or link_text_env)
        link_text     = link_text_env or href
        link_color    = os.getenv("PDF_LINK_COLOR", "#2563EB")

        ss = getSampleStyleSheet()
        base = ParagraphStyle("Base", parent=ss["Normal"],
                              fontName=fonts["main"], fontSize=10, leading=14, textColor=colors.black)
        code = ParagraphStyle("Code", parent=base, fontName=fonts["mono"], fontSize=9, leading=12,
                              backColor=colors.whitesmoke, borderPadding=(4,3,3,3))
        title_style = ParagraphStyle("Title", parent=base, fontSize=14, leading=18, spaceAfter=6)
        meta_style  = ParagraphStyle("Meta",  parent=base, fontSize=8,  textColor=colors.grey, spaceAfter=8)
        label_style = ParagraphStyle("Label", parent=base, fontSize=9,  textColor=colors.Color(0.33,0.33,0.38), spaceAfter=4)

        # compact headings for _md_to_flowables
        h1 = ParagraphStyle("h1", parent=base, fontSize=16, leading=19, spaceBefore=1.5, spaceAfter=1.5, fontName=fonts.get("bold", fonts["main"]))
        h2 = ParagraphStyle("h2", parent=base, fontSize=14, leading=18, spaceBefore=1.5, spaceAfter=1.5, fontName=fonts.get("bold", fonts["main"]))
        h3 = ParagraphStyle("h3", parent=base, fontSize=12.5,leading=16, spaceBefore=1.2, spaceAfter=1.2, fontName=fonts.get("bold", fonts["main"]))
        h4 = ParagraphStyle("h4", parent=base, fontSize=11.5,leading=15, spaceBefore=1.0, spaceAfter=1.0, fontName=fonts.get("bold", fonts["main"]))
        h5 = ParagraphStyle("h5", parent=base, fontSize=11,  leading=14, spaceBefore=0.8, spaceAfter=0.8, fontName=fonts.get("bold", fonts["main"]))
        h6 = ParagraphStyle("h6", parent=base, fontSize=10.5,leading=13, spaceBefore=0.5, spaceAfter=0.5, fontName=fonts.get("bold", fonts["main"]))
        quote = ParagraphStyle("quote", parent=base, leftIndent=10, borderPadding=(6,4,6,4),
                               borderColor=colors.HexColor("#E5E7EB"), borderWidth=1, borderLeft=1,
                               backColor=colors.whitesmoke, textColor=colors.Color(0.15,0.15,0.15))
        styles = {"base": base, "code": code, "label": label_style,
                  "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "h6": h6, "quote": quote}

        big_link_style = ParagraphStyle("BigLink", parent=base, fontSize=12, leading=16,
                                        spaceBefore=6, spaceAfter=10, alignment=1)

        story = []
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))
        story.append(Paragraph(f'<font color="#6B7280">Author:&nbsp;</font>{_xml_escape(PDF_AUTHOR)}', meta_style))

        # big centered link
        if href:
            story.append(Paragraph(
                f'<link href="{_xml_escape(href)}" color="{_xml_escape(link_color)}">'
                f'<b>{_xml_escape(link_text)}</b></link>', big_link_style))

        # all messages
        for m in history:
            role = (m.get("role") or "").lower()
            txt  = m.get("content") or ""
            if role == "assistant":
                story.append(Paragraph("Assistant", label_style))
                story.append(_bubble_split(_md_to_flowables(txt, styles, fonts), content_w,
                                           bg=colors.HexColor("#EAF7EA"), border=colors.HexColor("#CFEAD1")))
            elif role == "user":
                story.append(Paragraph("User", label_style))
                story.append(_bubble_split(_md_to_flowables(txt, styles, fonts), content_w,
                                           bg=colors.HexColor("#E8F0FF"), border=colors.HexColor("#D3E0FF")))
            elif role == "system":
                story.append(Paragraph("System", label_style))
                story.append(_bubble_split(_md_to_flowables(txt, styles, fonts), content_w,
                                           bg=colors.HexColor("#F3F4F6"), border=colors.HexColor("#E5E7EB")))
            story.append(Spacer(0, 8))
        story.append(Paragraph(f"{PDF_AUTHOR}", meta_style))

        # footer canvas with unique link annotation
        class NumberedCanvas(_rl_canvas.Canvas):
            def __init__(self, *args, fonts=None, footer_text="", footer_url="", **kwargs):
                super().__init__(*args, **kwargs)
                self._saved_page_states = []
                self._fonts = fonts or {"main":"Helvetica"}
                self._footer_text = footer_text
                self._footer_url  = footer_url
                self._footer_size = int(os.getenv("PDF_FOOTER_SIZE", "10"))
                self._footer_margin = 17 * mm
                self._page_label = os.getenv("PDF_PAGE_LABEL", "Page")
                self._link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")
                self._uuid = uuid4().hex

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                total = len(self._saved_page_states) or 1
                for st in self._saved_page_states:
                    self.__dict__.update(st)
                    self._ann_seq = 0
                    self._draw_footer(total)
                    _rl_canvas.Canvas.showPage(self)
                _rl_canvas.Canvas.save(self)

            def _draw_footer(self, total_pages: int):
                self.saveState()
                try:
                    page_w, _ = getattr(self, "_pagesize", (595.27, 841.89))
                    y = max(12 * mm, self._footer_margin - 6 * mm)
                    x_left, x_right = self._footer_margin, page_w - self._footer_margin

                    txt = (self._footer_text or "").strip()
                    href_local = (self._footer_url or "").strip()
                    if txt:
                        self.setFont(self._fonts.get("main","Helvetica"), self._footer_size)
                        self.setFillColor(colors.HexColor(self._link_color))
                        self.drawString(x_left, y, txt)
                        t_w = pdfmetrics.stringWidth(txt, self._fonts.get("main","Helvetica"), self._footer_size)
                        self.setStrokeColor(colors.HexColor(self._link_color))
                        self.setLineWidth(0.7)
                        self.line(x_left, y-1, x_left+t_w, y-1)

                        if href_local:
                            rect = [float(x_left), float(y-2), float(x_left+t_w), float(y+self._footer_size+3)]
                            rect = [min(rect[0],rect[2]), min(rect[1],rect[3]), max(rect[0],rect[2]), max(rect[1],rect[3])]
                            ann = PDFDictionary()
                            ann[PDFName("Type")]    = PDFName("Annot")
                            ann[PDFName("Subtype")] = PDFName("Link")
                            ann[PDFName("Rect")]    = rect
                            ann[PDFName("Border")]  = [0,0,1]
                            ann[PDFName("H")]       = PDFName("I")
                            ann[PDFName("C")]       = [0.15,0.45,0.92]
                            ann[PDFName("Contents")] = PDFString(txt)
                            act = PDFDictionary(); act[PDFName("S")] = PDFName("URI"); act[PDFName("URI")] = PDFString(href_local)
                            ann[PDFName("A")] = act
                            self._ann_seq += 1
                            self._addAnnotation(ann, name=f"FURL_{self._uuid}_{self._pageNumber}_{self._ann_seq}")

                    self.setFont(self._fonts.get("main","Helvetica"), self._footer_size)
                    self.setFillColor(colors.Color(0.42,0.45,0.50))
                    self.drawRightString(x_right, y, f"{self._page_label} {self.getPageNumber()} / {total_pages}")
                finally:
                    self.restoreState()

        bio = _io.BytesIO()
        doc = SimpleDocTemplate(
            bio, pagesize=A4,
            leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
            title=title, author=PDF_AUTHOR,
        )
        doc.build(
            story,
            canvasmaker=partial(
                NumberedCanvas,
                fonts=fonts,
                footer_text=(link_text or ""),
                footer_url=href,
            ),
        )
        return bio.getvalue()
    except Exception as e:
        if log:
            log.exception("ReportLab history renderer failed: %s", e)
        return None


def render_pdf_chat_history(history: List[Dict[str, Any]], title: str) -> bytes:
    # 1) HTML for WeasyPrint
    html = build_history_html(history, title)

    # 2) Try WeasyPrint (best HTML/CSS quality)
    try:
        pdf = _render_pdf_with_weasyprint(html)
        if pdf:
            if log: log.info("PDF(engine=weasyprint) history OK")
            return pdf
    except Exception as e:
        if log: log.warning("PDF(engine=weasyprint) failed: %s", e)

    # 3) Fallback: accurate ReportLab (if installed)
    try:
        pdf = _render_pdf_history_with_reportlab(history, title)
        if pdf:
            if log: log.info("PDF(engine=reportlab) history OK (fallback)")
            return pdf
    except Exception as e:
        if log: log.warning("PDF(engine=reportlab) failed: %s", e)

    # 4) Last resort: xhtml2pdf (pure Python)
    try:
        pdf = _render_pdf_with_xhtml2pdf(html)
        if pdf:
            if log: log.info("PDF(engine=xhtml2pdf) history OK (last resort)")
            return pdf
    except Exception as e:
        if log: log.warning("PDF(engine=xhtml2pdf) failed: %s", e)

    raise RuntimeError("Failed to render history PDF with all backends.")


# ---------- /export command ----------
async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.effective_user or not is_allowed_user(update.effective_user.id):
        return
    chat_id = update.effective_chat.id
    history = db_get_history(chat_id)  # [{'role': 'user'|'assistant'|'system', 'content': '...'}]

    if not history:
        await update.effective_message.reply_text(tr(update, "export_empty"))
        return

    try:
        title = _format_title(update) + " - Chat export"
        pdf_bytes = render_pdf_chat_history(history, title)
        bio = io.BytesIO(pdf_bytes)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bio.name = f"chat_export_{ts}.pdf"
        await update.effective_message.reply_document(document=bio, caption=tr(update, "export_pdf_caption"))
    except Exception as e:
        log.exception("Export PDF failed: %s", e)
        await update.effective_message.reply_text(tr(update, "export_fail", error=e))


def _render_pdf_with_reportlab(user_text: str, assistant_text: str, title: str) -> Optional[bytes]:
    """
    ReportLab renderer:
      - Header shows Author from PDF_AUTHOR (plain text, no link).
      - Big centered clickable link in the body (always shown).
      - Footer keeps a robust clickable /Link annotation with a unique name per page.
      - Chat bubbles are wrapped in a single Flowable (Table) so Platypus never receives raw lists.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfgen import canvas as _rl_canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFName, PDFString
        from functools import partial
        from xml.sax.saxutils import escape as _xml_escape
        import io as _io
        import os
        import datetime
        from uuid import uuid4
        from urllib.parse import urlsplit, urlunsplit, quote

        # -------------------- URL helpers --------------------
        def _ensure_url(u: str) -> str:
            """Ensure the URL has a scheme; accept 'www.' or bare host."""
            u = (u or "").strip()
            if not u:
                return ""
            if u.startswith("www."):
                return "https://" + u
            if not (u.startswith("http://") or u.startswith("https://")):
                return "https://" + u
            return u

        def _safe_url(u: str) -> str:
            """Percent-encode path/query/fragment; keep Unicode host as-is."""
            if not u:
                return u
            try:
                p = urlsplit(u)
                path  = quote(p.path,     safe="/:@-._~!$&'()*+,;=")
                query = quote(p.query,    safe="=&:@-._~!$'()*+,;/?")
                frag  = quote(p.fragment, safe=":@-._~!$&'()*+,;=")
                return urlunsplit((p.scheme, p.netloc, path, query, frag))
            except Exception:
                return u

        # -------------------- fonts & page geometry --------------------
        fonts = _register_reportlab_fonts()
        width, height = A4
        margin = 17 * mm
        content_w = width - 2 * margin

        # -------------------- link configuration --------------------
        link_text_env = os.getenv("PDF_FOOTER_LINK_TEXT", "").strip()
        link_url_env  = os.getenv("PDF_FOOTER_LINK_URL", "").strip()
        raw_href = _ensure_url(link_url_env or link_text_env)
        href = _safe_url(raw_href)
        link_text_shown = link_text_env or (raw_href if raw_href else "")
        page_label = os.getenv("PDF_PAGE_LABEL", "Page")
        link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")

        # -------------------- styles --------------------
        ss = getSampleStyleSheet()
        base = ParagraphStyle(
            "Base", parent=ss["Normal"],
            fontName=fonts["main"], fontSize=10, leading=14, textColor=colors.black
        )
        code = ParagraphStyle(
            "Code", parent=base, fontName=fonts["mono"], fontSize=9, leading=12,
            backColor=colors.whitesmoke, borderPadding=(4,3,3,3),
            wordWrap="CJK"
        )
        title_style = ParagraphStyle("Title", parent=base, fontSize=14, leading=18, spaceAfter=6)
        meta_style  = ParagraphStyle("Meta",  parent=base, fontSize=8,  textColor=colors.grey, spaceAfter=8)
        label_style = ParagraphStyle("Label", parent=base, fontSize=9,  textColor=colors.Color(0.33,0.33,0.38), spaceAfter=4)

        # Compact headings used by markdown conversion
        h1 = ParagraphStyle("h1", parent=base, fontSize=16,   leading=19, spaceBefore=1.5, spaceAfter=1.5, fontName=fonts.get("bold", fonts["main"]))
        h2 = ParagraphStyle("h2", parent=base, fontSize=14,   leading=18, spaceBefore=1.5, spaceAfter=1.5, fontName=fonts.get("bold", fonts["main"]))
        h3 = ParagraphStyle("h3", parent=base, fontSize=12.5, leading=16, spaceBefore=1.2, spaceAfter=1.2, fontName=fonts.get("bold", fonts["main"]))
        h4 = ParagraphStyle("h4", parent=base, fontSize=11.5, leading=15, spaceBefore=1.0, spaceAfter=1.0, fontName=fonts.get("bold", fonts["main"]))
        h5 = ParagraphStyle("h5", parent=base, fontSize=11,   leading=14, spaceBefore=0.8, spaceAfter=0.8, fontName=fonts.get("bold", fonts["main"]))
        h6 = ParagraphStyle("h6", parent=base, fontSize=10.5, leading=13, spaceBefore=0.5, spaceAfter=0.5, fontName=fonts.get("bold", fonts["main"]))

        quote = ParagraphStyle(
            "quote", parent=base, leftIndent=10, borderPadding=(6,4,6,4),
            borderColor=colors.HexColor("#E5E7EB"), borderWidth=1, borderLeft=1,
            backColor=colors.whitesmoke, textColor=colors.Color(0.15,0.15,0.15)
        )

        styles = {"base": base, "code": code, "label": label_style,
                  "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "h6": h6, "quote": quote}

        big_link_style = ParagraphStyle(
            "BigLink", parent=base, fontSize=12, leading=16, spaceBefore=6, spaceAfter=10, alignment=1  # center
        )

        # -------------------- story (Flowables only) --------------------
        story = []
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style))

        # Author line: plain text (same as PDF metadata 'author')
        story.append(Paragraph(
            f'<font color="#6B7280">Author:&nbsp;</font>{_xml_escape(PDF_AUTHOR)}',
            meta_style
        ))

        # Big centered clickable link (ALWAYS shown as requested)
        if href:
            story.append(Paragraph(
                f'<link href="{_xml_escape(href)}" color="{_xml_escape(link_color)}"><b>{_xml_escape(link_text_shown)}</b></link>',
                big_link_style
            ))

        # User bubble
        story.append(Paragraph("User", label_style))
        story.append(
            _bubble_split(
                _md_to_flowables(user_text or "", styles, fonts),
                content_w,
                bg=colors.HexColor("#E8F0FF"),
                border=colors.HexColor("#D3E0FF")
            )
        )
        story.append(Spacer(0, 10))

        # Assistant bubble
        story.append(Paragraph("Assistant", label_style))
        story.append(
            _bubble_split(
                _md_to_flowables(assistant_text or "", styles, fonts),
                content_w,
                bg=colors.HexColor("#EAF7EA"),
                border=colors.HexColor("#CFEAD1")
            )
        )
        story.append(Spacer(0, 8))
        story.append(Paragraph(f"{PDF_AUTHOR}", meta_style))

        # -------------------- two-pass canvas with footer link --------------------
        class NumberedCanvas(_rl_canvas.Canvas):
            """
            Two-pass canvas:
              - showPage(): stash page state only.
              - save(): replay pages, draw footer, attach a unique /Link annotation, and output.
            """
            def __init__(self, *args, fonts=None, footer_text="", footer_url="", **kwargs):
                super().__init__(*args, **kwargs)
                self._saved_page_states = []
                self._fonts = fonts or {"main": "Helvetica"}
                self._footer_text = footer_text or ""
                self._footer_url  = footer_url or ""
                self._footer_size = int(os.getenv("PDF_FOOTER_SIZE", "10"))
                self._footer_margin = 17 * mm
                self._page_label = os.getenv("PDF_PAGE_LABEL", "Page")
                self._link_color = os.getenv("PDF_LINK_COLOR", "#2563EB")
                self._doc_uuid = uuid4().hex

            def showPage(self):
                self._saved_page_states.append(dict(self.__dict__))
                self._startPage()

            def save(self):
                total_pages = len(self._saved_page_states) or 1
                for st in self._saved_page_states:
                    self.__dict__.update(st)
                    self._ann_seq = 0
                    self._draw_footer(total_pages)
                    _rl_canvas.Canvas.showPage(self)
                _rl_canvas.Canvas.save(self)

            def _draw_footer(self, total_pages: int):
                """Draw a left-aligned footer link + right-aligned page counter and add a /Link annotation."""
                self.saveState()
                try:
                    page_w, _page_h = getattr(self, "_pagesize", (595.27, 841.89))
                    margin = self._footer_margin

                    # Lift the footer above bottom overlays
                    y = max(12 * mm, margin - 6 * mm)
                    x_left, x_right = margin, page_w - margin

                    txt = (self._footer_text or "").strip()
                    href_local = (self._footer_url or "").strip()

                    if txt:
                        # Visible text + underline
                        self.setFont(self._fonts.get("main", "Helvetica"), self._footer_size)
                        self.setFillColor(colors.HexColor(self._link_color))
                        self.drawString(x_left, y, txt)
                        t_w = pdfmetrics.stringWidth(txt, self._fonts.get("main", "Helvetica"), self._footer_size)
                        self.setStrokeColor(colors.HexColor(self._link_color))
                        self.setLineWidth(0.7)
                        self.line(x_left, y - 1, x_left + t_w, y - 1)

                        # Clickable annotation with a unique name
                        if href_local:
                            rect = [float(x_left), float(y - 2),
                                    float(x_left + t_w), float(y + self._footer_size + 3)]
                            rect = [min(rect[0], rect[2]), min(rect[1], rect[3]),
                                    max(rect[0], rect[2]), max(rect[1], rect[3])]

                            ann = PDFDictionary()
                            ann[PDFName("Type")]    = PDFName("Annot")
                            ann[PDFName("Subtype")] = PDFName("Link")
                            ann[PDFName("Rect")]    = rect
                            ann[PDFName("Border")]  = [0, 0, 1]
                            ann[PDFName("H")]       = PDFName("I")
                            ann[PDFName("C")]       = [0.15, 0.45, 0.92]
                            ann[PDFName("Contents")] = PDFString(txt)

                            action = PDFDictionary()
                            action[PDFName("S")]   = PDFName("URI")
                            action[PDFName("URI")] = PDFString(href_local)
                            ann[PDFName("A")]      = action

                            self._ann_seq += 1
                            unique_name = f"FURL_{self._doc_uuid}_{self._pageNumber}_{self._ann_seq}"
                            self._addAnnotation(ann, name=str(unique_name))

                    # Right-side page counter
                    self.setFont(self._fonts.get("main", "Helvetica"), self._footer_size)
                    self.setFillColor(colors.Color(0.42, 0.45, 0.50))
                    self.drawRightString(x_right, y, f"{self._page_label} {self.getPageNumber()} / {total_pages}")
                finally:
                    self.restoreState()

        # -------------------- build --------------------
        bio = _io.BytesIO()
        doc = SimpleDocTemplate(
            bio, pagesize=A4,
            leftMargin=margin, rightMargin=margin, topMargin=margin, bottomMargin=margin,
            title=title, author=PDF_AUTHOR,
        )
        doc.build(
            story,
            canvasmaker=partial(
                NumberedCanvas,
                fonts=fonts,
                footer_text=link_text_shown,
                footer_url=href,  # e.g., "https://t.me/bot"
            ),
        )
        return bio.getvalue()

    except Exception as e:
        if log:
            log.exception("ReportLab pretty renderer failed: %s", e)
        return None


# Turn bare URLs into [pretty](url) and emulate headings with bold.
_URL_RE = re.compile(r'(?<![\w/])((?:https?://|www\.)[^\s<>()\[\]{}"\'\u00A0]+)', re.IGNORECASE)
_URL_RE_BARE = re.compile(r'(?<!")\bhttps?://[^\s<>()]+', re.IGNORECASE)

def _iri_to_uri(href: str) -> str:
    """Converts IRI (Russian/Unicode) to a safe URI (ASCII + %XX)."""
    from urllib.parse import urlsplit, urlunsplit, unquote, quote
    try:
        s = urlsplit(href)
        # Host in Punycode (IDNA)
        host = s.hostname.encode("idna").decode("ascii") if s.hostname else ""
        netloc = host
        if s.port:
            netloc += f":{s.port}"
        if s.username:
            auth = s.username
            if s.password:
                auth += f":{s.password}"
            netloc = f"{auth}@{netloc}"
        # No double coding
        path  = quote(unquote(s.path or ""),     safe="/:@-._~!$&'()*+,;=")
        query = quote(unquote(s.query or ""),    safe="=&/?@-._~!$&'()*+,;=")
        frag  = quote(unquote(s.fragment or ""), safe="@-._~!$&'()*+,;=/")
        return urlunsplit((s.scheme, netloc, path, query, frag))
    except Exception:
        return href

def _pretty_url_label(href: str) -> str:
    """Nice view for display: unicode domain + decoded path."""
    from urllib.parse import urlsplit, unquote
    try:
        s = urlsplit(href)
        # We'll show the Unicode domain if it's in punycode.
        try:
            host = s.hostname.encode("ascii").decode("idna") if s.hostname else s.netloc
        except Exception:
            host = s.hostname or s.netloc
        path = unquote(s.path or "")
        if len(path) > 40:
            path = path[:37] + "…"
        return (host + path) if path else host
    except Exception:
        return href

def _humanize_url(url: str) -> str:
    from urllib.parse import urlparse, unquote
    try:
        p = urlparse(url)
        host = p.netloc
        path = unquote(p.path or "")
        if len(path) > 32:
            path = path[:29] + "…"
        label = (host + path) if path else host
        # strip Telegram-markdown control chars in label
        label = re.sub(r'[\[\]\(\)\*_`]', ' ', label).strip()
        return label or host or url
    except Exception:
        return url

def _compose_attachment_note(
    text_files: List[Tuple[str, str]],
    pdf_files: List[Tuple[bytes, str]],
    image_doc_names: List[str],
    photo_count: int,
    other_files: List[str],
) -> str:
    """
    Build a human-readable one-liner with attached filenames.
    - text_files: [(fname, text)]
    - pdf_files:  [(bytes, fname)]
    - image_doc_names: ['pic.png', ...] for image *documents* (not Telegram photos)
    - photo_count: count of Telegram photos (no filenames)
    - other_files: ['bin.dat', ...]
    """
    names: List[str] = []
    names += [fname for fname, _ in text_files]
    names += [fname for _, fname in pdf_files]
    names += image_doc_names
    names += other_files
    if photo_count:
        names.append(f"photos×{photo_count}")
    return f"\n\n[Attachments: {', '.join(names)}]" if names else ""

def _escape_md_legacy_text(s: str) -> str:
    """Escape Telegram legacy Markdown control chars in visible text."""
    return re.sub(r'([_*\[\]()`])', r'\\\1', s)

def _escape_md_legacy_url(s: str) -> str:
    """Escape critical chars inside (url) for Telegram legacy Markdown."""
    # Order matters: escape backslash first
    s = s.replace('\\', r'\\')
    s = s.replace('(', r'\(').replace(')', r'\)')
    s = s.replace('_', r'\_').replace('*', r'\*')
    return s

def prettify_telegram_text(text: str) -> str:
    """
    1) Stash code (fences & inline) so we don't touch it.
    2) Convert headings '# ...' -> **...** (Telegram has no real headings).
    3) Convert bare URLs to [pretty](url), trimming trailing punctuation.
    4) Adapt to Telegram legacy Markdown.
    """
    if not text:
        return text

    placeholders = []
    def _stash(m: re.Match) -> str:
        placeholders.append(m.group(0))
        return f"{{TG_CODE_{len(placeholders)-1}}}"

    s = _CODE_BLOCK_RE.sub(_stash, text)
    s = _CODE_SPAN_RE.sub(_stash, s)

    # Headings -> bold
    s = re.sub(r"(?m)^(#{1,6})\s+(.+)$", lambda m: f"**{m.group(2).strip()}**", s)

    # Helper: strip trailing punctuation (. , ; : ! ? …) and unmatched ) ] }
    def _strip_trailing(url: str) -> tuple[str, str]:
        tail = ""
        while url and url[-1] in ".,;:!?…":
            tail = url[-1] + tail
            url = url[:-1]
        pairs = {')': '(', ']': '[', '}': '{'}
        while url and url[-1] in pairs and url.count(pairs[url[-1]]) < url.count(url[-1]):
            tail = url[-1] + tail
            url = url[:-1]
        return url, tail

    def _ensure_scheme(u: str) -> str:
        return u if u.startswith(("http://", "https://")) else "https://" + u

    # Bare URLs -> [decoded-label](escaped-url)
    def _link_sub(m: re.Match) -> str:
        raw = m.group(1)
        clean, tail = _strip_trailing(raw)
        url = _ensure_scheme(clean)
        label = _humanize_url(url)  # already makes short convenient view
        return f"[{_escape_md_legacy_text(label)}]({_escape_md_legacy_url(url)}){tail}"

    s = _URL_RE.sub(_link_sub, s)

    # Unstash code
    def _unstash(m: re.Match) -> str:
        return placeholders[int(m.group(1))]
    s = re.sub(r"\{TG_CODE_(\d+)\}", _unstash, s)

    return normalize_for_telegram_markdown(s)

def prettify_telegram_html(text: str) -> str:
    """Prepares HTML for Telegram:
    - triple backticks -> <pre><code>
    - `inline` -> <code>
    - [label](url) and naked URLs -> <a href="...">label</a> (label without URL-encoded)
    - # ... -> <b>...</b>
    - **bold** / *italic* -> <b>/<i>
    Everything else is safely escaped.
    """
    if not text:
        return ""

    from html import escape as _esc
    from urllib.parse import urlparse, unquote

    # 0) We save the code so we don't have to mess around with its design
    blocks, inlines, links = [], [], []

    def _put_block(m):
        lang = (m.group(1) or "").strip()
        inner = m.group(2)
        html = f"<pre><code>{_esc(inner)}</code></pre>"
        blocks.append(html)
        return f"{{TG_BLOCK_{len(blocks)-1}}}"

    def _put_inline(m):
        html = f"<code>{_esc(m.group(1))}</code>"
        inlines.append(html)
        return f"{{TG_INLINE_{len(inlines)-1}}}"

    def _human_label(url: str) -> str:
        try:
            p = urlparse(url)
            path = unquote(p.path or "")
            if len(path) > 32:
                path = path[:29] + "…"
            lab = (p.netloc + path) if path else p.netloc
            return lab or url
        except Exception:
            return url

    def _put_md_link(m):
        label, href = m.group(1), m.group(2)
        html = f'<a href="{_esc(href)}">{_esc(label)}</a>'
        links.append(html)
        return f"{{TG_LINK_{len(links)-1}}}"

    s = text

    # Fenced code ```lang\n...\n```
    s = re.sub(r"```([a-zA-Z0-9_-]+)?\s*\n(.*?)```", _put_block, s, flags=re.DOTALL)
    # Inline `code`
    s = re.sub(r"(?<!`)`([^`\n]+)`", _put_inline, s)
    # Markdown links [label](url)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", _put_md_link, s)

    # We screen everything else
    s = _esc(s)

    # Headings '# ...' -> bold
    s = re.sub(r"(?m)^(#{1,6})\s+(.+)$", lambda m: f"<b>{m.group(2).strip()}</b>", s)

    # Bold / Italic
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)

    # Naked URLs -> <a>
    def _link_sub(m):
        url = m.group(0)
        return f'<a href="{_esc(url)}">{_esc(_human_label(url))}</a>'
    s = _URL_RE.sub(_link_sub, s)

    # Returning placeholders
    for i, html in enumerate(links):
        s = s.replace(f"{{TG_LINK_{i}}}", html)
    for i, html in enumerate(inlines):
        s = s.replace(f"{{TG_INLINE_{i}}}", html)
    for i, html in enumerate(blocks):
        s = s.replace(f"{{TG_BLOCK_{i}}}", html)

    # Note: Telegram understands line breaks in HTML itself, <br> is not needed
    return s

def render_pdf_dialog(user_text: str, assistant_text: str, title: str) -> bytes:
    """Render a nice PDF dialog. Prefer WeasyPrint; then pretty ReportLab; finally xhtml2pdf."""
    html = build_dialog_html(user_text, assistant_text, title)
    font_path = _find_font_path()

    # 1) WeasyPrint (best HTML/CSS)
    try:
        pdf = _render_pdf_with_weasyprint(html)
        if pdf:
            if log: log.info("PDF engine=weasyprint, font=%s", font_path)
            return pdf
    except Exception:
        pass

    # 2) Pretty ReportLab (guaranteed Cyrillic via TTF)
    try:
        pdf = _render_pdf_with_reportlab(user_text, assistant_text, title)
        if pdf:
            if log: log.info("PDF engine=reportlab, font=%s", font_path)
            return pdf
    except Exception:
        pass

    # 3) Last-resort xhtml2pdf
    try:
        pdf = _render_pdf_with_xhtml2pdf(html)
        if pdf:
            if log: log.info("PDF engine=xhtml2pdf, font=%s", font_path)
            return pdf
    except Exception:
        pass

    raise RuntimeError("Failed to render PDF with all backends.")

def _get_long_msg_threshold() -> int:
    """Read threshold from env every time, so .env is respected regardless of import order."""
    try:
        return int(os.getenv("TELEGRAM_LONG_MSG_THRESHOLD", "1002"))
    except Exception:
        return 1002

async def send_text_or_pdf(update: Update, raw_reply_text: str, user_text_for_pdf: Optional[str] = None):
    """Send a Telegram message; if it's too long (or ALWAYS_PDF), send as a pretty PDF."""
    text = raw_reply_text or ""
    force_pdf = _env_true("ALWAYS_PDF", "false")
    threshold = _get_long_msg_threshold()

    # Decide: always PDF, or only if length exceeds threshold
    if force_pdf or len(text) > threshold:
        try:
            usr = (user_text_for_pdf or "").strip()
            title = _format_title(update)
            pdf_bytes = render_pdf_dialog(usr, text, title)
            bio = io.BytesIO(pdf_bytes)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            bio.name = f"chat_reply_{ts}.pdf"
            await update.effective_message.reply_document(document=bio, caption=tr(update, "long_reply_pdf_caption"))
            return
        except Exception as e:
            log.exception("PDF generation failed, falling back to text: %s", e)

    # Fallback / short messages
    chunks = split_telegram_message(text) if text else ["(empty response)"]
    for chunk in chunks:
        safe = prettify_telegram_html(chunk)
        await update.effective_message.reply_text(safe, parse_mode=ParseMode.HTML)


# ==============================
# App bootstrap (sync runner)
# ==============================

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")
    db_init()
    application: Application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .concurrent_updates(True)
        .build()
    )

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("model", cmd_model))
    application.add_handler(CallbackQueryHandler(cb_model, pattern=r"^model:"))
    application.add_handler(CallbackQueryHandler(cb_topup, pattern=r"^topup:"))
    application.add_handler(CommandHandler("setmodel", cmd_setmodel))
    application.add_handler(CommandHandler("mode", cmd_mode))
    application.add_handler(CommandHandler("new", cmd_new))
    application.add_handler(CommandHandler("export", cmd_export))
    application.add_handler(CommandHandler("balance", cmd_balance))
    application.add_handler(CommandHandler("topup", cmd_topup))

    application.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    application.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_handler))

    # 1) Media (photos + any documents) -> go through the album/loose router
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, router_message), group=0)
    # 2) Plain text messages (no attachments) -> buffer for a short time to catch upcoming files
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router), group=1)


    log.info("Bot starting (long polling)...")
    application.run_polling()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped.")
