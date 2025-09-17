# -*- coding: utf-8 -*-
"""
Telegram football signal bot (single file).
- No persistence (in-memory only).
- Adjustable thresholds via /set_thresholds.
- Background poller fetches live matches and sends signals.
- Providers: mock (JSON), LiveScoreAPI, Sportmonks.

Env vars (use .env or export before run):
  BOT_TOKEN=...                 (required)
  PROVIDER=mock|livescoreapi|sportmonks
  DATA_FILE=mock_live.json      (for PROVIDER=mock)
  LSA_KEY=...                   (for LiveScoreAPI)
  LSA_SECRET=...                (for LiveScoreAPI)
  SPORTMONKS_TOKEN=...          (for Sportmonks)
  SPORTMONKS_BASE=https://api.sportmonks.com/v3/football  (optional)
  POLL_INTERVAL_SEC=20          (optional, default 20)
  CHAT_WHITELIST=               (optional, comma-separated chat ids)
"""

import asyncio
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.bot import DefaultBotProperties

import httpx

# .env support (safe if package missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# ----------------- config -----------------
BOT_TOKEN = os.getenv("BOT_TOKEN")
PROVIDER = (os.getenv("PROVIDER") or "mock").lower()
DATA_FILE = os.getenv("DATA_FILE", "mock_live.json")
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", 20))
CHAT_WHITELIST = os.getenv("CHAT_WHITELIST", "")
WHITELIST = {x.strip() for x in CHAT_WHITELIST.split(",") if x.strip()}

SPORTMONKS_TOKEN = os.getenv("SPORTMONKS_TOKEN")
SPORTMONKS_BASE = (os.getenv("SPORTMONKS_BASE") or "https://api.sportmonks.com/v3/football").rstrip("/")
LSA_KEY = os.getenv("LSA_KEY")
LSA_SECRET = os.getenv("LSA_SECRET")

# ----------------- engine -----------------
DEFAULT_ATTACKS_RATIO = 1.9
DEFAULT_XG_DIFF = 0.70
DEFAULT_XG_PCT = 50


@dataclass
class Thresholds:
    attacks_ratio: float = DEFAULT_ATTACKS_RATIO
    xg_diff: float = DEFAULT_XG_DIFF
    xg_pct: int = DEFAULT_XG_PCT

    def as_text(self) -> str:
        return (
            "Thresholds:\n"
            f"- Dangerous attacks ratio >= {self.attacks_ratio}\n"
            f"- xG difference >= {self.xg_diff}\n"
            f"- xG advantage >= {self.xg_pct}%"
        )


@dataclass
class LiveMatch:
    match_id: str
    league: str
    home: str
    away: str
    minute: int
    score_home: int
    score_away: int
    da_home: int
    da_away: int
    xg_home: Optional[float]
    xg_away: Optional[float]
    url: str = ""


class SignalEngine:
    @staticmethod
    def _ratio(a: int, b: int) -> Optional[float]:
        if a <= 0 and b <= 0:
            return None
        if b == 0:
            return float("inf") if a > 0 else None
        return a / b

    @staticmethod
    def _xg_ok(xh: Optional[float], xa: Optional[float], th: Thresholds) -> Tuple[bool, str]:
        if xh is None or xa is None:
            return True, "xG unavailable"
        diff = abs(xh - xa)
        hi, lo = max(xh, xa), min(xh, xa)
        pct = ((hi - lo) / lo * 100) if lo > 0 else float("inf")
        ok = (diff >= th.xg_diff) or (pct >= th.xg_pct)
        return ok, f"xG: {xh:.2f} vs {xa:.2f} (d={diff:.2f}, {pct:.0f}%)"

    def check(self, m: LiveMatch, th: Thresholds) -> Optional[Dict]:
        if m.minute < 75:
            return None
        if m.score_home != m.score_away:
            return None
        ratio = self._ratio(max(m.da_home, m.da_away), min(m.da_home, m.da_away))
        if ratio is None or ratio < th.attacks_ratio:
            return None
        xg_ok, xg_text = self._xg_ok(m.xg_home, m.xg_away, th)
        if not xg_ok:
            return None
        leader = m.home if m.da_home >= m.da_away else m.away
        return {
            "id": m.match_id,
            "league": m.league,
            "home": m.home,
            "away": m.away,
            "minute": m.minute,
            "score": f"{m.score_home}-{m.score_away}",
            "da": f"{m.da_home} vs {m.da_away}",
            "ratio": ratio,
            "xg_text": xg_text,
            "leader": leader,
            "url": m.url,
        }


# ----------------- providers -----------------
class MockProvider:
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)

    async def fetch_live(self) -> List[LiveMatch]:
        if not self.file_path.exists():
            return []
        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: List[LiveMatch] = []
        for i, item in enumerate(data.get("matches", [])):
            out.append(
                LiveMatch(
                    match_id=str(item.get("id", i)),
                    league=item.get("league", "League"),
                    home=item.get("home", "Home"),
                    away=item.get("away", "Away"),
                    minute=int(item.get("minute", 0)),
                    score_home=int(item.get("score_home", 0)),
                    score_away=int(item.get("score_away", 0)),
                    da_home=int(item.get("dangerous_attacks_home", 0)),
                    da_away=int(item.get("dangerous_attacks_away", 0)),
                    xg_home=(float(item["xg_home"]) if item.get("xg_home") is not None else None),
                    xg_away=(float(item["xg_away"]) if item.get("xg_away") is not None else None),
                    url=item.get("match_url", ""),
                )
            )
        return out


class LiveScoreAPIProvider:
    """Uses live-score-api.com; provides dangerous_attacks, no xG."""
    def __init__(self):
        if not (LSA_KEY and LSA_SECRET):
            raise RuntimeError("LSA_KEY/LSA_SECRET not set")
        self.base = "https://livescore-api.com/api-client"

    async def _get(self, path: str, params: Dict[str, str]) -> Dict:
        q = {"key": LSA_KEY, "secret": LSA_SECRET}
        q.update(params)
        url = f"{self.base}/{path}"
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=q)
            r.raise_for_status()
            return r.json()

    async def fetch_live(self) -> List[LiveMatch]:
        scores = await self._get("scores/live.json", {})
        games = ((scores.get("data") or {}).get("match") or [])
        out: List[LiveMatch] = []

        # fetch per-match stats in parallel
        sem = asyncio.Semaphore(6)

        async def fetch_stats(mid: str) -> Dict:
            async with sem:
                return await self._get("matches/stats.json", {"match_id": mid})

        tasks = {str(g.get("id")): asyncio.create_task(fetch_stats(str(g.get("id")))) for g in games}
        stats_res: Dict[str, Dict] = {}
        for mid, t in tasks.items():
            try:
                stats_res[mid] = await t
            except Exception:
                stats_res[mid] = {}

        for g in games:
            mid = str(g.get("id"))
            league = (g.get("competition_name") or g.get("league_name") or "League")
            home = g.get("home_name") or "Home"
            away = g.get("away_name") or "Away"
            minute = int(g.get("time") or 0)
            sh = int(g.get("home_score") or 0)
            sa = int(g.get("away_score") or 0)
            stat = ((stats_res.get(mid) or {}).get("data") or {}).get("match_stats") or {}
            da_s = stat.get("dangerous_attacks") or "0:0"
            try:
                da_h, da_a = da_s.split(":", 1)
                da_home, da_away = int(da_h), int(da_a)
            except Exception:
                da_home, da_away = 0, 0
            out.append(
                LiveMatch(
                    match_id=mid,
                    league=league,
                    home=home,
                    away=away,
                    minute=minute,
                    score_home=sh,
                    score_away=sa,
                    da_home=da_home,
                    da_away=da_away,
                    xg_home=None,
                    xg_away=None,
                    url="",
                )
            )
        return out


class SportmonksProvider:
    """Sportmonks livescores with statistics; xG requires add-on."""
    def __init__(self):
        if not SPORTMONKS_TOKEN:
            raise RuntimeError("SPORTMONKS_TOKEN not set")

    async def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Dict:
        url = f"{SPORTMONKS_BASE}{path}"
        q = {"api_token": SPORTMONKS_TOKEN}
        if params:
            q.update(params)
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=q)
            r.raise_for_status()
            return r.json()

    async def fetch_live(self) -> List[LiveMatch]:
        params = {"include": "participants;statistics;statistics.type"}
        data = await self._get("/livescores", params)
        fixtures = data.get("data") or []
        out: List[LiveMatch] = []
        for f in fixtures:
            minute = int((f.get("time") or {}).get("minute") or 0)
            scores = f.get("scores") or {}
            sh = int(scores.get("home_score") or 0)
            sa = int(scores.get("away_score") or 0)
            league_name = (f.get("league") or {}).get("name") or "League"
            parts = f.get("participants") or []
            home_name = parts[0].get("name") if parts else "Home"
            away_name = parts[1].get("name") if len(parts) > 1 else "Away"

            stats = f.get("statistics") or []

            def stat_val(type_ids: set, team_index: int) -> Optional[float]:
                for s in stats:
                    t_id = s.get("type_id")
                    if t_id in type_ids:
                        vals = s.get("value")
                        if isinstance(vals, str) and ":" in vals:
                            a, b = vals.split(":", 1)
                            pair = [a, b]
                            try:
                                return float(pair[team_index])
                            except Exception:
                                return None
                        if isinstance(vals, (int, float)):
                            return float(vals)
                        if isinstance(vals, dict):
                            key = "home" if team_index == 0 else "away"
                            if key in vals:
                                try:
                                    return float(vals[key])
                                except Exception:
                                    return None
                return None

            da_home = stat_val({44}, 0) or 0          # dangerous attacks id ~44
            da_away = stat_val({44}, 1) or 0
            xg_home = stat_val({5304}, 0)             # expected goals id may be 5304
            xg_away = stat_val({5304}, 1)

            out.append(
                LiveMatch(
                    match_id=str(f.get("id")),
                    league=league_name,
                    home=home_name,
                    away=away_name,
                    minute=minute,
                    score_home=sh,
                    score_away=sa,
                    da_home=int(da_home or 0),
                    da_away=int(da_away or 0),
                    xg_home=(float(xg_home) if xg_home is not None else None),
                    xg_away=(float(xg_away) if xg_away is not None else None),
                    url=str(f.get("id") or ""),
                )
            )
        return out


# ----------------- telegram bot -----------------
dp = Dispatcher()
engine = SignalEngine()

USER_TH: Dict[int, Thresholds] = {}   # per-user thresholds (in memory)
SENT_CACHE: Dict[int, set] = {}       # per-user sent match ids (in memory)


def get_th(uid: int) -> Thresholds:
    return USER_TH.get(uid) or Thresholds()


def set_th(uid: int, th: Thresholds) -> None:
    USER_TH[uid] = th


async def get_provider():
    if PROVIDER == "livescoreapi":
        return LiveScoreAPIProvider()
    if PROVIDER == "sportmonks":
        return SportmonksProvider()
    return MockProvider(Path(DATA_FILE))


async def poll_and_notify(chat_id: int, bot: Bot):
    await asyncio.sleep(1)
    SENT_CACHE.setdefault(chat_id, set())
    while True:
        try:
            provider = await get_provider()
            th = get_th(chat_id)
            matches = await provider.fetch_live()
            for m in matches:
                payload = engine.check(m, th)
                if not payload:
                    continue
                if payload["id"] in SENT_CACHE[chat_id]:
                    continue
                SENT_CACHE[chat_id].add(payload["id"])
                text = (
                    f"<b>{payload['league']}</b> — {payload['home']} vs {payload['away']}\n"
                    f"{payload['minute']}'  {payload['score']}\n"
                    f"Dangerous attacks: {payload['da']} (ratio={payload['ratio']:.2f})\n"
                    f"{payload['xg_text']}  | Leader: <b>{payload['leader']}</b>\n"
                    + (f"Link: {payload['url']}" if payload.get('url') else "")
                )
                await bot.send_message(chat_id, text)
        except Exception:
            # keep the loop running even if provider fails temporarily
            pass
        await asyncio.sleep(max(5.0, POLL_INTERVAL_SEC))


# ----------------- handlers -----------------
@dp.message(Command("start"))
async def cmd_start(m: Message, bot: Bot):
    if WHITELIST and str(m.from_user.id) not in WHITELIST:
        await m.answer("Access denied. Ask owner to add your chat_id to CHAT_WHITELIST.")
        return
    # start background poller once per chat
    key = f"_poll_{m.chat.id}"
    if not getattr(bot, key, False):
        asyncio.create_task(poll_and_notify(m.chat.id, bot))
        setattr(bot, key, True)
    await m.answer(
        "Bot is online. I will send signals when criteria match.\n"
        "Commands:\n"
        "/get_thresholds\n"
        "/set_thresholds attacks=2.0 xg_diff=0.8 xg_pct=60"
    )


@dp.message(Command("get_thresholds"))
async def cmd_get(m: Message):
    await m.answer(get_th(m.from_user.id).as_text())


@dp.message(Command("set_thresholds"))
async def cmd_set(m: Message):
    args = (m.text or "").split()[1:]
    cur = get_th(m.from_user.id)
    th = Thresholds(cur.attacks_ratio, cur.xg_diff, cur.xg_pct)
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            try:
                if k in ("attacks", "attacks_ratio", "ratio"):
                    th.attacks_ratio = float(v)
                elif k in ("xg_diff", "xgdiff", "diff"):
                    th.xg_diff = float(v)
                elif k in ("xg_pct", "xgpct", "pct"):
                    th.xg_pct = int(v)
            except ValueError:
                pass
    set_th(m.from_user.id, th)
    await m.answer("Updated:\n" + th.as_text())


@dp.message(F.text)
async def any_text(m: Message):
    if WHITELIST and str(m.from_user.id) not in WHITELIST:
        return
    await m.answer("Use /start, /get_thresholds or /set_thresholds.")


# ----------------- entrypoint -----------------
async def main():
    if not BOT_TOKEN:
        raise SystemExit("BOT_TOKEN not set")
    bot = Bot(
        BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    # auto-create mock file for quick demo
    if PROVIDER == "mock" and not Path(DATA_FILE).exists():
        Path(DATA_FILE).write_text(json.dumps({
            "matches": [
                {
                    "id": "12345",
                    "league": "Premier League",
                    "home": "Arsenal",
                    "away": "Chelsea",
                    "minute": 80,
                    "score_home": 0,
                    "score_away": 0,
                    "dangerous_attacks_home": 90,
                    "dangerous_attacks_away": 40,
                    "xg_home": 1.5,
                    "xg_away": 0.4,
                    "match_url": "https://example.com/match/12345"
                }
            ]
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    asyncio.run(main())
