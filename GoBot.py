import os
import re
import json
import asyncio
import unicodedata
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List
import configparser
import aiohttp
import discord
import random
from discord import app_commands
from discord.ext import commands
from discord.errors import Forbidden
from datetime import datetime, timedelta, timezone
import math
from collections import defaultdict, Counter
from zoneinfo import ZoneInfo

GITHUB_IMG_BASE = "https://raw.githubusercontent.com/hmxmilohax/GoBot/main/images"
DEFAULT_SONG_ART = f"{GITHUB_IMG_BASE}/_default.png"
_ART_URL_CACHE: Dict[str, str] = {}

SONG_MAP_PATH = Path(__file__).resolve().parent / "song_map.dta"
LEADERBOARDS_URL = "https://gocentral-service.rbenhanced.rocks/leaderboards"
BATTLES_URL = "https://gocentral-service.rbenhanced.rocks/battles"
BATTLE_LB_URL = "https://gocentral-service.rbenhanced.rocks/leaderboards/battle"
TOP_N = 10
BAND_TOP_N = 15
MANAGER_STATE_PATH = Path(__file__).resolve().parent / "battle_manager_state.json"
AUTO_BATTLE_PERIOD_DAYS = 1
BATTLE_DURATION_DAYS = 7
SUBSCRIPTIONS_LOOP_SECONDS = 60
SUBS_KEY = "battle_subscriptions"   # { "<battle_id>": [user_id, ...] }
TOPS_KEY = "battle_top_seen"        # { "<battle_id>": {"name": "...", "score": 123} }
OVERTAKES_KEY = "battle_overtakes"  # NEW: { "<battle_id>": int }

BASE_URL = "https://gocentral-service.rbenhanced.rocks"

CREATE_BATTLE_URL = f"{BASE_URL}/admin/battles/create"
DELETE_BATTLE_URL = f"{BASE_URL}/admin/battles"
ADMIN_USER_ID = 960524988824313876 #jnack

# Champion role for weekly winners
CHAMPION_ROLE_ID = 1411137812274872412
CHAMPION_GRANTS_KEY = "champion_role_grants"  # [{user_id, role_id, battle_id, granted_at, expires_at, removed_at, active}]

WINNERS_BATCH_WINDOW_SECONDS = 300   # group endings within 5 minutes into one announcement
RESULTS_GRACE_SECONDS = 5            # small buffer after expiry before fetching results
NEXT_WEEK_GAP_SECONDS = 3600         # <-- wait 1 hour before next week's post

STOPWORDS = {
    "the","and","a","of","to","in","on","for","feat","ft","vs","with","&"
}

# --- Locale (subset used by the vote UI) ---
LOCALE = {
    "dx_play_a_show": "Play a Show",
    "dx_play_a_show_info": "Vote on which songs to play!\nVoting ends when all players have voted!",
    "play_a_show_bandvote": "Band Vote:",
    "play_a_show_genre": "A %s song",
    "play_a_show_year": "A song from %i",
    "play_a_show_year_decade": "A song from the %is",
    "play_a_show_album": "A song from %s",  # (ignoring <alt> artist </alt> in Discord)
    "play_a_show_artist": "A song by %s",
    "play_a_show_diff": "A %s song on %s",
    "play_a_show_diff_an": "An %s song on %s",
    "play_a_show_fast": "A fast song",
    "play_a_show_med": "A completely random song",
    "play_a_show_slow": "A slow song",
    "play_a_show_long": "A long song",
    "play_a_show_really_long": "A REALLY long song",
    "play_a_show_short": "A short song",
    "play_a_show_really_short": "A REALLY short song",
    "play_a_show_source": "A song from %s",
    "play_a_show_author": "A song charted by %s",
    "play_a_show_missing_inst": "this song is missing one of your instruments",
}

LOCALE_SOURCES = {
    # Official Sources — Beta
    "gh1_beta": "Guitar Hero Beta",
    "gh2_beta": "Guitar Hero II Beta",
    "gh80s_beta": "Guitar Hero Encore: Rocks the 80s Beta",
    "gh3_beta": "Guitar Hero III: Legends of Rock Beta",
    "gh5_beta": "Guitar Hero 5 Beta",
    "djhero_beta": "DJ Hero Beta",
    "gh6_beta": "Guitar Hero 6",
    "djhero2_beta": "DJ Hero 2 Beta",
    "rb1_beta": "Rock Band 1 Beta",
    "rb2_beta": "Rock Band 2 Beta",
    "rb3_beta": "Rock Band 3 Beta",

    # Guitar Hero games
    "gh1": "Guitar Hero",
    "gh2": "Guitar Hero II",
    "gh2dlc": "Guitar Hero II DLC",
    "gh80s": "Guitar Hero Encore: Rocks the 80s",
    "gh3": "Guitar Hero III: Legends of Rock",
    "gh3dlc": "Guitar Hero III DLC",
    "ghot": "Guitar Hero: On Tour",
    "gha": "Guitar Hero: Aerosmith",
    "ghwt": "Guitar Hero: World Tour",
    "ghwtdlc": "Guitar Hero: World Tour DLC",
    "ghotd": "Guitar Hero On Tour: Decades",
    "ghm": "Guitar Hero: Metallica",
    "ghmdlc": "Death Magnetic DLC",
    "ghsh": "Guitar Hero: Smash Hits",
    "gh5": "Guitar Hero 5",
    "gh5dlc": "Guitar Hero 5 DLC",
    "djhero": "DJ Hero",
    "djherodlc": "DJ Hero DLC",
    "bh": "Band Hero",
    "bh2": "Band Hero 2",
    "bhds": "Band Hero DS",
    "ghvh": "Guitar Hero: Van Halen",
    "ghotmh": "Guitar Hero On Tour: Modern Hits",
    "ghwor": "Guitar Hero: Warriors of Rock",
    "ghwordlc": "Guitar Hero: Warriors of Rock DLC",
    "djhero2": "DJ Hero 2",
    "djhero2dlc": "DJ Hero 2 DLC",
    "ghl": "Guitar Hero Live",
    "ghtv": "Guitar Hero TV",
    "gh": "Guitar Hero Series",
    "ghdlc": "Guitar Hero Series DLC",

    # Harmonix games
    "acs": "A City Sleeps",
    "amp03": "Amplitude (2003)",
    "amp16": "Amplitude (2016)",
    "antigrav": "EyeToy: Antigrav",
    "audica": "Audica",
    "flux": "Fantasia",
    "fnfestival": "Fortnite Festival",
    "freq": "Frequency",
    "ham1": "Dance Central",
    "ham2": "Dance Central 2",
    "ham3": "Dance Central 3",
    "kr_cmt": "CMT Presents: Karaoke Revolution Country",
    "kr_party": "Karaoke Revolution Party",
    "kr_vol_1": "Karaoke Revolution (2003)",
    "kr_vol_2": "Karaoke Revolution Vol. 2",
    "kr_vol_3": "Karaoke Revolution Vol. 3",
    "kr": "Karaoke Revolution Series",
    "viper": "Dance Central Spotlight",

    # Rock Band games
    "dlc": "Rock Band 1/2 DLC",
    "rb1": "Rock Band 1",
    "rb2": "Rock Band 2",
    "rb3": "Rock Band 3",
    "lego": "LEGO: Rock Band",
    "greenday": "Green Day Rock Band",
    "greendaydlc": "Green Day Rock Band DLC",
    "rb1_dlc": "Rock Band 1/2 DLC",   # must exist for Play a Show
    "ugc1": "Rock Band Network 1.0",
    "rbtp_vol_1": "Track Pack Vol. 1",
    "rbtp_acdc": "AC/DC Live Track Pack",
    "rbtp_vol_2": "Track Pack Vol. 2",
    "rbtp_classic_rock": "Classic Rock Track Pack",
    "rbtp_country_1": "Country Track Pack",
    "beatles": "The Beatles: Rock Band",
    "beatles_dlc": "The Beatles: Rock Band DLC",
    "rbtp_metal": "Metal Track Pack",
    "rb3dlc": "Rock Band 3 DLC",
    "rbtp_country_2": "Country Track Pack 2",
    "ugc2": "Rock Band Network 2.0",
    "blitz": "Rock Band Blitz",
    "pearljam": "Pearl Jam: Rock Band",
    "rbjapan": "Rock Band Japan",
    "rb4": "Rock Band 4",
    "rb4dlc": "Rock Band 4 DLC",
    "rb4_dlc": "Rock Band 4 DLC",
    "rbvr": "Rock Band VR",
    "rb4_rivals": "Rock Band Rivals",
    "lost_dlc": "Lost Rock Band DLC",

    # Third-Party games
    "gd": "GITADORA",
    "gf1": "GuitarFreaks",
    "gf2dm1": "GuitarFreaks 2ndMIX & DrumMania",
    "pg": "PowerGig: Rise of the SixString",
    "pgdlc": "PowerGig: Rise of the SixString DLC",
    "rr": "Rock Revolution",
    "rrdlc": "Rock Revolution DLC",

    # Unofficial Sources — Charters
    "ataeaf": "ataeaf's Charts",
    "ganso": "Ganso's Charts",
    "goulart": "raphaelgoulart's Charts",
    "iasg14": "IaSg14's Charts",
    "linosrb": "Linos' Charts",
    "onyxite": "Onyxite Charts",
    "pistolofrb6": "PistolOfRB6's Charts",
    "plumato": "Plumato's Charts",
    "ruggy": "Ruggy's Customs",
    "sog": "SomeOldGuys' Charts",
    "sygenysis": "Sygenysis' Charts",
    "ted": "thardwardy's Charts",
    "3rs": "ThreeAreEss' Charts",

    # Community sources
    "a7xmegapack": "Avenged Sevenfold Mega Pack",
    "customs": "Custom Songs",
    "c3customs": "C3 Customs",
    "c3legacy": "Legacy C3 Customs",
    "euterpe": "Project Euterpe",
    "finnish": "Suomibiisit",
    "jrb": "J-Rock Band",
    "lanebreakers": "Lane Breakers",
    "lanebreakersbonus": "Lane Breakers Bonus",
    "lanebreakersdlc": "Lane Breakers DLC",
    "milohax": "MiloHax Customs",
    "paramoremegapack": "Paramore Mega Pack",
    "rbbr": "Rock Band Brasil",
    "rbee": "Rock Band en Español",
    "rbrc": "Rock Band Road Crew",
    "tbrbcdlc": "The Beatles: Rock Band Custom DLC",
    "ugc": "Custom Songs",
    "ugc_c3": "C3 Customs",
    "ugc_lost": "Lost Rock Band Network",
    "ugc_plus": "Custom Songs",
    "ugc_rv": "RhythmVerse Customs",

    # Fan-Made games
    "ch": "Clone Hero",
    "fof": "Frets on Fire",
    "fretsmasher": "Fret Smasher",
    "phaseshift": "Phase Shift",
    "yarg": "YARG",
    "yargdlc": "YARG DLC",
}

# Human-readable names for genres (genre / subgenre)
LOCALE_GENRES = {
    # Core
    "alternative": "Alternative",
    "bluegrass": "Bluegrass",
    "blues": "Blues",
    "classic": "Classic",
    "classical": "Classical",
    "classicrock": "Classic Rock",
    "country": "Country",
    "disco": "Disco",
    "emo": "Emo",
    "experimental": "Experimental",
    "fusion": "Fusion",
    "glam": "Glam",
    "hardrock": "Hard Rock",
    "industrial": "Industrial",
    "inspirational": "Inspirational",
    "jazz": "Jazz",
    "jrock": "J-Rock",
    "latin": "Latin",
    "metal": "Metal",
    "novelty": "Novelty",
    "numetal": "Nu-Metal",
    "pop": "Pop",
    "poprock": "Pop-Rock",
    "prog": "Prog",
    "progrock": "Prog Rock",
    "psychadelic": "Psychedelic",
    "punk": "Punk",
    "rock": "Rock",
    "rockabilly": "Rockabilly",
    "rockandroll": "Rock 'n' Roll",
    "southernrock": "Southern Rock",
    "grunge": "Grunge",
    "indierock": "Indie Rock",
    "new_wave": "New Wave",
    "reggaeska": "Reggae/Ska",
    "rbsoulfunk": "R&B/Soul/Funk",
    "hiphoprap": "Hip-Hop/Rap",
    "hiphop": "Hip Hop",
    "other": "Other",
    "popdanceelectronic": "Pop/Dance/Electronic",
    "urban": "Urban",
    "world": "World",
    "surfrock": "Surf Rock",
    "trap": "Trap",

    # Extended / BR set
    "axebr": "Axé",
    "bigband": "Big Band",
    "blackmetal": "Black Metal",
    "bossanova": "Bossa Nova",
    "bregatechnobrega": "Brega/Techno Brega",
    "forro": "Forró",
    "funkbr": "Brazilian Funk",
    "deathmetal": "Death Metal",
    "dubstep": "Dubstep",
    "funkmelody": "Funk Melody",
    "gospel": "Gospel",
    "hairmetal": "Hair Metal",
    "hardcorepunk": "Hardcore Punk",
    "metalcore": "Metalcore",
    "mpb": "MPB",
    "pagode": "Pagode",
    "poppunk": "Pop Punk",
    "posthardcore": "Post-Hardcore",
    "psychobilly": "Psychobilly",
    "raprock": "Rap-Rock",
    "samba": "Samba",
    "sertanejo": "Sertanejo",
    "sertanejo_universitario": "Sertanejo Universitário",
    "skapunk": "Ska Punk",
    "speedmetal": "Speed Metal",

    # Subgenre aliases
    "subgenre_axebr": "Axé",
    "subgenre_bossanova": "Bossa Nova",
    "subgenre_brega": "Brega",
    "subgenre_technobrega": "Techno Brega",
    "subgenre_forro": "Forró",
    "subgenre_funkbr": "Brazilian Funk",
    "subgenre_funkmelody": "Funk Melody",
    "subgenre_gospel": "Gospel",
    "subgenre_mpb": "MPB",
    "subgenre_pagode": "Pagode",
    "subgenre_poprock": "Pop-Rock",
    "subgenre_samba": "Samba",
    "subgenre_sertanejo": "Sertanejo",
    "subgenre_sertanejo_universitario": "Sertanejo Universitário",
}

# Merge into LOCALE with explicit prefixes for easy lookup/override
LOCALE.update({f"source_{k}": v for k, v in LOCALE_SOURCES.items()})
LOCALE.update({f"genre_{k}": v for k, v in LOCALE_GENRES.items()})

def pretty_source(code: Optional[str]) -> str:
    """Localized display for game_origin values."""
    if not code:
        return "—"
    k = str(code).strip().lower()
    return LOCALE.get(f"source_{k}", LOCALE_SOURCES.get(k, k.replace("_", " ").title()))

def pretty_genre(code: Optional[str]) -> str:
    """Localized display for genre / subgenre values."""
    if not code:
        return "—"
    k = str(code).strip().lower()
    return LOCALE.get(f"genre_{k}", LOCALE_GENRES.get(k, k.replace("_", " ").title()))

def loc(key: str, *args):
    s = LOCALE.get(key, key)
    try:
        return (s % args) if args else s
    except Exception:
        return s


# Source policy
EXCLUDED_SOURCES = {"fnfestival", "beatles"}
UGC_DOWNWEIGHT_SOURCES = {"ugc1", "ugc2"}
UGC_WEIGHT_MULTIPLIER = 0.4

def get_api_key() -> str:
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config.get("api", "key", fallback=None)

def _load_discord_ids():
    cfg = configparser.ConfigParser()
    cfg.read("config.ini")
    return {
        "admin_user_id": cfg.getint("discord", "admin_id", fallback=ADMIN_USER_ID),
        "announce_channel_id": cfg.getint("discord", "announce_channel_id", fallback=None),
        "ping_role_id": cfg.getint("discord", "ping_role_id", fallback=None),
        "overtakes_channel_id": cfg.getint("discord", "overtakes_channel_id", fallback=None),
        "guild_id": cfg.getint("guild", "id", fallback=cfg.getint("discord", "guild_id", fallback=None)),
    }

INSTR_ROLE_IDS: Dict[str, int] = {
    # Plastic instruments
    "drums": 0,
    "bass": 1,
    "guitar": 2,
    "vocals": 3,
    "harmony": 4,
    "keys": 5,

    # Pro instruments
    "prodrums": 6,
    "proguitar": 7,
    "probass": 8,
    "prokeys": 9,

    # Band score
    "band": 10,
}

INSTR_BY_ROLE_ID: Dict[int, str] = {v: k for k, v in INSTR_ROLE_IDS.items()}
INSTR_SET_WEEK = ["guitar", "bass", "drums", "vocals", "band"]
PRO_ONE_PER_WEEK = ["proguitar", "probass", "prokeys", "prodrums", "harmony", "keys"]
_ALIAS_TO_BASE = {"prodrums": "drums", "harmony": "vocals"}
LINKS_KEY = "player_links"  # { "<discord_user_id>": [display_name, ...] }
ALLOWED_RANDOM_INSTRUMENTS = [
    k for k in INSTR_ROLE_IDS.keys()
    if k not in ("proguitar", "probass", "prokeys", "keys")
]

INSTR_ALIASES: Dict[str, str] = {
    # Plastic Guitar
    "gtr": "guitar",
    "g": "guitar",
    "lead": "guitar",
    "guitar": "guitar",

    # Pro Guitar
    "pg": "proguitar",
    "real guitar": "proguitar",
    "pro guitar": "proguitar",
    "proguitar": "proguitar",

    # Plastic Bass
    "b": "bass",
    "bass": "bass",

    # Pro Bass
    "pb": "probass",
    "probass": "probass",
    "pro bass": "probass",
    "real bass": "probass",

    # Plastic Drums
    "drum": "drums",

    # Pro Drums
    "pd": "prodrums",
    "prodrums": "prodrums",
    "realdrum": "prodrums",
    "real drum": "prodrums",
    "real drums": "prodrums",
    "pro drums": "prodrums",

    # Vocals
    "v": "vocals",
    "vox": "vocals",
    "vocals": "vocals",
    "sing": "vocals",
    "singing": "vocals",

    # Harmonies
    "h": "harmony",
    "harm": "harmony",
    "harms": "harmony",
    "harm1": "harmony",
    "harm2": "harmony",
    "harm3": "harmony",

    # Plastic Keys
    "keyboard": "keys",
    "keys": "keys",
    "k": "keys",

    # Pro Keys
    "pk": "prokeys",
    "prokeys": "prokeys",
    "realkeys": "prokeys",
    "real keys": "prokeys",
    "pro keys": "prokeys",

    # Band
    "bandscore": "band",
    "band": "band",
}

INSTR_DISPLAY_NAMES = {
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "vocals": "Vocals",
    "harmony": "Harmony",
    "keys": "Keys",
    "prodrums": "Pro Drums",
    "proguitar": "Pro Guitar",
    "probass": "Pro Bass",
    "prokeys": "Pro Keys",
    "band": "Band",
}

DIFF_LEVELS = ("Warmup", "Apprentice", "Solid", "Challenging", "Nightmare", "Impossible")

DIFF_THRESHOLDS = {
    "guitar":     [139, 176, 221, 267, 333, 409],
    "drums":      [124, 151, 178, 242, 345, 448],
    "bass":       [135, 181, 228, 293, 364, 436],
    "vocals":     [132, 175, 218, 279, 353, 427],
    "band":       [163, 215, 243, 267, 292, 345],
    "keys":       [153, 211, 269, 327, 385, 443],
    "prokeys":    [153, 211, 269, 327, 385, 443],
    "proguitar":  [150, 205, 264, 323, 382, 442],
    "probass":    [150, 208, 267, 325, 384, 442],
}

DIFF_THRESHOLDS["harmony"] = DIFF_THRESHOLDS["vocals"]
DIFF_THRESHOLDS.setdefault("prodrums", DIFF_THRESHOLDS["drums"])
DIFF_MAP = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Expert"}
DIFF_LETTER = {1: "E", 2: "M", 3: "H", 4: "X"}

DIFF_BUCKET_WEIGHT = [0.7, 0.85, 1.0, 1.35, 1.5, 1.5]  # tweak to taste
GENRE_BOOST = 1.25          # small boost for unused genre
REPEAT_BUCKET_PENALTY = 0.85  # soft penalty per time a bucket was already used (0.85^count)

# Pretty bits for embeds
INSTR_EMOJI = {
    "guitar": "🎸",
    "bass": "🎸",
    "drums": "🥁",
    "vocals": "🎤",
    "band": "👥",
    "proguitar": "🎸",
    "probass": "🎸",
    "prokeys": "🎹",
    "prodrums": "🥁",
    "harmony": "🎤",
    "keys": "🎹",
}

INSTR_COLOR = {
    "guitar": 0xF59E0B,  # amber
    "bass":   0x10B981,  # emerald
    "drums":  0xEF4444,  # red
    "vocals": 0x8B5CF6,  # violet
    "band":   0x3B82F6,  # blue
    "prodrums": 0xEF4444,
    "proguitar": 0xF59E0B,
    "harmony":  0x8B5CF6,
    "keys":     0x22C55E,
    "probass": 0x10B981,
    "prokeys": 0x22C55E,
}

# Map rank keys from the DTA to our instrument keys
RANK_KEY_MAP = {
    "drum": "drums",
    "guitar": "guitar",
    "bass": "bass",
    "vocals": "vocals",
    "keys": "keys",
    "real_keys": "prokeys",
    "real_guitar": "proguitar",
    "real_bass": "probass",
    "band": "band",
}


def _to_unix_ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _base_part(instr_key: str) -> str:
    return _ALIAS_TO_BASE.get(instr_key, instr_key)

def _as_dt(v) -> Optional[datetime]:
    """Accepts epoch int/float, digit string, or ISO8601 → timezone-aware UTC datetime."""
    if isinstance(v, (int, float)):
        return datetime.utcfromtimestamp(int(v)).replace(tzinfo=timezone.utc)
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return datetime.utcfromtimestamp(int(s)).replace(tzinfo=timezone.utc)
        return _from_iso(s)
    return None

def _as_int(v) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None

@dataclass
class Song:
    slug: str
    name: str
    artist: str
    song_id: int
    year: Optional[int] = None
    album_name: Optional[str] = None
    genre: Optional[str] = None
    source: Optional[str] = None
    author: Optional[str] = None
    ranks: Dict[str, int] = field(default_factory=dict)

class SongIndex:
    def __init__(self, songs: List[Song]):
        self.songs = songs

        # Fast lookups
        self.by_norm_name = {self._norm(s.name): s for s in songs}
        self.by_id = {s.song_id: s for s in songs}

        # Shortname/slug normalization (allow hyphens/underscores)
        self.by_slug: Dict[str, Song] = {}
        for s in songs:
            slug_norm = _sanitize_shortname(s.slug)
            self.by_slug[slug_norm] = s

        # Inverted index over title, slug, and artist (token -> set of slugs)
        self._inv_title: Dict[str, set[str]] = defaultdict(set)
        self._inv_slug: Dict[str, set[str]] = defaultdict(set)

        # Keep tokenized fields around for scoring
        self._title_tokens: Dict[str, list[str]] = {}
        self._slug_tokens: Dict[str, list[str]] = {}

        for s in songs:
            slug_key = _sanitize_shortname(s.slug)

            title_norm = self._norm(s.name)
            slug_norm = _sanitize_shortname(s.slug).replace("-", " ").replace("_", " ")

            t_title = _tokenize(title_norm)
            t_slug  = _tokenize(slug_norm)

            self._title_tokens[slug_key] = t_title
            self._slug_tokens[slug_key] = t_slug

            for t in set(t_title):
                self._inv_title[t].add(slug_key)
            for t in set(t_slug):
                self._inv_slug[t].add(slug_key)

        # Precompute IDF per token based on title+slug+artist df
        N = max(1, len(songs))
        self._idf: Dict[str, float] = {}
        all_tokens = set(self._inv_title) | set(self._inv_slug)
        for t in all_tokens:
            df = len(self._inv_title.get(t, set()) | self._inv_slug.get(t, set()))
            self._idf[t] = math.log1p(N / max(1, df))

    @staticmethod
    def _strip_accents(s: str) -> str:
        nk = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nk if not unicodedata.combining(c))

    @classmethod
    def _norm(cls, s: str) -> str:
        s = cls._strip_accents(s).lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    def _score_candidate(self, slug_key: str, q_tokens: list[str], q_norm: str, q_is_phrase: bool) -> float:
        # Field weights
        W_TITLE, W_SLUG = 1.6, 1.3

        t_title = self._title_tokens.get(slug_key, [])
        t_slug  = self._slug_tokens.get(slug_key, [])

        c_title = Counter(t_title)
        c_slug  = Counter(t_slug)

        score = 0.0
        matched_tokens = 0

        for t in q_tokens:
            idf = self._idf.get(t, 0.0)
            title_tf = c_title.get(t, 0)
            slug_tf  = c_slug.get(t, 0)

            if title_tf or slug_tf:
                matched_tokens += 1

            score += idf * (
                W_TITLE * min(1, title_tf) +
                W_SLUG  * min(1, slug_tf)
            )

            # word-prefix bonus (e.g., "tim" matching "timmy") in TITLE only
            if not title_tf and any(tok.startswith(t) for tok in t_title):
                score += 0.2 * (idf + 0.1)

        # phrase / substring bonuses on normalized strings
        song = self.by_slug[slug_key]
        name_norm = self._norm(song.name)
        slug_phrase = _sanitize_shortname(song.slug).replace("-", " ").replace("_", " ")

        if name_norm == q_norm:
            score += 5.0
        if q_is_phrase and q_norm and q_norm in name_norm:
            score += 2.5
        if q_norm and q_norm in name_norm:
            score += 0.75

        # shortname bonuses
        if q_norm and q_norm == slug_phrase:
            score += 4.0
        if q_norm and slug_phrase.startswith(q_norm):
            score += 2.0

        # coverage boost (prefer candidates that hit more of the query)
        if q_tokens:
            coverage = matched_tokens / len(q_tokens)
            score *= (1.0 + 0.25 * coverage)

        return score


    def find_all(self, query: str, max_results: int = 5) -> List[Song]:
        if not query or not query.strip():
            return []

        # 1) SONG ID direct hit
        if _is_numeric(query):
            sid = int(query)
            s = self.by_id.get(sid)
            return [s] if s else []

        # 2) quoted exact-title phrase?
        exact_q = None
        m = re.search(r'"([^"]+)"', query)
        if m:
            exact_q = m.group(1)

        raw_q = exact_q if exact_q else query
        q_norm = self._norm(raw_q)
        q_tokens = _tokenize(q_norm)
        q_is_phrase = exact_q is not None

        # allow slug tokens only for sluggy queries or single-token queries
        is_slug_like = bool(re.fullmatch(r"[a-z0-9_\-]+", query.strip()))
        allow_slug_tokens = is_slug_like or (len(q_tokens) == 1)

        # 3) Shortname exact/prefix (treat a-z0-9_- as shortname-y)
        shorty = re.fullmatch(r"[a-zA-Z0-9_\-]+", query.strip())
        if shorty:
            q_slug = _sanitize_shortname(query)
            # exact slug
            s = self.by_slug.get(q_slug)
            if s:
                return [s]
            # prefix on slug string
            slug_prefix_hits = [self.by_slug[k] for k in self.by_slug.keys() if k.startswith(q_slug)]
            if slug_prefix_hits:
                # still rank them a bit by phrase proximity
                ranked = sorted(
                    slug_prefix_hits,
                    key=lambda x: (0 if _sanitize_shortname(x.slug) == q_slug else 1,
                                   -len(os.path.commonprefix([_sanitize_shortname(x.slug), q_slug])) )
                )
                return ranked[:max_results]

        # 4) Title exact
        if q_norm in self.by_norm_name:
            return [self.by_norm_name[q_norm]]

        # 5) Build candidate set from inverted indexes
        candidates: set[str] = set()
        for t in q_tokens:
            # title and shortname tokens are strongest
            candidates |= self._inv_title.get(t, set())
            if allow_slug_tokens:
                candidates |= self._inv_slug.get(t, set())

        # If nothing yet and we have a single token, try prefix-expansion on title tokens
        if not candidates and len(q_tokens) == 1:
            t = q_tokens[0]
            # broaden by any title token that starts with the query token
            for tok, slugs in self._inv_title.items():
                if tok.startswith(t):
                    candidates |= slugs

        # Also allow substring-in-title as last resort candidate builder
        if not candidates and q_norm:
            for k, s in self.by_slug.items():
                if q_norm in self._norm(s.name):
                    candidates.add(k)

        # 6) Score & sort candidates
        scored: list[tuple[float, Song]] = []
        for k in candidates:
            if len(q_tokens) >= 2:
               title_toks = self._title_tokens.get(k, [])
               if not any(t in title_toks for t in q_tokens):
                   continue
            score = self._score_candidate(k, q_tokens, q_norm, q_is_phrase)
            if score > 0:
                scored.append((score, self.by_slug[k]))

        # If still nothing decent, do a **constrained** fuzzy pass on names & slugs
        if not scored:
            # Limit fuzz to top 400 by rough name-length proximity to keep it cheap and precise
            approx_pool = sorted(
                self.songs,
                key=lambda s: abs(len(self._norm(s.name)) - len(q_norm))
            )[:400]

            for s in approx_pool:
                name_norm = self._norm(s.name)
                slug_norm = _sanitize_shortname(s.slug)
                sim = max(
                    difflib.SequenceMatcher(None, q_norm, name_norm).ratio(),
                    difflib.SequenceMatcher(None, q_norm, slug_norm).ratio()
                )
                # tighter threshold to avoid garbage
                if sim >= 0.76 or (len(q_norm) >= 6 and sim >= 0.72):
                    # small boost for word-boundary presence
                    bonus = 0.4 if q_norm in name_norm else 0.0
                    scored.append((sim + bonus, s))

        # Dedup by slug and return
        seen = set()
        out: List[Song] = []
        for score, s in sorted(scored, key=lambda t: -t[0]):
            if s.slug in seen:
                continue
            seen.add(s.slug)
            out.append(s)
            if len(out) >= max_results:
                break
        return out


def _iter_top_level_blocks(text: str):
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _parse_block(block: str) -> Optional[Song]:
    j = 1
    while j < len(block) and block[j].isspace():
        j += 1
    k = j
    while k < len(block) and (not block[k].isspace()) and block[k] not in "()":
        k += 1
    slug = block[j:k]

    def rx(p: str) -> Optional[str]:
        m = re.search(p, block, re.DOTALL)
        return m.group(1).strip() if m else None

    name   = rx(r'\(name\s+"([^"]+)"\)')
    artist = rx(r'\(artist\s+"([^"]+)"\)') or ""
    album  = rx(r'\(album_name\s+"([^"]+)"\)')
    year   = rx(r'\(year_released\s+(\d+)\)')
    song_id = rx(r'\(song_id\s+(\d+)\)')
    genre  = rx(r'\(genre\s+([a-zA-Z0-9_]+)\)') or rx(r'\(genre\s+"([^"]+)"\)')
    source = rx(r'\(game_origin\s+([a-zA-Z0-9_]+)\)') or rx(r'\(game_origin\s+"([^"]+)"\)')
    author = rx(r'\(author\s+"([^"]+)"\)')

    ranks: Dict[str, int] = {}
    m_rank = re.search(r'\(rank\s*(\([^()]+\)\s*)+\)', block, re.DOTALL)
    if m_rank:
        inside = m_rank.group(0)
        for inst, val in re.findall(r'\(([a-zA-Z_]+)\s+(\d+)\)', inside):
            key = RANK_KEY_MAP.get(inst)
            if key:
                ranks[key] = int(val)

    if song_id and name:
        return Song(
            slug=slug,
            name=name,
            artist=artist,
            song_id=int(song_id),
            year=int(year) if year else None,
            album_name=album,
            genre=genre,
            source=source,
            author=author,
            ranks=ranks,
        )
    return None
    
def _is_numeric(s: str) -> bool:
    s = s.strip()
    return s.isdigit()

def _tokenize(norm_text: str) -> list[str]:
    # norm_text should already be lowercased, ascii, and space-separated
    toks = [t for t in norm_text.split() if t and t not in STOPWORDS]
    return toks

def _sanitize_shortname(x: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "", (x or "").strip().lower())

def get_song_art_url_fast(shortname: str) -> str:
    key = _sanitize_shortname(shortname)

    # If we've already decided, return it.
    if key in _ART_URL_CACHE:
        return _ART_URL_CACHE[key]

    # Optimistic candidate; cache it immediately.
    candidate = f"{GITHUB_IMG_BASE}/{key}.png"
    _ART_URL_CACHE[key] = candidate
    return candidate

async def _verify_art_url(session: aiohttp.ClientSession, shortname: str):
    key = _sanitize_shortname(shortname)
    url = f"{GITHUB_IMG_BASE}/{key}.png"
    try:
        async with session.head(url, allow_redirects=True,
                                timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if not (200 <= resp.status < 300):
                _ART_URL_CACHE[key] = DEFAULT_SONG_ART
    except Exception:
        _ART_URL_CACHE[key] = DEFAULT_SONG_ART

async def fetch_song_art_url(session: aiohttp.ClientSession, shortname: str) -> str:
    """Return raw GitHub PNG for the song shortname if it exists; else default."""
    key = _sanitize_shortname(shortname)
    if key in _ART_URL_CACHE:
        return _ART_URL_CACHE[key]

    candidate = f"{GITHUB_IMG_BASE}/{key}.png"
    try:
        async with session.head(candidate, allow_redirects=True,
                                timeout=aiohttp.ClientTimeout(total=5)) as resp:
            url = candidate if 200 <= resp.status < 300 else DEFAULT_SONG_ART
    except Exception:
        url = DEFAULT_SONG_ART

    _ART_URL_CACHE[key] = url
    return url

def difficulty_label(instr_key: str, rank_val: Optional[int]) -> str:
    """Return e.g. 'Challenging' from a song's numeric rank for the given instrument."""
    if not rank_val or rank_val <= 0:
        return "—"
    th = DIFF_THRESHOLDS.get(instr_key)
    if not th:
        return "—"
    # thresholds are lower-bounds for each label
    for i in range(len(th) - 1, -1, -1):
        if rank_val >= th[i]:
            return DIFF_LEVELS[i]
    return DIFF_LEVELS[0]

def difficulty_bucket(instr_key: str, rank_val: Optional[int]) -> Optional[int]:
    """Return 0..5 bucket index for rank_val on instr_key (None if no part)."""
    if not rank_val or rank_val <= 0:
        return None
    th = DIFF_THRESHOLDS.get(instr_key)
    if not th:
        return None
    for i in range(len(th) - 1, -1, -1):
        if rank_val >= th[i]:
            return i
    return 0

def difficulty_bucket_index(instr_key: str, rank_val: Optional[int]) -> Optional[int]:
    """Return difficulty bucket index 0..5 from thresholds, or None if no part."""
    if not rank_val or rank_val <= 0:
        return None
    th = DIFF_THRESHOLDS.get(instr_key)
    if not th:
        return None
    for i in range(len(th) - 1, -1, -1):
        if rank_val >= th[i]:
            return i
    return 0

def _th_key(instr: str) -> str:
    return {"harmony": "vocals", "prodrums": "drums"}.get(instr, instr)

def _rank(song: Song, instr: str) -> Optional[int]:
    return (song.ranks or {}).get(_base_part(instr))

def has_part(song: Song, instr_key: str) -> bool:
    v = (song.ranks or {}).get(_base_part(instr_key))
    return isinstance(v, int) and v > 0

def load_song_map(path: Path) -> SongIndex:
    text = path.read_text(encoding="utf-8")
    songs: List[Song] = []
    for block in _iter_top_level_blocks(text):
        s = _parse_block(block)
        if s:
            songs.append(s)
    return SongIndex(songs)


def resolve_instrument(instr_raw: str) -> Optional[str]:
    x = instr_raw.strip().lower()
    x = SongIndex._norm(x)
    if x in INSTR_ROLE_IDS:
        return x
    if x in INSTR_ALIASES:
        return INSTR_ALIASES[x]
    # fuzzy alias lookup
    all_keys = list(INSTR_ROLE_IDS.keys()) + list(INSTR_ALIASES.keys())
    best = None
    best_score = 0.0
    for key in all_keys:
        score = difflib.SequenceMatcher(None, x, key).ratio()
        if score > best_score:
            best, best_score = key, score
    if best:
        return INSTR_ALIASES.get(best, best)
    return None

async def fetch_battle_top_winner(session: aiohttp.ClientSession, battle_id: int) -> Optional[dict]:
    rows = await fetch_battle_leaderboard(session, battle_id, page_size=1)
    if not rows:
        return None
    r = rows[0]
    r["_score_str"] = f"{int(r.get('score', 0)):,.0f}" if isinstance(r.get('score'), (int, float)) else "?"
    # strip platform/fluff tags like [360] [PS3] [RPCS3] [WII] from display
    r["_name"] = _strip_platform_tags(str(r.get("name", "Unknown")))
    return r

async def fetch_battles(session: aiohttp.ClientSession) -> Optional[List[dict]]:
    try:
        async with session.get(BATTLES_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("battles", [])
    except Exception:
        return None


async def fetch_battle_leaderboard(session: aiohttp.ClientSession, battle_id: int, page_size: int = TOP_N) -> Optional[List[dict]]:
    params = {"battle_id": str(battle_id), "page": "1", "page_size": str(page_size)}
    try:
        async with session.get(BATTLE_LB_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("leaderboard", [])
    except Exception:
        return None

def _utcnow():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _from_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _read_manager_state() -> dict:
    if MANAGER_STATE_PATH.exists():
        try:
            st = json.loads(MANAGER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            st = {}
    else:
        st = {}

    # defaults
    st.setdefault("enabled", False)
    st.setdefault("next_run_at", None)
    st.setdefault("created_battles", [])
    st.setdefault("songs_per_run", 1)
    st.setdefault("last_run_at", None)
    st.setdefault("last_error", None)
    st.setdefault("week", 0)
    st.setdefault("vote_enabled", True)          # turn off to disable daily vote flow
    st.setdefault("vote_next_start_at", None)    # ISO time for next noon start (UTC)
    st.setdefault("daily_vote", None)            # active vote payload or None
    st.setdefault("daily_core_pool", [])  # rotates through INSTR_SET_WEEK (Mon–Fri)
    st.setdefault("daily_pro_pool", [])   # rotates through PRO_ONE_PER_WEEK (Sat–Sun)
    st.setdefault(SUBS_KEY, {})
    st.setdefault(TOPS_KEY, {})
    st.setdefault(OVERTAKES_KEY, {})
    st.setdefault("last_week_increment_at", None)
    st.setdefault(LINKS_KEY, {})  # NEW: { "<discord_user_id>": [display_name, ...] }
    st.setdefault(CHAMPION_GRANTS_KEY, [])
    if not isinstance(st[CHAMPION_GRANTS_KEY], list):
        st[CHAMPION_GRANTS_KEY] = []

    # --- Normalize LIVE overtake map (more permissive: trims strings) ---
    live = {}
    for k, v in (st.get(OVERTAKES_KEY) or {}).items():
        try:
            live[str(k)] = int(str(v).strip())
        except Exception:
            continue
    st[OVERTAKES_KEY] = live

    # --- Merge from active created_battles; prefer the larger (manual edits respected) ---
    for rec in st.get("created_battles", []):
        if rec.get("winner_announced"):
            continue  # finished battles are snapshots only
        bid = rec.get("battle_id")
        if isinstance(bid, int):
            bkey = str(bid)
            try:
                ov_rec = int(str(rec.get("overtakes", 0)).strip())
            except Exception:
                ov_rec = 0
            ov_live = int(live.get(bkey, 0))
            if ov_rec > ov_live:
                live[bkey] = ov_rec

    return st

def _strip_platform_tags(name: str) -> str:
    s = (name or "").strip()
    # remove any number of trailing bracketed tags: "jnack [RPCS3] (modded)" -> "jnack"
    while True:
        new = re.sub(r"\s*[\[(][^\[\]()]{1,32}[\])]\s*$", "", s)
        if new == s:
            break
        s = new
    return s.strip()

def _norm_link_name(name: str) -> str:
    # case-insensitive, accent-insensitive, collapse spaces
    s = unicodedata.normalize("NFKD", _strip_platform_tags(name)).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _user_links(state: dict, user_id: int) -> list[str]:
    book = state.setdefault(LINKS_KEY, {})
    return list(book.get(str(user_id), []))

def _save_user_links(state: dict, user_id: int, names: list[str]) -> None:
    book = state.setdefault(LINKS_KEY, {})
    # keep unique by normalized name, preserve first display form order
    seen = set()
    out = []
    for n in names:
        key = _norm_link_name(n)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(_strip_platform_tags(n))
    book[str(user_id)] = out

def _link_owner_user_id(state: dict, norm_key: str) -> Optional[int]:
    book = state.get(LINKS_KEY, {}) or {}
    for uid_str, names in book.items():
        for n in (names or []):
            if _norm_link_name(n) == norm_key:
                try:
                    return int(uid_str)
                except Exception:
                    return None
    return None

def _winner_payload(row: Optional[dict]) -> Optional[dict]:
    if not row:
        return None
    score = row.get("score")
    return {
        "name": str(row.get("name", "Unknown")),
        "score": int(score) if isinstance(score, (int, float)) else None,
        "recorded_at": _to_iso(_utcnow()),
    }

def _write_manager_state(state: dict) -> None:
    tmp = MANAGER_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(MANAGER_STATE_PATH)

class WeeklyBattleManager:
    def __init__(self, bot: "LBClient"):
        self.bot = bot
        self.state = _read_manager_state()
        try:
            cur = int(self.state.get("songs_per_run") or 0)
        except Exception:
            cur = 0
        if cur < 1:
            self.state["songs_per_run"] = 1
            _write_manager_state(self.state)
        self.lock = asyncio.Lock()
        self.task = asyncio.create_task(self._loop())
        self.results_task = asyncio.create_task(self._watch_expirations())
        self.role_task = asyncio.create_task(self._watch_role_grants())

    async def _get_guild(self) -> Optional[discord.Guild]:
        gid = getattr(self.bot, "dev_guild_id", None)
        if not gid:
            return None
        g = self.bot.get_guild(gid)
        if g is None:
            try:
                g = await self.bot.fetch_guild(gid)
            except Exception:
                return None
        return g

    async def _grant_champion_role(self, discord_user_id: int, battle_id: int, leader_name: str) -> bool:
        g = await self._get_guild()
        if not g:
            return False
        role = g.get_role(CHAMPION_ROLE_ID)
        # Idempotency: if an active grant already exists for this user/battle, don't re-add
        async with self.lock:
            for g in self.state.setdefault(CHAMPION_GRANTS_KEY, []):
                if int(g.get("user_id", 0)) == int(discord_user_id) and int(g.get("battle_id", 0)) == int(battle_id) and g.get("active"):
                    return True
        if not role:
            return False
        try:
            member = await g.fetch_member(discord_user_id)
        except Exception:
            return False
        try:
            await member.add_roles(role, reason=f"Score Snipe champion for battle {battle_id} ({leader_name})")
        except Exception:
            return False
        rec = {
            "user_id": int(discord_user_id),
            "role_id": int(CHAMPION_ROLE_ID),
            "battle_id": int(battle_id),
            "granted_at": _to_iso(_utcnow()),
            "expires_at": _to_iso(_utcnow() + timedelta(days=7)),
            "removed_at": None,
            "active": True,
        }
        async with self.lock:
            grants = self.state.setdefault(CHAMPION_GRANTS_KEY, [])
            grants.append(rec)
            _write_manager_state(self.state)
        return True

    async def _remove_champion_role_now(self, discord_user_id: int) -> bool:
        g = await self._get_guild()
        if not g:
            return False
        role = g.get_role(CHAMPION_ROLE_ID)
        if not role:
            return False
        try:
            member = await g.fetch_member(discord_user_id)
        except Exception:
            return False
        try:
            await member.remove_roles(role, reason="Score Snipe champion role expired")
        except Exception:
            return False
        return True

    async def _watch_role_grants(self):
        await self.bot.wait_until_ready()
        while True:
            try:
                now = _utcnow()
                to_update: list[dict] = []
                # snapshot first (avoid holding the lock during network calls)
                async with self.lock:
                    grants = list(self.state.get(CHAMPION_GRANTS_KEY, []))
                for rec in grants:
                    if not rec.get("active"):
                        continue
                    exp = _from_iso(rec.get("expires_at"))
                    if exp and exp <= now:
                        to_update.append(rec)
                # process expirations
                for rec in to_update:
                    uid = int(rec.get("user_id"))
                    ok = await self._remove_champion_role_now(uid)
                    async with self.lock:
                        # mark inactive whether or not removal succeeded (it will retry next boot if still on member)
                        rec["active"] = False
                        rec["removed_at"] = _to_iso(_utcnow())
                        # write back the mutated object (same identity)
                        _write_manager_state(self.state)
            except Exception as e:
                async with self.lock:
                    self.state["last_error"] = f"champ role watcher: {e!r}"
                    _write_manager_state(self.state)
            await asyncio.sleep(60)

    def _pending_unannounced(self) -> list[dict]:
        now = _utcnow()
        recs = []
        for r in (self.state.get("created_battles") or []):
            if r.get("winner_announced"):
                continue
            dt = _from_iso(r.get("expires_at")) if isinstance(r.get("expires_at"), str) else None
            if dt:
                recs.append((dt, r))
        # sort by expiry time
        recs.sort(key=lambda t: t[0])
        return [r for _, r in recs]

    def _ensure_pool(self, key: str, all_values: list[str]) -> list[str]:
        """Return the remaining pool for `key`, refilling with a new random order if empty."""
        pool = list(self.state.get(key) or [])
        if not pool:
            pool = random.sample(all_values, k=len(all_values))
            self.state[key] = pool
            _write_manager_state(self.state)
        return pool

    def _pop_from_pool(self, key: str, all_values: list[str]) -> str:
        pool = self._ensure_pool(key, all_values)
        pick = pool.pop(0)              # pop the next instrument (randomized order)
        self.state[key] = pool
        _write_manager_state(self.state)
        return pick

    def _instrument_for_today(self, now: Optional[datetime] = None) -> str:
        """Mon–Fri: cycle core instruments; Sat–Sun: cycle Pro instruments."""
        now = now or _utcnow()
        dow = now.weekday()  # Mon=0 ... Sun=6
        if dow in (2, 6):    # Wed/Sun → Pro pool (no repeats until exhausted)
            return self._pop_from_pool("daily_pro_pool", PRO_ONE_PER_WEEK)
        else:                # Mon–Fri, not Wed → core pool (no repeats until exhausted)
            return self._pop_from_pool("daily_core_pool", INSTR_SET_WEEK)

    def _choose_daily_battle(self) -> Optional[tuple[Song, str]]:
        """Pick today's instrument (from the proper pool), then a song for it."""
        instr = self._instrument_for_today()
        pair = self._pick_varied_by_genre([instr])
        return pair[0] if pair else None

    async def _watch_expirations(self):
        await self.bot.wait_until_ready()
        while True:
            try:
                pending = self._pending_unannounced()
                if not pending:
                    await asyncio.sleep(60)
                    continue

                now = _utcnow()

                # Find the next expiry we haven't announced
                next_dt = None
                for rec in pending:
                    dt = _from_iso(rec.get("expires_at"))
                    if dt and (next_dt is None or dt < next_dt):
                        next_dt = dt

                if not next_dt:
                    await asyncio.sleep(60)
                    continue

                # sleep until just after the next expiry
                delay = max(1, int((next_dt - now).total_seconds()) + RESULTS_GRACE_SECONDS)
                # cap the sleep so we re-check occasionally
                await asyncio.sleep(min(delay, 3600))

                # Batch: pick all that are now expired or expiring very soon (within window)
                cutoff = _utcnow() + timedelta(seconds=WINNERS_BATCH_WINDOW_SECONDS)
                batch: list[dict] = []
                for rec in self._pending_unannounced():
                    dt = _from_iso(rec.get("expires_at"))
                    if dt and dt <= cutoff and dt <= _utcnow():  # already expired (grace elapsed)
                        batch.append(rec)

                if batch:
                    await self._announce_winners(batch)

            except Exception as e:
                async with self.lock:
                    self.state["last_error"] = f"watcher error: {e!r}"
                    _write_manager_state(self.state)
                await asyncio.sleep(10)

    async def _announce_winners(self, finished_recs: list[dict]):
        if not finished_recs:
            return

        chan_id = getattr(self.bot, "announce_channel_id", None)
        role_id = getattr(self.bot, "ping_role_id", None)
        if not chan_id:
            async with self.lock:
                self.state["last_error"] = "announce_channel_id not configured (results)"
                _write_manager_state(self.state)
            return

        channel = self.bot.get_channel(chan_id) or await self.bot.fetch_channel(chan_id)

        # --- Single-winner fancy embed ---
        if len(finished_recs) == 1:
            rec = finished_recs[0]
            bid = int(rec["battle_id"])

            # Normalize instrument
            instr_key = rec.get("instrument")
            if isinstance(instr_key, int):
                instr_key = INSTR_BY_ROLE_ID.get(instr_key)
            elif isinstance(instr_key, str) and instr_key.isdigit():
                instr_key = INSTR_BY_ROLE_ID.get(int(instr_key))
            is_band = (instr_key == "band")
            instr_name = INSTR_DISPLAY_NAMES.get(instr_key or "", (instr_key or "Instrument").capitalize())
            emoji = INSTR_EMOJI.get(instr_key or "", "🎵")

            # fetch more rows for band
            pg_n = BAND_TOP_N if is_band else 5
            rows = await fetch_battle_leaderboard(self.bot.http_session, bid, page_size=pg_n) or []
            top = rows[0] if rows else None

            # Determine champion name(s)
            champ_names: list[str] = []
            if rows:
                top_score = rows[0].get("score")
                champ_names: list[str] = []
                if rows:
                    top_score = rows[0].get("score")
                    # collect all tied-at-top names, strip platform tags, keep order & de-dupe after stripping
                    seen = set()
                    for r in rows:
                        if r.get("score") == top_score:
                            dn = _strip_platform_tags(str(r.get("name", "Unknown")))
                            if dn and dn not in seen:
                                seen.add(dn)
                                champ_names.append(dn)

            # --- Header line ---
            if not champ_names:
                champ_header = "### 🏆 No entries" if is_band else "# 🏆 No entries"
            elif is_band:
                # band: never truncate; smaller heading
                champ_header = "### 🏆 " + ", ".join(champ_names)
            elif len(champ_names) == 1:
                champ_header = f"# 🏆 {champ_names[0]}"
            else:
                more = "…" if len(champ_names) > 3 else ""
                champ_header = "# 🏆 " + ", ".join(champ_names) + " — Co-champions!"

            # Table (band gets 15)
            table = format_battle_rows(rows or [], max_n=pg_n)

            # Song details
            song = self._song_from_id(int(rec["song_id"]))
            song_title = song.name if song else f"Song ID {rec.get('song_id')}"
            artist = (song.artist if song else None) or "—"
            album_line = (song.album_name or "—") if song else "—"
            if song and song.year:
                album_line = f"{album_line} ({song.year})"
            rank_val = _rank(song, instr_key) if (song and instr_key) else None
            diff_txt = difficulty_label(_th_key(instr_key or ""), rank_val) if (instr_key) else "—"

            # Overtakes
            over_map: Dict[str, int] = self.state.get(OVERTAKES_KEY, {}) or {}
            bkey = str(bid)
            changes = int(over_map.get(bkey, 0) or 0)

            # Table with runner-ups (top 5)
            table = format_battle_rows(rows or [], max_n=pg_n)

            # Time tags
            def _to_unix(v) -> Optional[int]:
                if isinstance(v, (int, float)):
                    return int(v)
                if isinstance(v, str):
                    s = v.strip()
                    if s.isdigit():
                        return int(s)
                    dt = _from_iso(s)
                    return int(dt.timestamp()) if dt else None
                return None

            starts_ts = _to_unix(rec.get("created_at") or rec.get("starts_at"))
            ends_ts   = _to_unix(rec.get("expires_at"))

            # Description
            parts = [
                champ_header,
                f"{emoji} **{instr_name}** — *{song_title}* by _{artist}_",
            ]
            parts.append(f"**Overtakes this week:** {changes}")
            parts.append("")  # spacer
            parts.append(table)

            embed = discord.Embed(
                title="Score Snipe — Champions!",
                description="\n".join(parts),
                color=0xFACC15,
            )

            # Thumbnail + info fields (like /battles)
            if song:
                thumb = await fetch_song_art_url(self.bot.http_session, song.slug)
                embed.set_thumbnail(url=thumb)
                embed.add_field(name="Artist", value=artist, inline=True)
                embed.add_field(name="Album", value=album_line, inline=True)
                embed.add_field(name="\u200B", value="\u200B", inline=True)
                embed.add_field(name="Difficulty", value=diff_txt, inline=True)
                embed.add_field(name="Instrument", value=instr_name, inline=True)
                embed.add_field(name="\u200B", value="\u200B", inline=True)
                details = f"{pretty_genre(song.genre)} • {pretty_source(song.source)} • ID `{song.song_id}`"
                embed.add_field(name="Details", value=details, inline=False)
            else:
                embed.add_field(name="Details", value=f"ID `{rec.get('song_id')}`", inline=False)

            embed.add_field(name="Started", value=(f"<t:{starts_ts}:R>" if starts_ts else "—"), inline=True)
            embed.add_field(name="Ended",   value=(f"<t:{ends_ts}:R>"   if ends_ts   else "—"), inline=True)

            # Send
            try:
                await channel.send(
                    content=(f"<@&{role_id}>" if role_id else None),
                    embeds=[embed],
                    allowed_mentions=discord.AllowedMentions(roles=True, users=False, everyone=False),
                )
            except Exception as e:
                async with self.lock:
                    self.state["last_error"] = f"results announce failed: {e!r}"
                    _write_manager_state(self.state)
                return

            # Persist winner + freeze/clear overtake count
            async with self.lock:
                by_id = {r["battle_id"]: r for r in self.state.get("created_battles", [])}
                if rec["battle_id"] in by_id:
                    by_id[rec["battle_id"]]["winner"] = _winner_payload(top)
                    by_id[rec["battle_id"]]["winner_announced"] = True
                    by_id[rec["battle_id"]]["announced_at"] = _to_iso(_utcnow())
                    by_id[rec["battle_id"]]["overtakes"] = int(over_map.get(bkey, 0) or 0)
                over_map.pop(bkey, None)
                self.state[OVERTAKES_KEY] = over_map
                _write_manager_state(self.state)

            # Champion role grant(s) if linked
            if champ_names:
                names_to_award = champ_names
                for nm in names_to_award:
                    uid = _link_owner_user_id(self.state, _norm_link_name(nm))
                    if uid:
                        try:
                            await self._grant_champion_role(uid, bid, nm)
                        except Exception:
                            pass

            return  # done

        # --- Fallback: your existing multi-winner flow (unchanged, condensed) ---
        winners: list[tuple[dict, Optional[dict]]] = []
        band_groups: dict[int, dict] = {}
        for rec in finished_recs:
            bid = int(rec["battle_id"])
            instr = rec.get("instrument")
            if isinstance(instr, int):
                instr_key = INSTR_BY_ROLE_ID.get(instr)
            elif isinstance(instr, str):
                instr_key = instr
            else:
                instr_key = None
            if instr_key == "band":
                rows = await fetch_battle_leaderboard(self.bot.http_session, bid, page_size=TOP_N) or []
                if rows:
                    top_score = rows[0].get("score")
                    grp = [r for r in rows if r.get("score") == top_score]
                    # clean display names for header/summary
                    names = [_strip_platform_tags(str(r.get("name","Unknown"))) for r in grp]
                    try:
                        sc_str = f"{int(top_score):,}"
                    except Exception:
                        sc_str = "?"
                    band_groups[bid] = {"score": int(top_score) if isinstance(top_score,(int,float)) else None,
                                        "names": names, "_score_str": sc_str}
                    top = dict(rows[0]); top["_score_str"] = sc_str; top["_name"] = names[0] if names else "Unknown"
                    winners.append((rec, top))
                else:
                    winners.append((rec, None))
            else:
                top = await fetch_battle_top_winner(self.bot.http_session, bid)
                winners.append((rec, top))

        weeks = {rec.get("week") for rec, _ in winners if rec.get("week")}
        title = f"Score Snipe Week {list(weeks)[0]} — Winners Circle" if len(weeks) == 1 else "Score Snipe — Winners Circle"
        SPACER = "\u200B"
        num_champs = sum(1 for _, w in winners if w)
        over_map: Dict[str, int] = self.state.get(OVERTAKES_KEY, {}) or {}
        lines: list[str] = [("No champions crowned" if num_champs == 0 else f"{num_champs} champion{'s' if num_champs != 1 else ''} crowned") + " Congratulations! Use `/battles` to view full tables.", SPACER]

        for rec, w in winners:
            song = self._song_from_id(int(rec["song_id"]))
            instr_key = rec.get("instrument")
            if isinstance(instr_key, int):
                instr_key = INSTR_BY_ROLE_ID.get(instr_key)
            instr_name = INSTR_DISPLAY_NAMES.get(instr_key, (instr_key or "").capitalize())
            emoji = INSTR_EMOJI.get(instr_key, "🎵")
            song_part = song.name if song else f"ID {rec.get('song_id')}"
            artist_part = f" by _{song.artist}_" if song and song.artist else ""
            bkey = str(rec["battle_id"])
            changes = int(over_map.get(bkey, 0) or 0)
            if instr_key == "band":
                info = band_groups.get(int(rec["battle_id"])) if w else None
                if info and info.get("score") is not None:
                    names = info["names"]; sc_str = info["_score_str"]
                    if len(names) == 1:
                        winner_line = f"🏆 **{names[0]}** — **{sc_str}**"
                    elif len(names) <= 4:
                        names_str = "; ".join(f"**{n}**" for n in names)
                        winner_line = f"🏆 **{sc_str}** — {names_str}"
                    else:
                        winner_line = f"🏆 **{sc_str}** — ×{len(names)}"
                else:
                    winner_line = "🚫 *No entries*"
            else:
                winner_line = f"🏆 **{w['_name']}** — **{w['_score_str']}**" if w else "🚫 *No entries*"
            if w and changes:
                winner_line += f" • **{changes}** overtake{'s' if changes != 1 else ''}"
            lines.append(f"{emoji} **{instr_name}** — *{song_part}*{artist_part}")
            lines.append(" " + winner_line)
            lines.append(SPACER)

        nxt = _from_iso(self.state.get("next_run_at"))
        if nxt and nxt > _utcnow():
            ts = int(nxt.timestamp())
            lines.append(f"*Next Week's Score Snipe begins <t:{ts}:R> — <t:{ts}:t>*")

        embed = discord.Embed(title=title, description="\n".join(lines), color=0xFACC15)
        try:
            await channel.send(
                content=(f"<@&{role_id}>" if role_id else None),
                embeds=[embed],
                allowed_mentions=discord.AllowedMentions(roles=True, users=False, everyone=False),
            )
        except Exception as e:
            async with self.lock:
                self.state["last_error"] = f"results announce failed: {e!r}"
                _write_manager_state(self.state)
            return

        # Persist + clear overtake counts for multi-winner fallback
        async with self.lock:
            by_id = {r["battle_id"]: r for r in self.state.get("created_battles", [])}
            for rec, w in winners:
                bkey = str(rec["battle_id"])
                rid = rec["battle_id"]
                if rid in by_id:
                    by_id[rid]["winner"] = _winner_payload(w)
                    by_id[rid]["winner_announced"] = True
                    by_id[rid]["announced_at"] = _to_iso(_utcnow())
                    by_id[rid]["overtakes"] = int(over_map.get(bkey, 0) or 0)
                over_map.pop(bkey, None)
            self.state[OVERTAKES_KEY] = over_map
            _write_manager_state(self.state)

    def _eligible_songs(self) -> list[Song]:
        out: list[Song] = []
        for s in self.bot.song_index.songs:
            if s.song_id is None:
                continue
            if s.song_id > 5_110_000:
                continue
            if "(2x bass pedal)" in s.name.lower():
                continue
            src = (s.source or "").lower()
            if src in EXCLUDED_SOURCES:
                continue
            out.append(s)
        return out

    def _pick_varied_by_genre(self, instrs: list[str]) -> list[tuple[Song, str]]:
        pool = self._eligible_songs()
        if not pool:
            return []

        chosen: list[tuple[Song, str]] = []
        used_ids: set[int] = set()
        used_genres: set[str] = set()
        used_buckets: Dict[int, int] = {}  # bucket index -> count so far (soft variety)

        for instr in instrs:
            # Only candidates with a real part for this instrument
            viable = [s for s in pool if s.song_id not in used_ids and has_part(s, instr)]
            if not viable:
                continue

            # Build weights
            items, weights = [], []
            for s in viable:
                rank_val = _rank(s, instr)
                b = difficulty_bucket_index(_th_key(instr), rank_val)
                if b is None:
                    continue

                w = DIFF_BUCKET_WEIGHT[b]

                # Small boost if this genre hasn't been used yet
                g = (s.genre or "").lower()
                if g and g not in used_genres:
                    w *= GENRE_BOOST

                # Soft penalty if we've already used this bucket 1+ times
                rep = used_buckets.get(b, 0)
                if rep > 0:
                    w *= (REPEAT_BUCKET_PENALTY ** rep)

                # NEW: down-weight ugc sources
                src = (s.source or "").lower()
                if src in UGC_DOWNWEIGHT_SOURCES:
                    w *= UGC_WEIGHT_MULTIPLIER

                items.append(s)
                weights.append(max(w, 0.01))


            # Fallback if weighting list ended up empty for some reason
            pick = random.choices(items, weights=weights, k=1)[0] if items else random.choice(viable)

            chosen.append((pick, instr))
            used_ids.add(pick.song_id)
            if pick.genre:
                used_genres.add(pick.genre.lower())

            # Track used difficulty bucket
            b_used = difficulty_bucket_index(_th_key(instr), _rank(pick, instr))
            if b_used is not None:
                used_buckets[b_used] = used_buckets.get(b_used, 0) + 1

        return chosen

    def _choose_week_set(self, count: int) -> list[tuple[Song, str]]:
        # Always prioritize the core instruments first
        main_n = min(count, len(INSTR_SET_WEEK))
        instrs = random.sample(INSTR_SET_WEEK, main_n)

        # If caller asked for more than the core set, add exactly one pro slot
        extra_slots = max(0, count - main_n)
        if extra_slots >= 1:
            instrs.append(random.choice(PRO_ONE_PER_WEEK))

        # (If someone sets count > 6, we still cap the "pro" to exactly one)
        return self._pick_varied_by_genre(instrs)


    def _song_from_id(self, sid: int) -> Optional[Song]:
        return self.bot.song_index.by_id.get(sid) if self.bot.song_index else None


    def is_enabled(self) -> bool:
        return bool(self.state.get("enabled"))

    async def enable(self):
        async with self.lock:
            self.state["enabled"] = True
            if not self.state.get("next_run_at"):
                self.state["next_run_at"] = _to_iso(_utcnow() + timedelta(days=AUTO_BATTLE_PERIOD_DAYS))
            _write_manager_state(self.state)

    async def disable(self):
        async with self.lock:
            self.state["enabled"] = False
            # clear any pending schedule so status shows "—"
            self.state["next_run_at"] = None
            _write_manager_state(self.state)

    def _compute_next(self, now: datetime) -> datetime:
        nxt = _from_iso(self.state.get("next_run_at"))
        if not nxt or nxt <= now:
            nxt = now + timedelta(days=AUTO_BATTLE_PERIOD_DAYS)
        return nxt

    async def post_announcement(self, recs: list[dict], increment_week: bool):
        # Decide week number; persist if changed
        async with self.lock:
            week = int(self.state.get("week") or 0)
            if week == 0:
                week = 1
            elif increment_week:
                week += 1
            self.state["week"] = week
            _write_manager_state(self.state)

        await self._post_to_channel(week, recs)

    async def _post_to_channel(self, week: int, recs: list[dict]):
        chan_id = getattr(self.bot, "announce_channel_id", None)
        role_id = getattr(self.bot, "ping_role_id", None)
        if not chan_id:
            async with self.lock:
                self.state["last_error"] = "announce_channel_id not configured"
                _write_manager_state(self.state)
            return

        # Resolve channel
        channel = self.bot.get_channel(chan_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(chan_id)
            except Exception as e:
                async with self.lock:
                    self.state["last_error"] = f"fetch_channel failed: {e!r}"
                    _write_manager_state(self.state)
                return

        # Compute a common "ends in" based on earliest expiry (they should all be the same)
        exp_dts = []
        for r in recs:
            dt = _from_iso(r.get("expires_at")) if isinstance(r.get("expires_at"), str) else None
            if dt:
                exp_dts.append(dt)
        ends_txt = ""
        if exp_dts:
            ends_ts = _to_unix_ts(min(exp_dts))
            ends_txt = f"\n**Ends:** <t:{ends_ts}:R>"

        # Build a compact summary list for the header embed
        summary_lines = []
        for r in recs:
            song = self._song_from_id(int(r["song_id"]))
            instr_key = r.get("instrument")
            emoji = INSTR_EMOJI.get(instr_key, "🎵")
            if song:
                thumb = await fetch_song_art_url(self.bot.http_session, song.slug)
                rank_val = _rank(song, instr_key)
                diff_txt = difficulty_label(_th_key(instr_key), rank_val)
                summary_lines.append(
                    f"{emoji} **{INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())}** — "
                    f"**{song.name}** *by {song.artist}* • *{diff_txt}*"
                )
            else:
                summary_lines.append(f"{emoji} **{INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())}** — *Unknown song* (ID {r.get('song_id')})")

        reminder = (
            "Reminder: Visit `Quickplay>Setlists / Battles` to find the battles. "
            "Playing the song by itself in normal quickplay will not count."
        )

        header = discord.Embed(
            title=f"Score Snipe — Week {week}",
            description="New battles are live! Good luck. 🔥"
                        + ends_txt
                        + "\n\n"
                        + reminder
                        + "\n\n"
                        + "\n".join(summary_lines),
            color=0x22C55E,  # green accent for the pack header
        )

        # One embed per song, color-coded by instrument
        song_embeds: list[discord.Embed] = []
        # One embed per song, color-coded by instrument
        for r in recs[:10]:
            song = self._song_from_id(int(r["song_id"]))
            instr_key = r.get("instrument")
            instr_name = INSTR_DISPLAY_NAMES.get(instr_key, (instr_key or "").capitalize())
            emoji = INSTR_EMOJI.get(instr_key, "🎵")
            color = INSTR_COLOR.get(instr_key, 0x5865F2)

            title_txt = f"{emoji} {instr_name} — Score Snipe — Week {week}"
            e = discord.Embed(title=title_txt, color=color)
            e.description = f"# {song.name}" if song else f"**Unknown Song** (ID {r.get('song_id')})"


            if song:
                # pieces
                thumb = await fetch_song_art_url(self.bot.http_session, song.slug)
                e.set_thumbnail(url=thumb)
                artist = song.artist or "—"
                album_line = song.album_name or "—"
                if song.year:
                    album_line = f"{album_line} ({song.year})"
                author = getattr(song, "author", None) or "—"
                rank_val = _rank(song, instr_key)
                diff_txt = difficulty_label(_th_key(instr_key), rank_val)

                # Row 1: Artist | Album  (+ pad)
                e.add_field(name="Artist", value=artist, inline=True)
                e.add_field(name="Album", value=album_line, inline=True)
                e.add_field(name="\u200B", value="\u200B", inline=True)  # pad third column

                # Row 2: Difficulty | Author  (+ pad)
                e.add_field(name="Difficulty", value=diff_txt, inline=True)
                e.add_field(name="Author", value=author, inline=True)
                e.add_field(name="\u200B", value="\u200B", inline=True)  # pad third column

                # Row 3: Details (full width)
                details = f"{pretty_genre(song.genre)} • {pretty_source(song.source)} • ID `{song.song_id}`"
                e.add_field(name="Details", value=details, inline=False)
            else:
                e.add_field(name="Details", value=f"ID `{r.get('song_id')}`", inline=False)

            song_embeds.append(e)

        try:
            await channel.send(
                content=(f"<@&{role_id}>" if role_id else None),  # real ping happens in content
                embeds=[header] + song_embeds,
                allowed_mentions=discord.AllowedMentions(roles=True, users=False, everyone=False),
            )
        except Exception as e:
            async with self.lock:
                self.state["last_error"] = f"announce send failed: {e!r}"
                _write_manager_state(self.state)



    async def _loop(self):
        # wait for bot to be fully ready
        await self.bot.wait_until_ready()
        while True:
            try:
                if not self.is_enabled():
                    await asyncio.sleep(30)
                    continue

                now = _utcnow()
                nxt = self._compute_next(now)
                sleep_s = max(1, int((nxt - now).total_seconds()))
                # nap until next scheduled time (cap to avoid huge sleeps)
                await asyncio.sleep(min(sleep_s, 3600))
                # re-check if we crossed the boundary
                if _utcnow() >= _from_iso(self.state.get("next_run_at")):
                    await self.run_once()  # create new weekly battle(s)
            except Exception as e:
                async with self.lock:
                    self.state["last_error"] = repr(e)
                    _write_manager_state(self.state)
                # brief backoff
                await asyncio.sleep(10)

    async def run_once(self, count: Optional[int] = None) -> list[dict]:
        now = _utcnow()
        if now.weekday() == 4:  # Friday
            last_inc = _from_iso(self.state.get("last_week_increment_at"))
            if not last_inc or last_inc.date() != now.date():
                async with self.lock:
                    cur = int(self.state.get("week") or 0)
                    self.state["week"] = cur + 1 if cur >= 1 else 1  # start at 1 if unset
                    self.state["last_week_increment_at"] = _to_iso(now)
                    _write_manager_state(self.state)

        n = max(1, int(count or 1))
        created: list[dict] = []
        for _ in range(n):
            pick = self._choose_daily_battle()
            if not pick:
                break
            song, instr = pick
            rec = await self._create_one_battle(song=song, instr_key=instr, week_num=int(self.state.get("week") or 1))
            if rec:
                created.append(rec)

        async with self.lock:
            self.state["last_run_at"] = _to_iso(_utcnow())
            if self.state.get("enabled"):
                self.state["next_run_at"] = _to_iso(_utcnow() + timedelta(days=AUTO_BATTLE_PERIOD_DAYS))
            self.state["created_battles"].extend(created)
            _write_manager_state(self.state)

        if created:
            await self.post_announcement(created, increment_week=False)
        return created

    def _pick_random_song(self) -> Song:
        return random.choice(self.bot.song_index.songs)

    def _pick_random_instr(self) -> str:
        return random.choice(ALLOWED_RANDOM_INSTRUMENTS)

    async def _create_one_battle(self, *, song: Song, instr_key: str, week_num: int) -> Optional[dict]:
        if not has_part(song, instr_key):
            return None
        role_id = INSTR_ROLE_IDS[instr_key]
        title = f"Score Snipe, Week {week_num} {INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())}"
        description = f"{song.name} — {song.artist} (ID {song.song_id})"

        api_key = get_api_key()
        if not api_key:
            async with self.lock:
                self.state["last_error"] = "Create failed: missing API key"
                _write_manager_state(self.state)
            return None

        starts_at  = _to_iso(_utcnow())
        expires_at = _to_iso(_utcnow() + timedelta(days=BATTLE_DURATION_DAYS))

        payload = {
            "title": title,
            "description": description,
            "song_ids": [song.song_id],
            "starts_at": starts_at,
            "expires_at": expires_at,
            "instrument": role_id,
            "flags": 0
        }
        headers = {
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        }

        async with self.bot.http_session.post(CREATE_BATTLE_URL, json=payload, headers=headers, timeout=10) as resp:
            raw = await resp.text()
            ok = 200 <= resp.status < 300
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {}
            if not ok or not data.get("success", True):
                # stash last_error but don't crash the loop
                async with self.lock:
                    self.state["last_error"] = f"Create failed: {resp.status} - {raw}"
                    _write_manager_state(self.state)
                return None

            battle_id = data.get("battle_id")
            return {
                "battle_id": battle_id,
                "song_id": song.song_id,
                "instrument": instr_key,
                "created_at": starts_at,
                "expires_at": expires_at,
                "title": title,
                "week": week_num,
                "winner_announced": False,
                "winner": None,
                "overtakes": 0,
            }

NUM_EMOJIS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
VOTE_CATEGORIES = ("year", "source", "song", "album", "artist", "diff", "genre")

def _supports_native_polls() -> bool:
    # feature-detect: new discord.py exposes these classes
    return all(hasattr(discord, attr) for attr in ("Poll", "PollAnswer", "PollMedia"))

async def _can_send_polls(ch: discord.abc.Messageable) -> bool:
    try:
        # Polls must be sent in guild text channels or threads
        if isinstance(ch, (discord.TextChannel, discord.Thread)):
            me = ch.guild.me  # type: ignore
            perms = ch.permissions_for(me)
            # Newer discord.py exposes perms.create_polls; if absent, assume allowed
            can_create = getattr(perms, "create_polls", True)
            return perms.send_messages and can_create
        return False
    except Exception:
        return False


class DailyVoteManager:
    """Posts a daily 1-hour vote with hidden hints, then creates a battle from the winner."""
    VOTE_KEY = "daily_vote"

    def __init__(self, bot: "LBClient", weekly: WeeklyBattleManager):
        self.bot = bot
        self.weekly = weekly
        self.lock = asyncio.Lock()
        self._finalize_task: Optional[asyncio.Task] = None
        # Ensure state slot exists
        st = self.weekly.state
        if self.VOTE_KEY not in st:
            st[self.VOTE_KEY] = None
            _write_manager_state(st)

    # ---- helpers ----
    def _vote(self) -> Optional[dict]:
        return self.weekly.state.get(self.VOTE_KEY) or None

    def _supports_polls(self) -> bool:
        # Conservative feature detection across discord.py minor versions
        return hasattr(discord, "Poll") and hasattr(discord, "PollAnswer")

    def _answer_count(self, ans) -> int:
        # Different minors exposed slightly different names; be robust.
        for attr in ("vote_count", "count", "votes"):
            if hasattr(ans, attr) and getattr(ans, attr) is not None:
                try:
                    return int(getattr(ans, attr))
                except Exception:
                    pass
        return 0

    async def _channel(self) -> Optional[discord.abc.Messageable]:
        chan_id = getattr(self.bot, "announce_channel_id", None)
        if not chan_id:
            return None
        return self.bot.get_channel(chan_id) or await self.bot.fetch_channel(chan_id)

    def active_vote_ends_at(self) -> Optional[datetime]:
        v = self._vote()
        return _from_iso(v.get("ends_at")) if v else None

    def _hint_for_song(self, song: Song, instr_key: str) -> str:
        """
        Build a localized hint line using your Play a Show strings.
        Tries several categories and picks one at random.
        """
        ideas: list[str] = []

        # Genre
        if getattr(song, "genre", None):
            # DTA uses either genre or subgenre — we'll just use genre here.
            ideas.append(loc("play_a_show_genre", pretty_genre(song.genre)))

        # Year (sometimes decade)
        year = None
        try:
            if getattr(song, "year", None):
                year = int(song.year)
        except Exception:
            year = None
        if year:
            if random.choice((True, False)):
                ideas.append(loc("play_a_show_year", year))
            else:
                decade = (year // 10) * 10
                ideas.append(loc("play_a_show_year_decade", decade))

        # Album
        if getattr(song, "album_name", None):
            # Original has "%s <alt>%s</alt>" (album, artist). Discord: album only.
            ideas.append(loc("play_a_show_album", song.album_name))

        # Artist
        if getattr(song, "artist", None):
            ideas.append(loc("play_a_show_artist", song.artist))

        # Length (ms) thresholds match the DTA logic: >420s, >240s, >120s
        length_ms = getattr(song, "length_ms", None)
        if isinstance(length_ms, (int, float)) and length_ms > 0:
            if length_ms > 420_000:
                ideas.append(loc("play_a_show_really_long"))
            elif length_ms > 240_000:
                ideas.append(loc("play_a_show_long"))
            elif length_ms > 120_000:
                ideas.append(loc("play_a_show_short"))
            else:
                ideas.append(loc("play_a_show_really_short"))

        # Tempo (if available) — map to slow/med/fast as in DTA
        tempo = getattr(song, "anim_tempo", None)  # if you store it
        if tempo in (16, 32, 64, 112):
            ideas.append({
                16: loc("play_a_show_slow"),
                32: loc("play_a_show_med"),
                64: loc("play_a_show_fast"),
                112: loc("play_a_show_fast"),
            }[tempo])

        # Difficulty (localized, “A/An %s song on %s”)
        # Use your existing helpers to get tier and label.
        try:
            tier = _rank(song, instr_key) or 0  # 0..?
            diff_label = difficulty_label(_th_key(instr_key), tier)  # e.g., "Apprentice"
            instr_name = INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())

            # Match DTA special-casing: tiers 1 and 6 use the "An %s..." key.
            use_an = tier in (1, 6)
            if not use_an:
                # Fallback: article by vowel sound
                use_an = str(diff_label).strip().lower()[:1] in "aeiou"

            if diff_label:
                ideas.append(
                    loc("play_a_show_diff_an" if use_an else "play_a_show_diff", diff_label, instr_name)
                )
        except Exception:
            pass

        # Source / UGC author (if you track them)
        if getattr(song, "author", None):
            ideas.append(loc("play_a_show_author", song.author))
        elif getattr(song, "source", None):
            # e.g., "RBN", "RB4", "Customs", etc.
            ideas.append(loc("play_a_show_source", pretty_source(song.source)))

        # Fallback
        if not ideas:
            ideas.append(loc("play_a_show_med"))  # “A completely random song”

        return random.choice(ideas)

    def _decade_label(self, year: int) -> str:
        """Return 'the 70's' style label from a year."""
        decade = (int(year) // 10) * 10
        # "the 70's" rather than "the 70s"
        short = str(decade)[-2:]
        return f"the {short}'s"

    def _difficulty_hint(self, s, instr_key: str) -> str | None:
        """Localized-ish difficulty line: 'A <Tier> song on <Instrument>' with correct article."""
        try:
            tier = _rank(s, instr_key) or 0
            tier_name = difficulty_label(_th_key(instr_key), tier)  # your existing helper
            instr_name = INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())
            if not tier_name:
                return None
            needs_an = str(tier_name).strip().lower()[:1] in "aeiou"
            return f"{'An' if needs_an else 'A'} {tier_name} song on {instr_name}"
        except Exception:
            return None

    def _hint_for_category(self, s, category: str, instr_key: str) -> str | None:
        """Build the display hint for one category (never reveals the song title)."""
        if category == "genre":
            g = getattr(s, "genre", None)
            if g:
                return f"A {pretty_genre(g)} song"

        if category == "year":
            y = getattr(s, "year", None)
            try:
                y = int(y) if y is not None else None
            except Exception:
                y = None
            if y:
                if random.choice((True, False)):
                    return f"A song from {y}"
                else:
                    return f"A song from {self._decade_label(y)}"

        if category == "length":
            # (Not used; keeping for parity if you add later)
            ms = getattr(s, "length_ms", None)
            if isinstance(ms, int):
                if ms > 420_000: return "A REALLY long song"
                if ms > 240_000: return "A long song"
                if ms > 120_000: return "A short song"
                return "A REALLY short song"

        if category == "song":
            return "A completely random song"

        if category == "tempo":
            tempo = getattr(s, "anim_tempo", None)
            if tempo in (16, 32, 64, 112):
                return {16: "A slow song", 32: "A completely random song", 64: "A fast song", 112: "A fast song"}[tempo]

        if category == "album":
            alb = getattr(s, "album_name", None)
            if alb:
                return f"A song from {alb}"

        if category == "artist":
            art = getattr(s, "artist", None)
            if art:
                return f"A song by {art}"

        if category == "diff":
            return self._difficulty_hint(s, instr_key)

        if category == "source":
            src = getattr(s, "source", None)
            if src:
                return f"A song from {pretty_source(src)}"

        return None

    def _pick_song_for_category(self, pool: list, category: str, instr_key: str, used_ids: set[int]) -> tuple[object, str] | None:
        """
        Pick a song from pool that:
          - has the instrument,
          - hasn't been used,
          - has metadata required by the category,
          - yields a non-empty hint.
        """
        random.shuffle(pool)
        for s in pool:
            sid = getattr(s, "song_id", None)
            if sid in used_ids:
                continue
            if not has_part(s, instr_key):
                continue

            # ensure metadata presence for the specific category
            needed_ok = True
            if category == "year":
                try:
                    y = int(getattr(s, "year", 0))
                    needed_ok = y > 0
                except Exception:
                    needed_ok = False
            elif category == "album":
                needed_ok = bool(getattr(s, "album_name", None))
            elif category == "artist":
                needed_ok = bool(getattr(s, "artist", None))
            elif category == "genre":
                needed_ok = bool(getattr(s, "genre", None))
            elif category == "diff":
                needed_ok = self._difficulty_hint(s, instr_key) is not None
            elif category == "source":
                needed_ok = bool(getattr(s, "source", None))
            elif category == "song":
                needed_ok = True

            if not needed_ok:
                continue

            hint = self._hint_for_category(s, category, instr_key)
            if hint:
                return s, hint

        return None

    def _build_vote_options(self, instr_key: str) -> list[dict]:
        """
        Create five unique option *types* (no duplicates), each bound to a hidden song.
        Respects _eligible_songs() and instrument availability.
        """
        base_pool = self.weekly._eligible_songs()
        used_song_ids: set[int] = set()
        chosen: list[dict] = []

        # Randomize category order, then fill up to 5
        cats = list(VOTE_CATEGORIES)
        random.shuffle(cats)

        def try_fill_from(categories: list[str]):
            for cat in categories:
                if len(chosen) >= 5:
                    break
                pick = self._pick_song_for_category(base_pool, cat, instr_key, used_song_ids)
                if pick:
                    s, hint = pick
                    sid = getattr(s, "song_id", None)
                    used_song_ids.add(sid)
                    chosen.append({
                        "category": cat,
                        "song_id": sid,
                        "hint": hint
                    })

        # Primary pass: randomized categories
        try_fill_from(cats)

        # Fallbacks if we couldn't reach 5: allow "song" category to pad
        if len(chosen) < 5:
            pad_needed = 5 - len(chosen)
            pads = ["song"] * pad_needed
            try_fill_from(pads)

        return chosen[:5]

    def _default_thumb_url(self) -> str:
        # Use the canonical default art used elsewhere in the bot.
        return DEFAULT_SONG_ART

    def _build_announce_embed(self, vote: dict) -> discord.Embed:
        instr_key = vote["instrument"]
        pretty_instr = INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())
        ends = _from_iso(vote.get("ends_at"))
        ts = int(ends.timestamp()) if ends else None

        lines = [
            "🗳️ **Daily Vote — Score Snipe**",
            f"**Instrument:** {pretty_instr}",
            "React with **1–5** to vote. The actual songs are hidden until voting ends.",
            "",
        ]
        if ts:
            lines.append(f"**Ends:** <t:{ts}:R>")

        e = discord.Embed(description="\n".join(lines), color=0xF59E0B)
        e.set_thumbnail(url=self._default_thumb_url())
        return e

    def _build_embed(self, vote: dict) -> discord.Embed:
        instr_key = vote["instrument"]
        ends = _from_iso(vote.get("ends_at"))
        ts = int(ends.timestamp()) if ends else None

        pretty_instr = INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())

        # The three header lines you asked for:
        header_lines = [
            "🗳️ **Daily Vote — Score Snipe**",
            f"**Instrument:** {pretty_instr}",
            "React with **1–5** to vote. The actual songs are hidden until voting ends."
        ]

        # Show 1–5 with their hints
        lines = []
        for i, opt in enumerate(vote["options"][:5], start=1):
            emoji = NUM_EMOJIS[i-1]
            lines.append(f"{emoji} {opt['hint']}")

        desc = "\n".join(header_lines + [""] + lines + (["", f"**Ends:** <t:{ts}:R>"] if ts else []))

        e = discord.Embed(
            description=desc,
            color=0xF59E0B
        )
        e.set_thumbnail(url=self._default_thumb_url())  # top-right default image
        return e


    async def start_vote_now(self) -> bool:
        """Start a vote right now (1 hour). False if one is already active."""
        async with self.lock:
            v = self._vote()
            if v and _utcnow() < _from_iso(v["ends_at"]):
                return False

            instr_key = self.weekly._instrument_for_today(_utcnow())
            options = self._build_vote_options(instr_key)
            if not options:
                return False

            ends_at = _utcnow() + timedelta(hours=1)
            ch = await self._channel()
            if not ch:
                return False

            # Persisted vote payload (store both messages + channel)
            payload = {
                "instrument": instr_key,
                "options": options,
                "ends_at": _to_iso(ends_at),
                "channel_id": ch.id,
                "announce_message_id": None,
                "poll_message_id": None,
            }

            # 1) Send the announcement embed that HIDES the options and pings the role.
            role_id = getattr(self.bot, "ping_role_id", None)
            announce_msg = await ch.send(
                content=(f"<@&{role_id}>" if role_id else None),
                embed=self._build_announce_embed(payload),
                allowed_mentions=discord.AllowedMentions(roles=True, users=False, everyone=False),
            )
            payload["announce_message_id"] = str(announce_msg.id)

            # 2) Send the actual poll in a NEW message (no role ping here).
            poll = discord.Poll(
                question=f"Daily Vote — {INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())}",
                duration=timedelta(hours=1),
                multiple=False,
            )
            for opt in options[:10]:
                poll.add_answer(text=opt["hint"][:55])  # Poll answers themselves are still the hints

            poll_msg = await ch.send(poll=poll)
            payload["poll_message_id"] = str(poll_msg.id)

            # Save state and arm the finalizer
            self.weekly.state[self.VOTE_KEY] = payload
            _write_manager_state(self.weekly.state)

            if self._finalize_task and not self._finalize_task.done():
                self._finalize_task.cancel()
            self._finalize_task = asyncio.create_task(self._finalize_when_due())

            return True

    async def _finalize_when_due(self):
        try:
            while True:
                async with self.lock:
                    v = self._vote()
                    if not v:
                        return
                    due = _from_iso(v["ends_at"])
                    if not due:
                        return
                    now = _utcnow()
                    if now >= due:
                        break
                    await asyncio.sleep(min(60, max(1, int((due - now).total_seconds()))))
        except asyncio.CancelledError:
            return
        except Exception:
            return

        await self._finalize_vote()

    async def _finalize_vote(self):
        async with self.lock:
            v = self._vote()
            if not v:
                return

        ch = await self._channel()
        if not ch:
            return

        poll_msg_id = v.get("poll_message_id") or v.get("message_id")  # back-compat
        if not poll_msg_id:
            return

        msg = await ch.fetch_message(int(poll_msg_id))
        p = msg.poll

        # If still active, end it…
        if p and not p.is_finalized():
            try:
                await p.end()
                await asyncio.sleep(2)
                msg = await ch.fetch_message(msg.id)
                p = msg.poll
            except Exception:
                pass

        # Build counts aligned to Discord's displayed answer order
        counts: list[int] = []
        if p:
            try:
                counts = [self._answer_count(ans) for ans in p.answers]   # vote_count/count/votes
            except Exception:
                counts = []
        if not counts:
            # extreme fallback: no poll or unreadable → pretend all zeros
            counts = [0] * len(v.get("options", []))

        # Pick winner (tie-break randomly)
        if any(counts):
            maxc = max(counts)
            finalists = [i for i, c in enumerate(counts) if c == maxc]
            win_ix = random.choice(finalists)
        else:
            win_ix = random.randrange(max(1, len(v.get("options", [0]))))

        options = v["options"]
        winner  = options[win_ix]
        instr_key = v["instrument"]
        sid = int(winner["song_id"])
        song = self.bot.song_index.by_id.get(sid) if self.bot.song_index else None

        # Reveal message (map hints -> real titles; mark the winner)
        if msg:
            reveal_lines = []
            for i, opt in enumerate(options):
                s = self.bot.song_index.by_id.get(int(opt["song_id"])) if self.bot.song_index else None
                title = s.name if s else f"ID {opt['song_id']}"
                artist = s.artist if (s and s.artist) else "—"
                mark = "✅" if i == win_ix else "—"
                reveal_lines.append(f"{i+1}. **{opt['hint']}** → *{title}* by _{artist}_ {mark}")

            desc = "\n".join([
                f"**Instrument:** {INSTR_DISPLAY_NAMES.get(instr_key, instr_key.capitalize())}",
                "Results:",
                "",
                *reveal_lines,
            ])
            e = discord.Embed(title="🗳️ Daily Vote — Results", description=desc, color=0x10B981)
            try:
                await msg.reply(embed=e, mention_author=False)
            except Exception:
                pass

        # Create the weekly battle from the winning song
        if song and has_part(song, instr_key):
            rec = await self.weekly._create_one_battle(
                song=song,
                instr_key=instr_key,
                week_num=int(self.weekly.state.get("week") or 1),
            )
            if rec:
                await self.weekly.post_announcement([rec], increment_week=False)

        # Clear vote state
        async with self.lock:
            self.weekly.state[self.VOTE_KEY] = None
            _write_manager_state(self.weekly.state)

class SubscriptionManager:
    """Tracks battle subscriptions and sends DM alerts when 1st place changes hands."""
    def __init__(self, bot: "LBClient"):
        self.bot = bot
        self.lock = asyncio.Lock()
        self.task = asyncio.create_task(self._loop())

    # --- State helpers ---
    def _overtakes(self) -> Dict[str, int]:
        if not self.bot.manager:
            return {}
        return self.bot.manager.state.setdefault(OVERTAKES_KEY, {})

    def _subs(self) -> Dict[str, List[int]]:
        # subscriptions live in the same state file under SUBS_KEY
        if not self.bot.manager:
            return {}
        return self.bot.manager.state.setdefault(SUBS_KEY, {})

    def _tops(self) -> Dict[str, dict]:
        if not self.bot.manager:
            return {}
        return self.bot.manager.state.setdefault(TOPS_KEY, {})

    async def _save(self):
        # Reuse manager's lock if present; otherwise our own lock
        if self.bot.manager:
            async with self.bot.manager.lock:
                _write_manager_state(self.bot.manager.state)
        else:
            async with self.lock:
                # If manager isn't ready, nothing to save
                pass

    # --- Public API for the button ---
    async def toggle_subscription(self, battle_id: int, user_id: int, subscribe: bool) -> bool:
        if not self.bot.manager:
            return False
        bkey = str(battle_id)
        subs = self._subs()
        users = set(subs.get(bkey, []))
        if subscribe:
            users.add(user_id)
        else:
            users.discard(user_id)
        subs[bkey] = sorted(users)
        await self._save()
        return subscribe

    # --- Helpers ---
    def _extract_song_id(self, battle: dict) -> Optional[int]:
        sid = None
        if isinstance(battle.get("song_ids"), list) and battle["song_ids"]:
            sid = battle["song_ids"][0]
        elif isinstance(battle.get("song_id"), (int, str)):
            sid = battle["song_id"]
        elif isinstance(battle.get("description"), str):
            m = re.search(r"\bID\s*(\d{4,9})\b", battle["description"])
            if m:
                sid = m.group(1)
        try:
            return int(sid) if sid is not None else None
        except (TypeError, ValueError):
            return None

    async def _thumb_for_battle(self, battle: dict) -> Optional[str]:
        sid = self._extract_song_id(battle)
        if sid is None or not self.bot.song_index:
            return None
        song = self.bot.song_index.by_id.get(sid)
        if not song:
            return None
        return await fetch_song_art_url(self.bot.http_session, song.slug)  # type: ignore

    def _fmt_new_leader_line(self, top_row: dict) -> str:
        score = top_row.get("score")
        score_str = f"{int(score):,}" if isinstance(score, (int, float)) else "?"
        name = str(top_row.get("name", "Unknown"))
        return f"**{name}** — **{score_str}**"

    def _battle_instr_key(self, battle: Optional[dict]) -> Optional[str]:
        if not battle:
            return None
        v = battle.get("instrument")
        if isinstance(v, int):
            return INSTR_BY_ROLE_ID.get(v)
        if isinstance(v, str):
            if v.isdigit():
                return INSTR_BY_ROLE_ID.get(int(v))
            return v  # we store string keys in created_battles
        return None

    def _leader_key_for(self, battle: Optional[dict], top_row: Optional[dict]) -> Optional[str]:
        """Return a normalized leader key for baseline/changed checks."""
        if not top_row:
            return None
        instr = self._battle_instr_key(battle)
        if instr == "band":
            sc = top_row.get("score")
            try:
                return f"SCORE::{int(sc)}"
            except (TypeError, ValueError):
                return None
        # non-band: compare by name like before
        return f"NAME::{str(top_row.get('name', 'Unknown')).strip()}"

    def _baseline_or_changed(self, bkey: str, top_row: Optional[dict], battle: Optional[dict] = None) -> Optional[bool]:
        """None -> set baseline; True -> changed; False -> unchanged."""
        if not top_row:
            return None
        tops = self._tops()
        cur = tops.get(bkey)

        new_key = self._leader_key_for(battle, top_row)
        if not new_key:
            return None

        if not cur:
            return None  # first observation -> seed baseline

        # Back-compat with older records (had no 'leader_key')
        old_key = cur.get("leader_key")
        if not old_key:
            instr = self._battle_instr_key(battle)
            if instr == "band":
                # fall back to score if present, else name
                if "score" in cur and isinstance(cur["score"], int):
                    old_key = f"SCORE::{cur['score']}"
                else:
                    old_key = f"NAME::{cur.get('name','Unknown')}"
            else:
                old_key = f"NAME::{cur.get('name','Unknown')}"

        return new_key != old_key

    def _set_baseline(self, bkey: str, top_row: Optional[dict], battle: Optional[dict] = None):
        if not top_row:
            return
        sc = top_row.get("score")
        try:
            sc_int = int(sc) if isinstance(sc, (int, float, str)) and str(sc).isdigit() else None
        except Exception:
            sc_int = None
        record = {
            "leader_key": self._leader_key_for(battle, top_row),
            "name": str(top_row.get("name", "Unknown")),
            "score": sc_int,
        }
        self._tops()[bkey] = record

    async def _send_alert(
        self,
        battle: dict,
        rows: List[dict],
        new_top: dict,
        user_ids: List[int],
        *,
        broadcast_channel_id: Optional[int] = None,
        overtake_count: Optional[int] = None,
    ):
        # Table + header lines
        desc = format_battle_rows(rows or [], max_n=TOP_N)
        title = f"🥇 Overtake Alert — {battle.get('title','(untitled)')}"
        header_line = f"New leader: {self._fmt_new_leader_line(new_top)}"
        changes_line = f"Overtakes so far: **{overtake_count}**" if isinstance(overtake_count, int) else None
        head = "\n".join([l for l in (header_line, changes_line) if l])

        # 👇 NEW: put the song title as the very first line under the header
        sid = self._extract_song_id(battle)
        song = self.bot.song_index.by_id.get(sid) if (sid and self.bot.song_index) else None  # type: ignore
        song_title_line = f"# {song.name}" if song else f"# Song ID {sid or 'Unknown'}"

        embed = discord.Embed(
            title=title,
            description="\n".join([song_title_line, head, "", desc]),  # song title → header lines → blank → table
            color=0xF97316
        )

        # Thumbnail (fast path, verified in background)
        thumb = await self._thumb_for_battle(battle)
        if thumb:
            embed.set_thumbnail(url=thumb)

        # --- NEW: Song + instrument details (like other embeds) ---
        sid = self._extract_song_id(battle)
        song = self.bot.song_index.by_id.get(sid) if (sid and self.bot.song_index) else None  # type: ignore

        # instrument on battle is role_id (int); map back to our key
        role_val = battle.get("instrument")
        role_id = int(role_val) if isinstance(role_val, (int, float)) else (int(role_val) if isinstance(role_val, str) and role_val.isdigit() else None)
        instr_key = INSTR_BY_ROLE_ID.get(role_id) if role_id is not None else None
        # Human-readable instrument name for the embed
        instr_name = INSTR_DISPLAY_NAMES.get(instr_key or "", (instr_key or "Instrument").capitalize())

        limit = BAND_TOP_N if instr_key == "band" else TOP_N

        # Table + header lines
        desc = format_battle_rows(rows or [], max_n=limit)
        title = f"🥇 Overtake Alert — {battle.get('title','(untitled)')}"

        # If it's a band battle and there's a tie for top score, list the names.
        if instr_key == "band" and new_top:
            top_score = new_top.get("score") if new_top else None  # <-- fix 'Non' -> None
            tie_count = sum(1 for r in (rows or []) if r.get("score") == top_score)
            if tie_count > 1:
                tied_names = [str(r.get("name", "Unknown")) for r in rows if r.get("score") == top_score]
                embed.add_field(
                    name="Band",
                    value=", ".join(tied_names[:6]) + ("…" if len(tied_names) > 6 else ""),
                    inline=False
                )

        # difficulty from the song's ranks for this instrument
        rank_val = (song.ranks or {}).get(instr_key) if (song and instr_key) else None
        diff_txt = difficulty_label(_th_key(instr_key or ""), rank_val) if instr_key else "—"

        if song:
            artist = song.artist or "—"
            album_line = song.album_name or "—"
            if song.year:
                album_line = f"{album_line} ({song.year})"
            details = f"{pretty_genre(song.genre)} • {pretty_source(song.source)} • ID `{song.song_id}`"

            # Row 1: Artist | Album | pad
            embed.add_field(name="Artist", value=artist, inline=True)
            embed.add_field(name="Album", value=album_line, inline=True)
            embed.add_field(name="\u200B", value="\u200B", inline=True)  # pad

            # Row 2: Difficulty | Instrument | pad
            embed.add_field(name="Difficulty", value=diff_txt, inline=True)
            embed.add_field(name="Instrument", value=instr_name, inline=True)
            embed.add_field(name="\u200B", value="\u200B", inline=True)  # pad

            # Row 3: Details
            embed.add_field(name="Details", value=details, inline=False)
        else:
            # Fallback if we couldn't resolve the song
            fallback = f"Song ID: `{sid or 'Unknown'}`"
            embed.add_field(name="Details", value=fallback, inline=False)

        # --- Time tags (kept from your original) ---
        def _to_unix(v):
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str):
                s = v.strip()
                if s.isdigit():
                    return int(s)
                dt = _from_iso(s)
                return int(dt.timestamp()) if dt else None
            return None

        starts_ts = _to_unix(battle.get("starts_at"))
        ends_ts   = _to_unix(battle.get("expires_at"))
        embed.add_field(name="Started", value=(f"<t:{starts_ts}:R>" if starts_ts else "—"), inline=True)
        embed.add_field(name="Ends",    value=(f"<t:{ends_ts}:R>"   if ends_ts   else "—"), inline=True)

        # DM subscribers
        for uid in user_ids:
            try:
                user = await self.bot.fetch_user(uid)
                await user.send(embed=embed)
            except discord.Forbidden:
                pass
            except Exception:
                pass

        # Broadcast to channel, if configured
        if broadcast_channel_id:
            try:
                channel = self.bot.get_channel(broadcast_channel_id) or await self.bot.fetch_channel(broadcast_channel_id)
                await channel.send(
                    embed=embed,
                    allowed_mentions=discord.AllowedMentions(roles=False, users=False, everyone=False),
                )
            except Exception:
                pass


    async def _loop(self):
        await self.bot.wait_until_ready()
        while True:
            try:
                if not self.bot.http_session:
                    await asyncio.sleep(SUBSCRIPTIONS_LOOP_SECONDS);  continue

                battles = await fetch_battles(self.bot.http_session) or []
                now = _utcnow()

                # Index active battles
                active_by_id = {}
                for b in battles:
                    bid = b.get("battle_id")
                    if not isinstance(bid, int):
                        continue
                    dt = _from_iso(b.get("expires_at")) if isinstance(b.get("expires_at"), str) else None
                    if dt and dt <= now:
                        continue
                    active_by_id[str(bid)] = b

                subs = self._subs()
                tops = self._tops()

                # Clean up for ended battles (both subs and tops)
                ended_keys = [k for k in list(tops.keys()) if k not in active_by_id]
                for k in ended_keys:
                    tops.pop(k, None)
                ended_subs = [k for k in list(subs.keys()) if k not in active_by_id]
                for k in ended_subs:
                    subs.pop(k, None)
                if ended_keys or ended_subs:
                    await self._save()

                watch_all = bool(getattr(self.bot, "overtakes_channel_id", None))

                # If no subs and no broadcast channel, nap.
                if not subs and not watch_all:
                    await asyncio.sleep(SUBSCRIPTIONS_LOOP_SECONDS);  continue

                # Determine which battles to check:
                #   - always the subscribed ones
                #   - plus ALL active battles if we have a broadcast channel
                keys_to_check = set(subs.keys())
                if watch_all:
                    keys_to_check |= set(active_by_id.keys())

                for bkey in sorted(keys_to_check):
                    battle = active_by_id.get(bkey)
                    if not battle:
                        continue

                    bid = int(bkey)
                    top_row = await fetch_battle_top_winner(self.bot.http_session, bid)

                    # Decide whether to alert & update baseline
                    changed = self._baseline_or_changed(bkey, top_row, battle)
                    self._set_baseline(bkey, top_row, battle)

                    if changed:
                        count = self._overtakes().get(bkey, 0) + 1
                        self._overtakes()[bkey] = count
                        await self._save()

                        lb_rows = await fetch_battle_leaderboard(self.bot.http_session, bid, page_size=TOP_N) or []
                        user_ids = subs.get(bkey, [])
                        await self._send_alert(
                            battle,
                            lb_rows,
                            top_row,
                            user_ids,
                            broadcast_channel_id=getattr(self.bot, "overtakes_channel_id", None) if watch_all else None,
                            overtake_count=count,
                        )

                await asyncio.sleep(SUBSCRIPTIONS_LOOP_SECONDS)

            except Exception as e:
                if self.bot.manager:
                    async with self.bot.manager.lock:
                        self.bot.manager.state["last_error"] = f"subs watcher error: {e!r}"
                        _write_manager_state(self.bot.manager.state)
                await asyncio.sleep(5)

async def fetch_leaderboards(session: aiohttp.ClientSession, song_id: int, role_id: int) -> Optional[List[dict]]:
    params = {
        "song_id": str(song_id),
        "role_id": str(role_id),
        "page": "1",
        "page_size": "100"
    }
    try:
        async with session.get(LEADERBOARDS_URL, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            lb = data.get("leaderboard")
            if not isinstance(lb, list):
                return None
            return lb
    except asyncio.TimeoutError:
        return None
    except aiohttp.ClientError:
        return None

def is_admin():
    async def predicate(interaction: discord.Interaction) -> bool:
        admin_id = getattr(interaction.client, "admin_user_id", ADMIN_USER_ID)
        return interaction.user.id == admin_id
    return app_commands.check(predicate)

def format_battle_rows(
    rows: List[dict],
    *,
    start: int = 0,
    max_n: int = TOP_N,
    highlight_index: Optional[int] = None
) -> str:
    """Render a code-block table for *battle* leaderboards: placement, name, score only."""
    header = "  #  Player                   SCORE\n"
    lines: List[str] = []

    end = min(len(rows), start + max_n)
    for i in range(start, end):
        r = rows[i]
        rank = r.get("rank") or r.get("orank") or (i + 1)
        name = str(r.get("name", "?"))[:22]
        score = r.get("score")
        score_str = f"{int(score):,}" if isinstance(score, (int, float)) else "?"
        mark = ">>>" if (highlight_index is not None and i == highlight_index) else f"{rank:>3}"
        lines.append(f"{mark}  {name:<22}  {score_str:>8}")

    body = "\n".join(lines) if lines else "No entries."
    return f"```\n{header}{body}\n```"

def format_leaderboard_rows(
    rows: List[dict],
    *,
    start: int = 0,
    max_n: int = TOP_N,
    highlight_index: Optional[int] = None
) -> str:
    """Render a code-block table for leaderboard rows.
    start = global index offset; highlight_index = global index to mark with '>>>'.
    """
    header = "  #  Player                   %      SCORE\n"
    lines: List[str] = []

    end = min(len(rows), start + max_n)
    for i in range(start, end):
        r = rows[i]

        # safe fetches
        rank = r.get("rank") or r.get("orank") or "?"
        name = str(r.get("name", "?"))[:18]

        score = r.get("score")
        score_str = "?"
        if isinstance(score, (int, float)):
            score_str = f"{int(score):,}"

        pct = r.get("notes_pct")
        if isinstance(pct, (int, float)):
            pct_val = int(round(pct))
            pct_str = "FC" if pct_val >= 101 else f"{pct_val:>3d}"
        else:
            pct_str = "  -"

        diff_id = r.get("diff_id")
        diff_key = None
        if isinstance(diff_id, (int, float)):
            diff_key = int(diff_id)
        elif isinstance(diff_id, str) and diff_id.isdigit():
            diff_key = int(diff_id)
        diff = DIFF_LETTER.get(diff_key, "?")

        mark = ">>>" if (highlight_index is not None and i == highlight_index) else f"{rank:>3}"
        lines.append(f"{mark}  {name:<18}  {diff:<2}  {pct_str:>3}  {score_str:>8}")

    body = "\n".join(lines) if lines else "No entries."
    return f"```\n{header}{body}\n```"

class LBClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        # We don't need message content for slash commands; using Client avoids the warning.
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.song_index: Optional[SongIndex] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.manager: Optional[WeeklyBattleManager] = None
        # Configurable IDs
        ids = _load_discord_ids()
        self.admin_user_id = ids["admin_user_id"]
        self.announce_channel_id = ids["announce_channel_id"]
        self.ping_role_id = ids["ping_role_id"]
        self.overtakes_channel_id = ids["overtakes_channel_id"]
        self.dev_guild_id = ids["guild_id"]
        self.dev_guild = discord.Object(id=self.dev_guild_id) if self.dev_guild_id else None
        self.subscriptions: Optional[SubscriptionManager] = None

    async def setup_hook(self) -> None:
        self.song_index = load_song_map(SONG_MAP_PATH)
        self.http_session = aiohttp.ClientSession()
        self.manager = WeeklyBattleManager(self)
        self.daily_vote = DailyVoteManager(self, self.manager)
        self.subscriptions = SubscriptionManager(self)

        # --- Guild-only commands, like your example bot ---
        if getattr(self, "dev_guild_id", None):
            guild = discord.Object(id=self.dev_guild_id)

            # 1) Rebuild the guild command set from your global definitions
            #    (clears old guild copies to avoid stale dupes)
            self.tree.clear_commands(guild=guild)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            print(f"Synced {len(synced)} guild commands to {self.dev_guild_id}")

            # 2) PURGE GLOBAL COMMANDS so they don't also appear in the guild
            #    (safe to run every boot; removes any globally-registered duplicates)
            self.tree.clear_commands(guild=None)   # ← keyword-only arg; None targets GLOBAL registry
            try:
                await self.tree.sync()             # push the empty global set to Discord (deletes globals)
                print("Cleared global commands")
            except Exception as e:
                print(f"[WARN] Clearing global commands failed: {e!r}")
        else:
            # Fallback: no guild configured → keep global sync
            synced = await self.tree.sync()
            print(f"Synced {len(synced)} global commands")

    async def close(self) -> None:
        if self.http_session:
            await self.http_session.close()
        await super().close()


bot = LBClient()

class BattleListView(discord.ui.View):
    def __init__(self, battles: List[dict], user: Optional[discord.User] = None, *, show_subscribe: bool = False):
        super().__init__(timeout=60)
        self.user = user
        self.battles = battles or []
        self.page = 0
        self.leaderboards: Dict[int, List[dict]] = {}
        self.message: Optional[discord.Message] = None
        self.show_subscribe = show_subscribe
        self._refresh_buttons()
        self._warm_task = asyncio.create_task(self._warm_cache())

    async def _warm_cache(self):
        if not bot.http_session:
            return
        sem = asyncio.Semaphore(4)  # be nice to the API

        async def fetch_one(battle: dict):
            bid = battle.get("battle_id")
            if not isinstance(bid, int):
                return
            # detect band
            role_val = battle.get("instrument")
            role_id = int(role_val) if isinstance(role_val, (int, float)) else (int(role_val) if isinstance(role_val, str) and role_val.isdigit() else None)
            instr_key = INSTR_BY_ROLE_ID.get(role_id) if role_id is not None else None
            size = BAND_TOP_N if instr_key == "band" else TOP_N
            async with sem:
                if bid not in self.leaderboards:
                    rows = await fetch_battle_leaderboard(bot.http_session, bid, page_size=size)
                    if rows:
                        self.leaderboards[bid] = rows
            # also warm the thumbnail cache (non-blocking verification)
            try:
                _ = await self._thumb_for_battle(battle)
            except Exception:
                pass

        # Prefetch everything (or slice if you expect dozens)
        await asyncio.gather(*(fetch_one(b) for b in self.battles))

    def _refresh_buttons(self):
        self.clear_items()
        total_pages = max(1, len(self.battles))
        self.add_item(BattlePrevButton(disabled=(self.page <= 0)))
        self.add_item(PageIndicator(label=f"{self.page + 1}/{total_pages}"))
        self.add_item(BattleNextButton(disabled=(self.page >= total_pages - 1)))

        #subscribe only for active battles, else show "Expired" chip
        if self.show_subscribe and self.battles:
            battle = self.battles[self.page]
            ends_ts = self._to_unix(battle.get("expires_at"))
            now_ts = int(_utcnow().timestamp())
            is_expired = (ends_ts is not None) and (ends_ts <= now_ts)

            if is_expired:
                self.add_item(ExpiredIndicator())
            else:
                bid = battle.get("battle_id")
                bkey = str(bid)
                subs = (bot.manager.state.get(SUBS_KEY, {}) if bot.manager else {})
                is_subbed = self.user.id in set(subs.get(bkey, [])) if self.user else False
                self.add_item(SubscribeToggleButton(is_subbed))

    def _to_unix(self, v) -> Optional[int]:
        # Accept int/float epochs, digit-strings, or ISO8601 strings
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, str):
            v = v.strip()
            if v.isdigit():
                return int(v)
            try:
                dt = _from_iso(v)
                if dt:
                    return int(dt.timestamp())
            except Exception:
                pass
        return None

    def _extract_song_id(self, battle: dict) -> Optional[int]:
        """Best-effort: look for song_ids, song_id, or parse from description '(ID 1234567)'."""
        sid = None
        if isinstance(battle.get("song_ids"), list) and battle["song_ids"]:
            sid = battle["song_ids"][0]
        elif isinstance(battle.get("song_id"), (int, str)):
            sid = battle["song_id"]
        elif isinstance(battle.get("description"), str):
            m = re.search(r"\bID\s*(\d{4,9})\b", battle["description"])
            if m:
                sid = m.group(1)
        try:
            return int(sid) if sid is not None else None
        except (TypeError, ValueError):
            return None

    async def _thumb_for_battle(self, battle: dict) -> Optional[str]:
        """Return a song art URL for this battle, if we can resolve the song."""
        sid = self._extract_song_id(battle)
        if sid is None or not bot.song_index:
            return None
        song = bot.song_index.by_id.get(sid)
        if not song:
            return None
        # fetch_song_art_url already falls back to DEFAULT_SONG_ART if missing
        # FAST: never block on network here
        url = get_song_art_url_fast(song.slug)
        # optional: verify once in background; harmless if scheduled multiple times
        if bot.http_session:
            asyncio.create_task(_verify_art_url(bot.http_session, song.slug))
        return url

    async def build_embed(self) -> discord.Embed:
        if not self.battles:
            return discord.Embed(title="Battles", description="No active battles.")

        battle = self.battles[self.page]
        bid = battle.get("battle_id")

        lb = self.leaderboards.get(bid)
        if not lb:  # None or []
            lb = await fetch_battle_leaderboard(bot.http_session, bid, page_size=TOP_N)  # type: ignore
            if lb:  # only cache when we actually have rows
                self.leaderboards[bid] = lb

        # Resolve song (and thumbnail)
        sid = self._extract_song_id(battle)
        song = bot.song_index.by_id.get(sid) if (sid and bot.song_index) else None  # type: ignore
        # detect band
        role_val = battle.get("instrument")
        role_id = int(role_val) if isinstance(role_val, (int, float)) else (int(role_val) if isinstance(role_val, str) and role_val.isdigit() else None)
        instr_key = INSTR_BY_ROLE_ID.get(role_id) if role_id is not None else None
        max_n = BAND_TOP_N if instr_key == "band" else TOP_N

        desc_table = format_battle_rows(lb or [], max_n=max_n)

        parts = []
        if song:
            parts.append(f"# {song.name}")
        else:
            parts.append(f"# Song ID {sid or 'Unknown'}")
        parts.append(desc_table)

        if song:
            artist = song.artist or "—"
            album_line = song.album_name or "—"
            if song.year:
                album_line = f"{album_line} ({song.year})"
            parts.append("\n".join([
                f"**Artist:** {artist}",
                f"**Album:** {album_line}",
            ]))

        description = "\n\n".join(parts)
        embed = discord.Embed(
            title=f"Battle: {battle.get('title','(untitled)')}",
            description=description
        )

        # Thumbnail
        thumb = await self._thumb_for_battle(battle)
        if thumb:
            embed.set_thumbnail(url=thumb)

        # Started / Ends
        starts_ts = self._to_unix(battle.get("starts_at"))
        ends_ts   = self._to_unix(battle.get("expires_at"))
        embed.add_field(name="Started", value=(f"<t:{starts_ts}:R>" if starts_ts else "—"), inline=True)
        embed.add_field(name="Ends",    value=(f"<t:{ends_ts}:R>"   if ends_ts   else "—"), inline=True)

        # Footer (robust even if song isn't resolved)
        footer_txt = f"Artist: {song.artist} • Song ID: {song.song_id}" if song else f"Song ID: {sid or 'Unknown'}"
        embed.set_footer(text=footer_txt)

        try:
            import re as _re
            m = _re.search(r"Week\s+(\d+)", str(battle.get('title','')))
            if m:
                week_txt = m.group(1)
        except Exception:
            pass
        footer_txt = (
            f"Artist: {song.artist} • Song ID: {song.song_id}"
            if song
            else f"Song ID: {sid or 'Unknown'}"
        )

        # Optionally include "Week N" if present in the battle title
        if song:
            footer_txt = f"Genre: {pretty_genre(song.genre)} • Source: {pretty_source(song.source)} • Song ID: {song.song_id}"
        else:
            footer_txt = f"Song ID: {sid or 'Unknown'}"

        embed.set_footer(text=footer_txt)

        self._refresh_buttons()
        return embed

    async def update(self, interaction: discord.Interaction):
        # Ownership check
        if self.user and interaction.user != self.user:
            # We're likely already deferred; use followup
            await interaction.followup.send("This battle view isn't yours.", ephemeral=True)
            return

        embed = await self.build_embed()

        # If we haven't replied yet, use the normal response API
        if not interaction.response.is_done():
            try:
                await interaction.response.edit_message(embed=embed, view=self)
                return
            except discord.NotFound:
                pass  # fall through to followup path

        # Otherwise edit the original message via followup (or our saved handle)
        try:
            if self.message:
                await self.message.edit(embed=embed, view=self)
            else:
                # fallback: edit the message this component came from
                await interaction.followup.edit_message(
                    message_id=interaction.message.id, embed=embed, view=self
                )
        except discord.NotFound:
            # final fallback if the message vanished; send a new one
            await interaction.followup.send(embed=embed, view=self)

    async def on_timeout(self):
        try:
            for item in self.children:
                item.disabled = True
            if self.message:
                await self.message.edit(view=self)
        except Exception:
            pass


class LeaderboardView(discord.ui.View):
    def __init__(self, all_rows: List[dict], page_size: int, song_title: str, artist: str, instrument: str,
                 year: Optional[int] = None, genre: Optional[str] = None, user: Optional[discord.User] = None,
                 thumb_url: Optional[str] = None):
        super().__init__(timeout=90)  # 1 minute
        self.authorized_user = user
        self.all_rows = all_rows
        self.page_size = page_size
        self.page = 0
        self.highlight_index: Optional[int] = None
        self.song_title = song_title
        self.artist = artist
        self.instrument = instrument
        self.year = year
        self.genre = genre
        self.thumb_url = thumb_url
        self.message: Optional[discord.Message] = None
        self._refresh_buttons()

    def _refresh_buttons(self):
        self.clear_items()
        total_pages = max(1, (len(self.all_rows) + self.page_size - 1) // self.page_size)
        self.add_item(PrevButton(disabled=(self.page <= 0)))
        self.add_item(PageIndicator(label=f"{self.page + 1}/{total_pages}"))
        self.add_item(NextButton(disabled=(self.page >= total_pages - 1)))
        self.add_item(FindPlayerButton())

    async def update_message(self, interaction: Optional[discord.Interaction] = None):
        start = self.page * self.page_size
        desc = format_leaderboard_rows(self.all_rows, start=start, max_n=self.page_size, highlight_index=self.highlight_index)
        embed = discord.Embed(title=f"Leaderboards — {self.song_title} — {self.instrument}", description=desc)
        if self.thumb_url:
            embed.set_thumbnail(url=self.thumb_url)
        footer_parts = [f"Artist: {self.artist}"]
        if self.year:
            footer_parts.append(f"Year: {self.year}")
        if self.genre:
            footer_parts.append(f"Genre: {self.genre}")
        embed.set_footer(text=" • ".join(footer_parts))
        self._refresh_buttons()
        if interaction:
            await interaction.response.edit_message(embed=embed, view=self)
        elif self.message:
            await self.message.edit(embed=embed, view=self)

    async def on_timeout(self):
        try:
            for item in self.children:
                item.disabled = True
            if self.message:
                await self.message.edit(view=self)
        except Exception:
            pass


class PageIndicator(discord.ui.Button):
    def __init__(self, label: str):
        super().__init__(label=label, style=discord.ButtonStyle.secondary, disabled=True)

class PrevButton(discord.ui.Button):
    def __init__(self, disabled: bool = False):
        super().__init__(label="Prev", style=discord.ButtonStyle.secondary, disabled=disabled)
    async def callback(self, interaction: discord.Interaction):
        view: LeaderboardView = self.view  # type: ignore
        if interaction.user != view.authorized_user:
            await interaction.response.send_message(
                "This leaderboard view isn't yours — use `/leaderboards` to start your own session.",
                ephemeral=True
            )
            return
        if view.page > 0:
            view.page -= 1
        await view.update_message(interaction)

class NextButton(discord.ui.Button):
    def __init__(self, disabled: bool = False):
        super().__init__(label="Next", style=discord.ButtonStyle.secondary, disabled=disabled)
    async def callback(self, interaction: discord.Interaction):
        view: LeaderboardView = self.view  # type: ignore
        if interaction.user != view.authorized_user:
            await interaction.response.send_message(
                "This leaderboard view isn't yours — use `/leaderboards` to start your own session.",
                ephemeral=True
            )
            return
        total_pages = max(1, (len(view.all_rows) + view.page_size - 1) // view.page_size)
        if view.page < total_pages - 1:
            view.page += 1
        await view.update_message(interaction)

class FindPlayerButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="Find Player", style=discord.ButtonStyle.primary)
    async def callback(self, interaction: discord.Interaction):
        view: LeaderboardView = self.view  # type: ignore
        if interaction.user != view.authorized_user:
            await interaction.response.send_message(
                "This leaderboard view isn't yours — use `/leaderboards` to start your own session.",
                ephemeral=True
            )
            return
        await interaction.response.send_modal(FindPlayerModal(view))

class ExpiredIndicator(discord.ui.Button):
    def __init__(self):
        super().__init__(label="Expired", style=discord.ButtonStyle.secondary, disabled=True)

class SubscribeToggleButton(discord.ui.Button):
    def __init__(self, is_subscribed: bool):
        label = "Unsubscribe" if is_subscribed else "Subscribe"
        style = discord.ButtonStyle.danger if is_subscribed else discord.ButtonStyle.success  # red when subscribed
        super().__init__(label=label, style=style)

    async def callback(self, interaction: discord.Interaction):
        view: BattleListView = self.view  # type: ignore
        if view.user and interaction.user.id != view.user.id:
            await interaction.response.send_message("This view isn't yours.", ephemeral=True)
            return
        if not view.battles:
            await interaction.response.send_message("No battle here.", ephemeral=True)
            return

        battle = view.battles[view.page]
        bid = battle.get("battle_id")
        if not isinstance(bid, int):
            await interaction.response.send_message("Invalid battle.", ephemeral=True)
            return

        # Determine current state
        subs = (bot.manager.state.get(SUBS_KEY, {}) if bot.manager else {})
        users = set(subs.get(str(bid), []))
        want_subscribe = interaction.user.id not in users

        if not bot.subscriptions:
            await interaction.response.send_message("Subscriptions not ready.", ephemeral=True)
            return

        await bot.subscriptions.toggle_subscription(bid, interaction.user.id, subscribe=want_subscribe)

        # Re-render buttons with new state & remind about DMs when subscribing
        view._refresh_buttons()
        embed = await view.build_embed()
        await interaction.response.edit_message(embed=embed, view=view)

        if want_subscribe:
            await interaction.followup.send(
                "✅ Subscribed. Make sure your DMs are open to receive alerts:\n"
                "• **User Settings → Privacy & Safety → Allow direct messages from server members**\n"
                "• And per-server Privacy Settings if needed.",
                ephemeral=True
            )
        else:
            await interaction.followup.send("🚫 Unsubscribed.", ephemeral=True)


class FindPlayerModal(discord.ui.Modal, title="Find a player"):
    def __init__(self, view: 'LeaderboardView'):
        super().__init__()
        self.view = view
        self.player = discord.ui.TextInput(label="Player name", placeholder="e.g., Lewis, Kyle", required=True, max_length=64)
        self.add_item(self.player)
    async def on_submit(self, interaction: discord.Interaction):
        query = str(self.player.value).strip().lower()
        idx = None
        for i, r in enumerate(self.view.all_rows):
            name = str(r.get("name", "")).lower()
            if query in name:
                idx = i
                break
        if idx is None:
            await interaction.response.send_message(f"No player matching '{query}'.", ephemeral=True)
            return
        self.view.highlight_index = idx
        self.view.page = idx // self.view.page_size
        await self.view.update_message(interaction)

class SongSelectView(discord.ui.View):
    def __init__(self, matches: List[Song], instrument: str, interaction: discord.Interaction, callback_func):
        super().__init__(timeout=60)  # 1 minute
        options = [
            discord.SelectOption(label=f"{s.name}", description=s.artist[:100], value=str(i))
            for i, s in enumerate(matches)
        ]
        self.add_item(SongSelect(options, matches, instrument, interaction, callback_func))
        self.message: Optional[discord.Message] = None

    async def on_timeout(self):
        try:
            for item in self.children:
                item.disabled = True
            if self.message:
                await self.message.edit(view=self)
        except Exception:
            pass

    def _refresh_buttons(self):
        self.clear_items()
        total_pages = max(1, len(self.battles))
        self.add_item(BattlePrevButton(disabled=(self.page == 0)))
        self.add_item(PageIndicator(label=f"{self.page + 1}/{total_pages}"))
        self.add_item(BattleNextButton(disabled=(self.page >= total_pages - 1)))

    async def build_embed(self) -> discord.Embed:
        battle = self.battles[self.page]
        bid = battle["battle_id"]
        lb = self.leaderboards.get(bid)

        if lb is None:
            lb = await fetch_battle_leaderboard(bot.http_session, bid)
            self.leaderboards[bid] = lb or []

        desc = format_leaderboard_rows(lb or [], max_n=TOP_N)
        embed = discord.Embed(title=f"Battle: {battle['title']}", description=desc)
        embed.add_field(name="Description", value=battle.get("description", "No description"), inline=False)

        # Format Discord live time tags
        embed.add_field(name="Starts", value=f"<t:{battle['starts_at']}:R>", inline=True)
        embed.add_field(name="Ends", value=f"<t:{battle['expires_at']}:R>", inline=True)

        self._refresh_buttons()
        return embed

    async def update(self, interaction: discord.Interaction):
        if interaction.user != self.user:
            await interaction.response.send_message("This battle view isn't yours.", ephemeral=True)
            return
        embed = await self.build_embed()
        await interaction.response.edit_message(embed=embed, view=self)

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True
        if self.message:
            await self.message.edit(view=self)

class BattlePrevButton(discord.ui.Button):
    def __init__(self, disabled=False):
        super().__init__(label="Prev", style=discord.ButtonStyle.secondary, disabled=disabled)

    async def callback(self, interaction: discord.Interaction):
        view: BattleListView = self.view  # type: ignore
        # ACK immediately to avoid 10062
        await interaction.response.defer()
        if view.page > 0:
            view.page -= 1
        await view.update(interaction)


class BattleNextButton(discord.ui.Button):
    def __init__(self, disabled=False):
        super().__init__(label="Next", style=discord.ButtonStyle.secondary, disabled=disabled)

    async def callback(self, interaction: discord.Interaction):
        view: BattleListView = self.view  # type: ignore
        # ACK immediately to avoid 10062
        await interaction.response.defer()
        if view.page < len(view.battles) - 1:
            view.page += 1
        await view.update(interaction)

class VoteButton(discord.ui.Button):
    def __init__(self, manager: "DailyVoteManager", vote_id: str, idx: int):
        super().__init__(
            label=str(idx),
            style=discord.ButtonStyle.primary,
            custom_id=f"dailyvote:{vote_id}:{idx}"  # persistent across restarts
        )
        self.manager = manager
        self.vote_id = vote_id
        self.idx = idx

    async def callback(self, interaction: discord.Interaction):
        # defer early so we don't hit the 3s window
        await interaction.response.defer(ephemeral=True, thinking=False)
        ok, msg = await self.manager.register_vote(self.vote_id, interaction.user.id, self.idx)
        try:
            # Always try to update the originating message with the latest tallies
            await self.manager.refresh_vote_message_from_interaction(interaction)
        except Exception:
            pass
        if ok:
            await interaction.followup.send(f"✅ Your vote for **#{self.idx}** is recorded.", ephemeral=True)
        else:
            await interaction.followup.send(f"⚠️ {msg}", ephemeral=True)


class DailyVoteView(discord.ui.View):
    """Persistent view so buttons work after bot restarts."""
    def __init__(self, manager: "DailyVoteManager", vote_id: str):
        super().__init__(timeout=None)  # persistent
        for i in range(1, 6):
            self.add_item(VoteButton(manager, vote_id, i))

class SongSelect(discord.ui.Select):
    def __init__(self, options: List[discord.SelectOption], matches: List[Song], instrument: str, interaction: discord.Interaction, callback_func):
        super().__init__(placeholder="Choose the correct song...", options=options)
        self.matches = matches
        self.instrument = instrument
        self.original_interaction = interaction
        self.callback_func = callback_func  # custom continuation logic

    async def callback(self, interaction: discord.Interaction):
        # Let the continuation handle deferring & responding.
        selected_index = int(self.values[0])
        selected_song = self.matches[selected_index]

        # Pass a normalized instrument to the continuation.
        resolved = resolve_instrument(self.instrument)
        await self.callback_func(interaction, selected_song, resolved)

@bot.tree.command(name="leaderboards", description="Search GoCentral leaderboards by song name/shortname/song_id.")
@app_commands.describe(song="Song name (use quotes for exact match)", instrument="Instrument (guitar, bass, drums, vocals, keys)")
async def leaderboards_cmd(interaction: discord.Interaction, song: str, instrument: str = "guitar"):
    await interaction.response.defer(ephemeral=True, thinking=True)

    if not bot.song_index:
        await interaction.followup.send("Song map not loaded.")
        return

    resolved = resolve_instrument(instrument)
    if not resolved or resolved not in INSTR_ROLE_IDS:
        valid = ", ".join(sorted(INSTR_ROLE_IDS.keys()))
        await interaction.followup.send(f"Unknown instrument '{instrument}'. Try one of: {valid}.")
        return

    candidates = bot.song_index.find_all(song)
    if not candidates:
        await interaction.followup.send(f"No song match for: {song}")
        return
    elif len(candidates) > 1:
        async def continue_leaderboard(inter: discord.Interaction, song_obj: Song, resolved_instr: str):
            if not inter.response.is_done():
                await inter.response.defer(thinking=True)

            if not resolved_instr or resolved_instr not in INSTR_ROLE_IDS:
                await inter.followup.send("Invalid instrument.", ephemeral=True)
                return

            role_id = INSTR_ROLE_IDS[resolved_instr]
            rows = await fetch_leaderboards(bot.http_session, song_obj.song_id, role_id)
            if rows is None:
                await inter.followup.send("No leaderboard data returned.")
                return

            thumb = await fetch_song_art_url(bot.http_session, song_obj.slug)

            view = LeaderboardView(
                all_rows=rows,
                page_size=TOP_N,
                song_title=song_obj.name,
                artist=song_obj.artist,
                instrument=INSTR_DISPLAY_NAMES.get(resolved_instr, resolved_instr.capitalize()),
                year=song_obj.year,
                genre=song_obj.genre,
                user=inter.user,
                thumb_url=thumb,
            )

            desc = format_leaderboard_rows(rows, start=0, max_n=TOP_N)
            embed = discord.Embed(
                title=f"Leaderboards — {song_obj.name} — {INSTR_DISPLAY_NAMES.get(resolved_instr, resolved_instr.capitalize())}",
                description=desc
            )
            embed.set_thumbnail(url=thumb)
            footer_parts = [f"Artist: {song_obj.artist}"]
            if song_obj.year:
                footer_parts.append(f"Year: {song_obj.year}")
            if song_obj.genre:
                footer_parts.append(f"Genre: {song_obj.genre}")
            embed.set_footer(text=" • ".join(footer_parts))

            msg = await inter.followup.send(embed=embed, view=view)
            view.message = msg
            # the message object is created by followup; if you need it later, capture the return value

        # Pass the *resolved* instrument into the select view
        view = SongSelectView(candidates, resolved, interaction, continue_leaderboard)
        msg = await interaction.followup.send(
            f"Multiple songs matched your query for '{song}'. Please select one:",
            view=view
        )
        view.message = msg
        return

    else:
        s = candidates[0]


    role_id = INSTR_ROLE_IDS[resolved]

    assert bot.http_session is not None
    rows = await fetch_leaderboards(bot.http_session, s.song_id, role_id)
    if rows is None:
        await interaction.followup.send("No leaderboard data returned.")
        return

    thumb = await fetch_song_art_url(bot.http_session, s.slug)

    # build the view first (so page_size is known)
    view = LeaderboardView(
        all_rows=rows,
        page_size=TOP_N,
        song_title=s.name,
        artist=s.artist,
        instrument=INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize()),
        year=s.year,
        genre=s.genre,
        user=interaction.user,
        thumb_url=thumb,
    )

    start_index = view.page * view.page_size
    desc = format_leaderboard_rows(rows, start=start_index, max_n=view.page_size, highlight_index=view.highlight_index)

    embed = discord.Embed(
        title=f"Leaderboards — {s.name} — {INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize())}",
        description=desc
    )
    embed.set_thumbnail(url=thumb)
    embed.set_footer(text=f"Artist: {s.artist} • Song ID: {s.song_id}")

    channel = interaction.channel
    posted_public = False

    # Try to post publicly if we have permission
    msg_obj = None
    try:
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            me = interaction.guild.me if interaction.guild else None
            if me is not None:
                perms = channel.permissions_for(me)
                can_send = perms.send_messages and (not isinstance(channel, discord.Thread) or perms.send_messages_in_threads)
                can_embed = perms.embed_links
                if can_send and can_embed:
                    msg_obj = await channel.send(embed=embed, view=view)
                    view.message = msg_obj
                    posted_public = True
    except Forbidden:
        posted_public = False

    if posted_public:
        try:
            await interaction.delete_original_response()
        except Exception:
            pass
        return
    else:
        # Fallback: still post leaderboard publicly in DMs or last-resort context
        try:
            msg_obj = await interaction.user.send("Here's your leaderboard result:", embed=embed, view=view)
            view.message = msg_obj
            await interaction.followup.send("Couldn't post in channel, so I DMed you instead.", ephemeral=True)
        except discord.Forbidden:
            await interaction.followup.send("I couldn't post in the channel or DM you. Please check my permissions.")

@bot.tree.command(name="battles", description="View active battles and top players.")
@app_commands.describe(song="(optional) Song name, shortname, or numeric ID to jump to")
async def battles_cmd(interaction: discord.Interaction, song: Optional[str] = None):
    await interaction.response.defer()

    if not bot.http_session:
        await interaction.followup.send("HTTP session not initialized.")
        return

    battles = await fetch_battles(bot.http_session)
    if not battles:
        await interaction.followup.send("No active battles found.")
        return

    now = _utcnow()
    def _is_expired(b: dict) -> bool:
        dt = _as_dt(b.get("expires_at"))  # accepts epoch/ISO
        return bool(dt and dt <= now)

    active = [b for b in battles if not _is_expired(b)]
    expired = [b for b in battles if _is_expired(b)]
    battles = active + expired

    view = BattleListView(battles, interaction.user, show_subscribe=True)

    # ---------- auto-jump ----------
    def _norm(s: str) -> str:
        try:
            return SongIndex._norm(s)
        except Exception:
            return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

    def _extract_song_id(b: dict) -> Optional[int]:
        # copy of BattleListView._extract_song_id so we can use it before building the view
        sid = None
        if isinstance(b.get("song_ids"), list) and b["song_ids"]:
            sid = b["song_ids"][0]
        elif isinstance(b.get("song_id"), (int, str)):
            sid = b["song_id"]
        elif isinstance(b.get("description"), str):
            m = re.search(r"\bID\s*(\d{4,9})\b", b["description"])
            if m:
                sid = m.group(1)
        try:
            return int(sid) if sid is not None else None
        except (TypeError, ValueError):
            return None

    if song:
        q = song.strip()
        qn = _norm(q)
        candidate_ids: set[int] = set()

        # numeric?
        if q.isdigit():
            try:
                candidate_ids.add(int(q))
            except ValueError:
                pass

        # lookup via SongIndex
        if bot.song_index:
            for s in bot.song_index.find_all(q, max_results=8):
                candidate_ids.add(int(s.song_id))

        # 1) exact ID hit → first match wins
        jump_idx = None
        for i, b in enumerate(battles):
            sid = _extract_song_id(b)
            if sid is not None and sid in candidate_ids:
                jump_idx = i
                break

        # 2) title contains query
        if jump_idx is None and bot.song_index:
            for i, b in enumerate(battles):
                sid = _extract_song_id(b)
                song_obj = bot.song_index.by_id.get(sid) if sid else None
                if song_obj and qn and qn in _norm(song_obj.name):
                    jump_idx = i
                    break

        # 3) description contains query (e.g., "Drive — Incubus (ID 1010546)")
        if jump_idx is None:
            for i, b in enumerate(battles):
                if qn and qn in _norm(str(b.get("description", ""))):
                    jump_idx = i
                    break

        if jump_idx is not None:
            view.page = jump_idx
    # ---------- end auto-jump ----------

    embed = await view.build_embed()
    view.message = await interaction.followup.send(embed=embed, view=view)

async def continue_battle_create(interaction: discord.Interaction, song_obj: Song, instrument: str, title: str, description: str):
    if not interaction.response.is_done():
        await interaction.response.defer(ephemeral=True)

    api_key = get_api_key()
    if not api_key:
        await interaction.followup.send("API key not configured.")
        return

    # instrument is already resolved; convert to role_id int
    role_id = INSTR_ROLE_IDS.get(instrument)
    if role_id is None:
        await interaction.followup.send("Invalid instrument.", ephemeral=True)
        return

    if not has_part(song_obj, instrument):
        pretty = INSTR_DISPLAY_NAMES.get(instrument, instrument.capitalize())
        await interaction.followup.send(
            f"❌ That song has no **{pretty}** part (rank is 0 or missing). Choose another song or instrument.",
            ephemeral=True
        )
        return

    starts_at = datetime.utcnow().isoformat() + "Z"
    expires_at = (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z"

    payload = {
        "title": title,
        "description": description,
        "song_ids": [song_obj.song_id],
        "starts_at": starts_at,
        "expires_at": expires_at,
        "instrument": role_id,
        "flags": 0
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with bot.http_session.post(CREATE_BATTLE_URL, json=payload, headers=headers, timeout=10) as resp:
            raw = await resp.text()  # read once
            ok = 200 <= resp.status < 300

            # Try to parse JSON, but don't crash if it's plain text
            data = None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                pass

            if ok and (not data or data.get("success") is True):
                battle_id = data.get("battle_id") if isinstance(data, dict) else None
                nice_instr = INSTR_DISPLAY_NAMES.get(instrument, instrument)
                suffix = f" (ID `{battle_id}`)" if battle_id is not None else ""
                await interaction.followup.send(f"✅ Battle created for **{song_obj.name}** — {nice_instr}{suffix}.")
                # Announce this battle in the configured channel (use current week, don't bump)
                if bot.manager:
                    rec = {
                        "battle_id": battle_id,
                        "song_id": song_obj.song_id,
                        "instrument": instrument,
                        "created_at": starts_at,
                        "expires_at": expires_at,
                        "title": title,
                    }
                    await bot.manager.post_announcement([rec], increment_week=False)
                    if bot.manager and battle_id is not None:
                        rec = {
                            "battle_id": int(battle_id),
                            "song_id": song_obj.song_id,
                            "instrument": instrument,
                            "created_at": starts_at,
                            "expires_at": expires_at,
                            "title": title,
                            "week": bot.manager.state.get("week") or 0,
                            "winner_announced": False,
                            "winner": None,
                            "overtakes": 0,
                        }
                        async with bot.manager.lock:
                            bot.manager.state["created_battles"].append(rec)
                            _write_manager_state(bot.manager.state)

            else:
                await interaction.followup.send(f"❌ Failed to create battle: {resp.status} - {raw}")

    except Exception as e:
        await interaction.followup.send(f"Exception while creating battle: {e}")

@bot.tree.command(name="info", description="What I can do + weekly battle rules.")
async def info_cmd(interaction: discord.Interaction):
    # keep it tidy for the channel
    await interaction.response.defer()

    # pull current limits if available
    max_battles = 6
    try:
        if bot.manager:
            max_battles = int(bot.manager.state.get("songs_per_run") or 6)
    except Exception:
        pass

    # what I do (now includes link/unlink)
    what_i_do = [
        "`/battles` — browse active battles, and subscribe for notifications",
        "`/leaderboards <song> [instrument]` — view a song’s top scores",
        "`/link <leaderboard name>` — link your Discord to your in-game leaderboard name",
        "`/unlink <leaderboard name>` — remove a linked name",
        "`/info` — print this info",
    ]

    # daily format (kept)
    weekly_core = "Mon–Sat, not Wed: one battle daily cycling **Guitar, Bass, Drums, Vocals, Band** (no repeats until all used)"
    weekly_pro  = "Wed & Sun: one **random Pro** battle each day **(Pro Guitar/Pro Bass/Bro Drums/Harmonies/Pro Keys)**, cycling without repeats"

    rules = [
        "**Official content + Rock Band Network** only for now",
        "No **Festival** or **Beatles** songs. RB4 is allowed",
        "No **2x bass pedal** versions",
    ]

    quick_start = [
        "Open `/battles`, pick a card, play the chart, post your best.",
        "In game, go to `Quickplay > Setlists / Battles` to find the battle.",
        "Playing the song by itself in normal quickplay will **NOT** count toward the battle.",
        "Use `/leaderboards` anytime to check global standings.",
        "Check out <https://rbpaths.github.io/> for pathing!",
    ]

    # NEW: linking section
    linking = [
        "Use `/link <GoCentral Gamertag>` to associate your Discord with your GoCentral leaderboard name.",
        "Platform tags like `[RPCS3]`, `[360]`, etc. are ignored — just type your gamertag exactly as shown.",
        "You can link **multiple names**. Use `/unlink` to remove linked gamertags.",
    ]

    desc = []
    desc.append(f"**What I can do**\n• " + "\n• ".join(what_i_do))
    desc.append(f"\n**Daily format**\n• {weekly_core}\n• {weekly_pro}\n")
    desc.append("\n**Song selection rules**\n• " + "\n• ".join(rules))
    desc.append("\n**Quick start**\n• " + "\n• ".join(quick_start))
    desc.append("\n**Link your account**\n• " + "\n• ".join(linking))
    desc.append("\n**Timing**\n• One new battle **every day**, each lasts **7 days**.")

    embed = discord.Embed(
        title="Score Snipe — Info",
        description="\n".join(desc),
        color=0x3B82F6
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="link", description="Link your Discord to a leaderboard name (platform tags are ignored).")
@app_commands.describe(name='Leaderboard name, e.g. "jnackmilo" (you don\'t need [RPCS3]/[Xenia])')
async def link_cmd(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True, thinking=True)

    if not bot.manager:
        await interaction.followup.send("Manager not ready.", ephemeral=True)
        return

    display = _strip_platform_tags(name)
    key = _norm_link_name(name)
    if not key:
        await interaction.followup.send("Provide a non-empty name.", ephemeral=True)
        return

    async with bot.manager.lock:
        st = bot.manager.state

        # Global uniqueness check
        owner = _link_owner_user_id(st, key)
        if owner is not None and owner != interaction.user.id:
            await interaction.followup.send(
                "That leaderboard name is already linked by another Discord account.",
                ephemeral=True
            )
            return

        current = _user_links(st, interaction.user.id)

        # If user already linked this (maybe different display), update the display
        idx_by_norm = {_norm_link_name(n): i for i, n in enumerate(current)}
        if key in idx_by_norm:
            i = idx_by_norm[key]
            if _strip_platform_tags(current[i]) != display:
                current[i] = display
                _save_user_links(st, interaction.user.id, current)
                _write_manager_state(st)
                final = _user_links(st, interaction.user.id)
                pretty = ", ".join(f"**{n}**" for n in final) if final else "—"
                await interaction.followup.send(f"Updated to **{display}**.\nYour linked names: {pretty}", ephemeral=True)
                return
            # no change
            final = _user_links(st, interaction.user.id)
            pretty = ", ".join(f"**{n}**" for n in final) if final else "—"
            await interaction.followup.send(f"You're already linked to **{display}**.\nYour linked names: {pretty}", ephemeral=True)
            return

        # Add new link
        current.append(display)
        _save_user_links(st, interaction.user.id, current)
        _write_manager_state(st)

    final = _user_links(bot.manager.state, interaction.user.id)
    pretty = ", ".join(f"**{n}**" for n in final) if final else "—"
    await interaction.followup.send(f"Linked to **{display}**.\nYour linked names: {pretty}", ephemeral=True)


@bot.tree.command(name="unlink", description="Remove a linked leaderboard name (platform tags are ignored).")
@app_commands.describe(name='Leaderboard name to remove, e.g. "jnackmilo" (you can include tags like [RPCS3])')
async def unlink_cmd(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True, thinking=True)

    if not bot.manager:
        await interaction.followup.send("Manager not ready.", ephemeral=True)
        return

    key = _norm_link_name(name)
    if not key:
        await interaction.followup.send("Provide a non-empty name.", ephemeral=True)
        return

    async with bot.manager.lock:
        st = bot.manager.state
        current = _user_links(st, interaction.user.id)
        kept = [n for n in current if _norm_link_name(n) != key]
        if len(kept) == len(current):
            await interaction.followup.send("No matching linked name found to remove.", ephemeral=True)
            return
        _save_user_links(st, interaction.user.id, kept)
        _write_manager_state(st)

    final = _user_links(bot.manager.state, interaction.user.id)
    pretty = ", ".join(f"**{n}**" for n in final) if final else "—"
    await interaction.followup.send(f"Unlinked.\nYour linked names: {pretty}", ephemeral=True)

@bot.tree.command(
    name="z_admin",
    description="Admin only: Manage Battles."
)
@app_commands.choices(action=[
    app_commands.Choice(name="Enable",              value="enable"),
    app_commands.Choice(name="Disable",             value="disable"),
    app_commands.Choice(name="Status",              value="status"),
    app_commands.Choice(name="Run now",             value="run"),
    app_commands.Choice(name="Create battle",       value="create"),
    app_commands.Choice(name="Delete battle",       value="delete"),
    app_commands.Choice(name="Test expired winners", value="test_expired"),
])
@app_commands.describe(
    action="What to do",

    # Run now
    count="(Run only) How many battles to create right now",

    # TEST winners
    ping="(Test only) Ping the configured role in the post",

    # Create battle
    song="(Create only) Song name (quotes for exact)",
    instrument="(Create only) Instrument (guitar, bass, drums, vocals, band, etc.)",
    title="(Create only) Battle title",
    description="(Create only) Battle description",

    # Delete battle
    battle_id="(Delete only) Numeric battle ID",
    window_hours="(Test-expired) Look back this many hours when auto-selecting (default 48)"
)
@is_admin()
async def z_admin_auto(
    interaction: discord.Interaction,
    action: app_commands.Choice[str],
    count: Optional[int] = None,
    ping: Optional[bool] = False,
    song: Optional[str] = None,
    instrument: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    battle_id: Optional[int] = None,
    window_hours: Optional[int] = 72,
):
    await interaction.response.defer(ephemeral=True, thinking=True)

    if not bot.manager:
        await interaction.followup.send("Manager not ready.", ephemeral=True)
        return

    act = action.value

    # ---------------- ENABLE ----------------
    if act == "enable":
        await bot.manager.enable()
        state = bot.manager.state
        nxt = _from_iso(state.get("next_run_at"))
        ts = int(nxt.timestamp()) if nxt else int(_utcnow().timestamp())
        await interaction.followup.send(
            f"✅ Auto battle manager **enabled**. Next run <t:{ts}:R>.",
            ephemeral=True
        )
        return

    # ---------------- DISABLE ---------------
    if act == "disable":
        await bot.manager.disable()
        await interaction.followup.send("⏸️ Auto battle manager **disabled**.", ephemeral=True)
        return

    # ---------------- STATUS ----------------
    if act == "status":
        st = bot.manager.state
        enabled = "ON" if st.get("enabled") else "OFF"
        nxt = _from_iso(st.get("next_run_at")) if st.get("enabled") else None
        last = _from_iso(st.get("last_run_at"))
        created = st.get("created_battles", [])
        parts = [
            f"Status: **{enabled}**",
            f"Next run: {f'<t:{int(nxt.timestamp())}:R>' if nxt else '—'}",
            f"Last run: {f'<t:{int(last.timestamp())}:R>' if last else '—'}",
            f"Songs per run: {st.get('songs_per_run', 6)}",
            f"Total created: {len(created)}",
        ]
        if st.get("last_error"):
            parts.append(f"Last error: `{st['last_error']}`")
        await interaction.followup.send("\n".join(parts), ephemeral=True)
        return

    # ---------------- RUN NOW ---------------
    if act == "run":
        dv = getattr(bot, "daily_vote", None)
        if not dv:
            await interaction.followup.send("DailyVoteManager not ready.", ephemeral=True)
            return

        # If a vote is already active, don't double-post
        active = dv._vote()
        if active and _utcnow() < _from_iso(active["ends_at"]):
            ends = _from_iso(active["ends_at"])
            ts = int(ends.timestamp()) if ends else None
            await interaction.followup.send(
                f"A vote is already running and ends <t:{ts}:R>.", ephemeral=True
            )
            return

        ok = await dv.start_vote_now()
        if not ok:
            await interaction.followup.send("Could not start a new vote (no candidates / channel missing).", ephemeral=True)
            return

        ends = dv.active_vote_ends_at()
        ts = int(ends.timestamp()) if ends else None
        await interaction.followup.send(
            f"✅ Started a new daily vote. Ends <t:{ts}:R>.", ephemeral=True
        )
        return

    ## ---------------- RUN NOW ---------------
    #if act == "run":
    #    try:
    #        created = await bot.manager.run_once(count=count)
    #    except Exception as e:
    #        await interaction.followup.send(f"❌ Run failed: {e}", ephemeral=True)
    #        return

    #    if not created:
    #        await interaction.followup.send("No battles created.", ephemeral=True)
    #        return

    #    lines = [f"✅ Created {len(created)} battle(s):"]
    #    for rec in created:
    #        lines.append(f"• ID `{rec['battle_id']}` — {rec['title']}")
    #    await interaction.followup.send("\n".join(lines), ephemeral=True)
    #    return

    # ------------- TEST EXPIRED WINNERS (iterate ALL) -------------
    if act == "test_expired":
        if not bot.http_session or not bot.manager:
            await interaction.followup.send("HTTP session / manager not ready.", ephemeral=True)
            return

        # helpers
        def _as_int(v):
            try:
                return int(str(v).strip())
            except Exception:
                return None

        def _extract_song_id(b: dict) -> Optional[int]:
            sid = None
            if isinstance(b.get("song_ids"), list) and b["song_ids"]:
                sid = b["song_ids"][0]
            elif isinstance(b.get("song_id"), (int, str)):
                sid = b["song_id"]
            elif isinstance(b.get("description"), str):
                m = re.search(r"\bID\s*(\d{4,9})\b", b["description"])
                if m:
                    sid = m.group(1)
            try:
                return int(sid) if sid is not None else None
            except (TypeError, ValueError):
                return None

        battles = await fetch_battles(bot.http_session)
        if not battles:
            await interaction.followup.send("No battles returned by API.", ephemeral=True)
            return

        now = _utcnow()
        lookback = now - timedelta(hours=int(window_hours or 72))

        # collect ALL expired (optionally within window)
        expired: list[dict] = []
        for b in battles:
            dt = _as_dt(b.get("expires_at"))       # accepts epoch ints/strings or ISO
            if not dt or dt > now:
                continue
            if window_hours is not None and dt < lookback:
                continue
            expired.append(b)

        if not expired:
            txt = "No expired battles found"
            if window_hours:
                txt += f" within last {int(window_hours)}h"
            await interaction.followup.send(txt + ".", ephemeral=True)
            return

        # oldest → newest to keep output tidy
        expired.sort(key=lambda x: (_as_dt(x.get("expires_at")) or now))

        # Respect the 'ping' flag without touching _announce_winners()
        original_ping_role_id = getattr(bot, "ping_role_id", None)
        if not ping:
            bot.ping_role_id = None

        posted_ids: list[str] = []

        try:
            for b in expired:
                bid = _as_int(b.get("battle_id"))
                if bid is None:
                    continue

                rec = {
                    "battle_id": bid,
                    "song_id": _extract_song_id(b) or b.get("song_id"),
                    "instrument": b.get("instrument"),
                    "created_at": b.get("starts_at") or b.get("created_at"),
                    "expires_at": b.get("expires_at"),
                    "title": b.get("title"),
                    "week": bot.manager.state.get("week") or 0,
                    "winner_announced": False,
                    "winner": None,
                    "overtakes": bot.manager.state.get(OVERTAKES_KEY, {}).get(str(bid), 0),
                }

                await bot.manager._announce_winners([rec])
                posted_ids.append(str(bid))

                # be gentle with rate limits
                await asyncio.sleep(0.7)
        finally:
            # restore ping role setting
            bot.ping_role_id = original_ping_role_id

        await interaction.followup.send(
            f"Posted winners for **{len(posted_ids)}** expired battle(s): "
            + (", ".join(posted_ids) if posted_ids else "—"),
            ephemeral=True
        )
        return


    # ---------------- CREATE ----------------
    if act == "create":
        # Validate inputs
        missing = [k for k, v in {
            "song": song, "instrument": instrument, "title": title, "description": description
        }.items() if not v]
        if missing:
            await interaction.followup.send(f"Missing required fields for create: {', '.join(missing)}", ephemeral=True)
            return

        if not bot.song_index:
            await interaction.followup.send("Song map not loaded.", ephemeral=True)
            return

        resolved_instr = resolve_instrument(instrument or "")
        if not resolved_instr or resolved_instr not in INSTR_ROLE_IDS:
            valid = ", ".join(sorted(INSTR_ROLE_IDS.keys()))
            await interaction.followup.send(f"Unknown instrument '{instrument}'. Try one of: {valid}.", ephemeral=True)
            return

        candidates = bot.song_index.find_all(song or "")
        if not candidates:
            await interaction.followup.send(f"No song match for: {song}", ephemeral=True)
            return
        elif len(candidates) > 1:
            # Let the user choose, then finish via continue_battle_create
            view = SongSelectView(
                candidates,
                resolved_instr,
                interaction,
                lambda i, s_obj, instr_ok: continue_battle_create(i, s_obj, instr_ok, title or "", description or "")
            )
            msg = await interaction.followup.send("Multiple matches found. Please choose:", view=view, ephemeral=True)
            view.message = msg
            return

        await continue_battle_create(interaction, candidates[0], resolved_instr, title or "", description or "")
        return

    # ---------------- DELETE ----------------
    if act == "delete":
        if battle_id is None:
            await interaction.followup.send("You must provide `battle_id`.", ephemeral=True)
            return

        api_key = get_api_key()
        if not api_key:
            await interaction.followup.send("API key not configured.", ephemeral=True)
            return

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"battle_id": int(battle_id)}

        try:
            async with bot.http_session.delete(DELETE_BATTLE_URL, headers=headers, json=payload, timeout=10) as resp:
                raw = await resp.text()
                if 200 <= resp.status < 300:
                    await interaction.followup.send(f"✅ Battle ID `{battle_id}` deleted.", ephemeral=True)
                else:
                    await interaction.followup.send(f"❌ Failed to delete battle: {resp.status} - {raw}", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f"Exception while deleting battle: {e}", ephemeral=True)
        return

    # Fallback
    await interaction.followup.send("Unknown action.", ephemeral=True)

def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    token = config.get("discord", "token", fallback=None)
    if not token:
        raise SystemExit("Missing Discord token in config.ini under [discord] section.")

    bot.run(token)

if __name__ == "__main__":
    main()
