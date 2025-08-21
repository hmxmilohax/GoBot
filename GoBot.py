import os
import re
import json
import asyncio
import unicodedata
import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import configparser
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands
from discord.errors import Forbidden

SONG_MAP_PATH = Path(__file__).resolve().parent / "song_map.dta"
LEADERBOARDS_URL = "https://gocentral-service.rbenhanced.rocks/leaderboards"
TOP_N = 10

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

DIFF_MAP = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Expert"}
DIFF_LETTER = {1: "E", 2: "M", 3: "H", 4: "X"}

@dataclass
class Song:
    slug: str
    name: str
    artist: str
    song_id: int
    year: Optional[int] = None
    album_name: Optional[str] = None
    genre: Optional[str] = None

class SongIndex:
    def __init__(self, songs: List[Song]):
        self.songs = songs
        self.by_norm_name = {self._norm(s.name): s for s in songs}

    @staticmethod
    def _strip_accents(s: str) -> str:
        nk = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nk if not unicodedata.combining(c))

    @classmethod
    def _norm(cls, s: str) -> str:
        s = cls._strip_accents(s).lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    def find_all(self, query: str, max_results: int = 5) -> List[Song]:
        exact_q = None
        m = re.search(r'"([^"]+)"', query)
        if m:
            exact_q = m.group(1)
        q = exact_q if exact_q else query
        nq = self._norm(q)

        matches = []

        # Exact match
        s = self.by_norm_name.get(nq)
        if s:
            matches.append((1.0, s))

        # Substring match
        for x in self.songs:
            name_norm = self._norm(x.name)
            if nq in name_norm:
                score = len(nq) / (1 + abs(len(name_norm) - len(nq)))
                matches.append((score, x))

        # Fuzzy fallback
        for candidate in self.songs:
            score = difflib.SequenceMatcher(None, nq, self._norm(candidate.name)).ratio()
            if score >= 0.5:
                matches.append((score, candidate))

        # Sort and deduplicate
        unique: Dict[str, Song] = {}
        for score, song in sorted(matches, key=lambda t: -t[0]):
            if song.slug not in unique:
                unique[song.slug] = song

        return list(unique.values())[:max_results]



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

    name = rx(r"\(name\s+\"([^\"]+)\"\)")
    artist = rx(r"\(artist\s+\"([^\"]+)\"\)") or ""
    album = rx(r"\(album_name\s+\"([^\"]+)\"\)")
    year = rx(r"\(year_released\s+(\d+)\)")
    song_id = rx(r"\(song_id\s+(\d+)\)")
    genre = rx(r"\(genre\s+([a-zA-Z0-9_]+)\)")

    if song_id and name:
        return Song(
            slug=slug,
            name=name,
            artist=artist,
            song_id=int(song_id),
            year=int(year) if year else None,
            album_name=album,
            genre=genre,
        )
    return None


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

    async def setup_hook(self) -> None:
        self.song_index = load_song_map(SONG_MAP_PATH)
        self.http_session = aiohttp.ClientSession()
        await self.tree.sync()

    async def close(self) -> None:
        if self.http_session:
            await self.http_session.close()
        await super().close()


bot = LBClient()


class LeaderboardView(discord.ui.View):
    def __init__(self, all_rows: List[dict], page_size: int, song_title: str, artist: str, instrument: str, year: Optional[int] = None, genre: Optional[str] = None):
        super().__init__(timeout=120)
        self.all_rows = all_rows
        self.page_size = page_size
        self.page = 0
        self.highlight_index: Optional[int] = None
        self.song_title = song_title
        self.artist = artist
        self.instrument = instrument
        self.year = year
        self.genre = genre
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
        if view.page > 0:
            view.page -= 1
        await view.update_message(interaction)

class NextButton(discord.ui.Button):
    def __init__(self, disabled: bool = False):
        super().__init__(label="Next", style=discord.ButtonStyle.secondary, disabled=disabled)
    async def callback(self, interaction: discord.Interaction):
        view: LeaderboardView = self.view  # type: ignore
        total_pages = max(1, (len(view.all_rows) + view.page_size - 1) // view.page_size)
        if view.page < total_pages - 1:
            view.page += 1
        await view.update_message(interaction)

class FindPlayerButton(discord.ui.Button):
    def __init__(self):
        super().__init__(label="Find Player", style=discord.ButtonStyle.primary)
    async def callback(self, interaction: discord.Interaction):
        view: LeaderboardView = self.view  # type: ignore
        await interaction.response.send_modal(FindPlayerModal(view))

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
            await interaction.response.send_message(f"No player matching '{query}'.")
            return
        self.view.highlight_index = idx
        self.view.page = idx // self.view.page_size
        await self.view.update_message(interaction)

class SongSelect(discord.ui.Select):
    def __init__(self, options: List[discord.SelectOption], matches: List[Song], instrument: str, interaction: discord.Interaction):
        super().__init__(placeholder="Choose the correct song...", options=options)
        self.matches = matches
        self.instrument = instrument
        self.original_interaction = interaction

    async def callback(self, interaction: discord.Interaction):
        selected_index = int(self.values[0])
        s = self.matches[selected_index]
        resolved = resolve_instrument(self.instrument)
        if not resolved or resolved not in INSTR_ROLE_IDS:
            await interaction.response.send_message(f"Invalid instrument '{self.instrument}'.")
            return
        role_id = INSTR_ROLE_IDS[resolved]

        assert bot.http_session is not None
        rows = await fetch_leaderboards(bot.http_session, s.song_id, role_id)
        if rows is None:
            await interaction.response.send_message("No leaderboard data returned.")
            return

        view = LeaderboardView(
            all_rows=rows,
            page_size=TOP_N,
            song_title=s.name,
            artist=s.artist,
            instrument=INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize()),
            year=s.year,
            genre=s.genre
        )

        start_index = view.page * view.page_size
        desc = format_leaderboard_rows(rows, start=start_index, max_n=view.page_size, highlight_index=view.highlight_index)

        embed = discord.Embed(
            title=f"Leaderboards — {s.name} — {INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize())}",
            description=desc
        )
        footer_parts = [f"Artist: {s.artist}"]
        if s.year:
            footer_parts.append(f"Year: {s.year}")
        if s.genre:
            footer_parts.append(f"Genre: {s.genre}")
        embed.set_footer(text=" • ".join(footer_parts))
        view.message = await interaction.response.send_message(embed=embed, view=view)

class SongSelectView(discord.ui.View):
    def __init__(self, matches: List[Song], instrument: str, interaction: discord.Interaction):
        super().__init__(timeout=60)
        options = [
            discord.SelectOption(label=f"{s.name}", description=s.artist[:100], value=str(i))
            for i, s in enumerate(matches)
        ]
        self.add_item(SongSelect(options, matches, instrument, interaction))


@bot.tree.command(name="leaderboards", description="Probe GoCentral leaderboards by song name.")
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
        view = SongSelectView(candidates, instrument, interaction)
        await interaction.followup.send(
            f"Multiple songs matched your query for '{song}'. Please select one:",
            view=view,
            ephemeral=True
        )
        return
    else:
        s = candidates[0]


    role_id = INSTR_ROLE_IDS[resolved]

    assert bot.http_session is not None
    rows = await fetch_leaderboards(bot.http_session, s.song_id, role_id)
    if rows is None:
        await interaction.followup.send("No leaderboard data returned.")
        return

    # build the view first (so page_size is known)
    view = LeaderboardView(
        all_rows=rows,
        page_size=TOP_N,
        song_title=s.name,
        artist=s.artist,
        instrument=INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize()),
        year=s.year,
        genre=s.genre
    )

    start_index = view.page * view.page_size
    desc = format_leaderboard_rows(rows, start=start_index, max_n=view.page_size, highlight_index=view.highlight_index)

    embed = discord.Embed(
        title=f"Leaderboards — {s.name} — {INSTR_DISPLAY_NAMES.get(resolved, resolved.capitalize())}",
        description=desc
    )
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
        await interaction.followup.send(f"Posted results in {channel.mention}")
    else:
        # Fallback: still post leaderboard publicly in DMs or last-resort context
        try:
            msg_obj = await interaction.user.send("Here's your leaderboard result:", embed=embed, view=view)
            view.message = msg_obj
            await interaction.followup.send("Couldn't post in channel, so I DMed you instead.")
        except discord.Forbidden:
            await interaction.followup.send("I couldn't post in the channel or DM you. Please check my permissions.")

def main():
    config = configparser.ConfigParser()
    config.read("config.ini")

    token = config.get("discord", "token", fallback=None)
    if not token:
        raise SystemExit("Missing Discord token in config.ini under [discord] section.")

    bot.run(token)

if __name__ == "__main__":
    main()
