"""
Unified Pipeline GUI: Media → SRT (AssemblyAI) → Script (Gemini) → Audio (ElevenLabs)

Features
- Transcribe video/audio to SRT using AssemblyAI API
- Adapt SRT to script (Text/Markdown/JSON) using Gemini (google-generativeai)
- Convert Script (.txt) or SRT to Audio using ElevenLabs with round-robin API keys
- Keep Vietnamese UI labels; settings persistence to settings.json

Requirements
  pip install requests google-generativeai pysrt pydub
  ffmpeg available in PATH for pydub
"""

from __future__ import annotations

import io
import json
import os
import re
import hashlib
import threading
import time
import tempfile
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional

# Optional external deps
try:
    import requests
except Exception:
    requests = None  # type: ignore
try:
    import pysrt
except Exception:
    pysrt = None  # type: ignore
try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None  # type: ignore


SETTINGS_FILE = Path(__file__).with_name("settings.json")
MODEL_ELEVEN_DEFAULT = "eleven_turbo_v2_5"
BG_COLOR = "#f7f9fb"


# ---------------- settings utils ----------------
def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_output_dir(src_path: str) -> str:
    """Return the `_out` directory for the given source file.

    If the parent folder already ends with `_out`, reuse it. Otherwise create a
    sanitized sibling folder based on ``<base>`` with a short hash suffix.
    """
    parent = os.path.dirname(src_path)
    if os.path.basename(parent).endswith("_out"):
        out_dir = parent
    else:
        base = os.path.splitext(os.path.basename(src_path))[0]
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
        safe = safe[:30] if safe else "file"
        hash_part = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
        out_dir = os.path.join(parent, f"{safe}_{hash_part}_out")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------- AssemblyAI (Media → SRT) ----------------
class AssemblyAIError(Exception):
    pass


def _require_requests():
    if requests is None:
        raise RuntimeError("Thiếu thư viện 'requests'. Cài bằng: pip install requests")


def aai_upload(api_key: str, media_path: str) -> str:
    _require_requests()
    url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": api_key}
    # Upload in chunks
    def _read_chunks(fp, chunk_size=5 * 1024 * 1024):
        while True:
            data = fp.read(chunk_size)
            if not data:
                break
            yield data
    with open(media_path, "rb") as f:
        resp = requests.post(url, headers=headers, data=_read_chunks(f))
    if resp.status_code != 200:
        raise AssemblyAIError(f"Upload thất bại: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    return data.get("upload_url") or ""


def aai_request_transcript(api_key: str, upload_url: str, language_code: str = "en") -> str:
    _require_requests()
    url = "https://api.assemblyai.com/v2/transcript"
    headers = {
        "authorization": api_key,
        "content-type": "application/json",
    }
    payload = {
        "audio_url": upload_url,
        "language_code": language_code or "en",
        "punctuate": True,
        "format_text": True,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise AssemblyAIError(f"Tạo transcript thất bại: {resp.status_code} {resp.text[:200]}")
    return resp.json().get("id", "")


def aai_poll_completed(api_key: str, transcript_id: str, poll_interval_s: float = 5.0) -> dict:
    _require_requests()
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {"authorization": api_key}
    while True:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            raise AssemblyAIError(f"Poll thất bại: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "error":
            raise AssemblyAIError(f"AssemblyAI báo lỗi: {data.get('error')}")
        time.sleep(poll_interval_s)


def aai_download_srt(api_key: str, transcript_id: str) -> str:
    _require_requests()
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/srt"
    headers = {"authorization": api_key}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise AssemblyAIError(f"Tải SRT thất bại: {resp.status_code} {resp.text[:200]}")
    return resp.text


# ---------------- length budget & prompts (Gemini) ----------------
CJK_DEFAULT = {"zh": 5.0, "ja": 5.0, "ko": 5.0}
WPS_DEFAULT = {
    "vi": 2.8, "en": 2.9, "es": 2.9, "fr": 2.6, "de": 2.5,
    "pt": 2.8, "id": 2.8, "hi": 2.7, "tr": 2.6, "ru": 2.6,
}
CJK_LANGS = set(CJK_DEFAULT.keys())


@dataclass
class LengthBudget:
    count_type: str
    budget: int
    min_count: int
    max_count: int
    rate: float


def srt_duration_seconds_from_text(srt_text: str) -> float:
    time_re = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")
    max_end_ms = 0
    for m in time_re.finditer(srt_text):
        h2, m2, s2, ms2 = m.group(5), m.group(6), m.group(7), m.group(8)
        end_ms = (int(h2) * 3600 + int(m2) * 60 + int(s2)) * 1000 + int(ms2)
        max_end_ms = max(max_end_ms, end_ms)
    return round(max_end_ms / 1000.0, 3)


def length_budget(duration_sec: float, lang_code: str, override_rate: Optional[float] = None) -> LengthBudget:
    tol = 0.13
    if lang_code in CJK_LANGS:
        cps = override_rate if (override_rate and override_rate > 0) else CJK_DEFAULT[lang_code]
        budget = round(duration_sec * cps)
        return LengthBudget("chars", budget, round(budget * (1 - tol)), round(budget * (1 + tol)), cps)
    wps = override_rate if (override_rate and override_rate > 0) else WPS_DEFAULT.get(lang_code, 2.8)
    budget = round(duration_sec * wps)
    return LengthBudget("words", budget, round(budget * (1 - tol)), round(budget * (1 + tol)), wps)


def build_prompt_text(srt_content: str, target_lang: str, locale: str,
                      duration_sec: float, lb: LengthBudget, lang_code: str) -> str:
    lang_type = "CJK" if lang_code in CJK_LANGS else "space-delimited"
    length_line = f"Target length ~{lb.budget} {lb.count_type} (acceptable {lb.min_count}-{lb.max_count}, ±13%)."
    return (
        f"Write a clean narration script in {target_lang}{f' ({locale})' if locale else ''} for TikTok,"
        f" based on the SRT content.\n"
        # 🔥 BẮT BUỘC HOOK NGAY ĐẦU
        f"- Start with a powerful HOOK (1–2 sentences) in the first 3 seconds that surprises, asks a sharp question, "
        f"or teases a counter-intuitive fact to grab attention. Keep it short and punchy.\n"
        # Cấu trúc & phong cách
        f"- Then continue the main narration based on the SRT; keep transitions smooth and engaging.\n"
        f"- If appropriate, end with a light, natural CTA (e.g., invite to follow) but avoid sounding like an ad.\n"
        # Độ dài/nhịp
        f"- Duration (from source): {int(round(duration_sec))} seconds. Language type: {lang_type}. {length_line}\n"
        # Phong cách biên soạn
        f"- Conversational, natural, no disclaimers, no headings, no bullets, no markdown.\n"
        f"- Keep names and facts consistent. If length is off, rewrite to fit.\n\n"
        f"SRT:\n{srt_content}\n\n"
        f"Return only the final narration text."
    )


def call_gemini(api_key: str, model_name: str, prompt: str, want_json: bool) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("Chưa cài 'google-generativeai'. Cài: pip install google-generativeai") from e
    genai.configure(api_key=api_key)
    generation_config = {}
    if want_json:
        generation_config["response_mime_type"] = "application/json"
    model = genai.GenerativeModel(model_name=model_name)
    resp = model.generate_content(prompt, generation_config=generation_config or None)
    text = getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        parts = []
        for c in resp.candidates:
            if getattr(c, "content", None) and getattr(c.content, "parts", None):
                for p in c.content.parts:
                    parts.append(str(getattr(p, "text", "")))
        text = "\n".join([t for t in parts if t.strip()])
    return text or str(resp)


# ---------------- ElevenLabs (TTS) ----------------
class ApiError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"API Error {status_code}: {message}")
        self.status_code = status_code
        self.message = message


class KeyExhaustedError(Exception):
    def __init__(self, file_path: str, seg_idx: int, seg_total: int):
        name = os.path.basename(file_path)
        msg = (
            f"Hết tất cả API key. Dừng tại file '{name}' đoạn {seg_idx}/{seg_total}.\n"
            "Vui lòng bổ sung/thay API key rồi chạy lại."
        )
        super().__init__(msg)
        self.file_path = file_path
        self.seg_idx = seg_idx
        self.seg_total = seg_total


class KeyRotator:
    def __init__(self, keys: List[str]):
        dedup: List[str] = []
        seen = set()
        for k in keys:
            if k and k not in seen:
                dedup.append(k)
                seen.add(k)
        self.keys: List[str] = dedup
        self.idx = 0

    def next(self) -> str:
        if not self.keys:
            raise RuntimeError("Không còn API key")
        k = self.keys[self.idx]
        self.idx = (self.idx + 1) % len(self.keys)
        return k

    def remove(self, key: str) -> None:
        if key in self.keys:
            i = self.keys.index(key)
            del self.keys[i]
            if i <= self.idx and self.idx > 0:
                self.idx -= 1

    def count(self) -> int:
        return len(self.keys)


def describe_api_error(e: ApiError) -> str:
    code = e.status_code
    if code == 401:
        return "API key không hợp lệ hoặc đã bị thu hồi (401)."
    if code == 403:
        return "API key bị từ chối quyền truy cập (403)."
    if code == 404:
        return "Voice ID không tồn tại (404)."
    if code == 422:
        return "Payload không hợp lệ (422). Vui lòng kiểm tra nội dung/giới hạn độ dài."
    if code == 429:
        return "API key đã đạt giới hạn (429). Đang thử key khác."
    if 500 <= code < 600:
        return f"Máy chủ gặp sự cố ({code}). Đang thử key khác."
    return f"Lỗi API ({code})."


def text_to_speech(api_key: str, voice_id: str, text: str, model_id: str) -> AudioSegment:
    if requests is None or AudioSegment is None:
        raise RuntimeError("Thiếu 'requests' hoặc 'pydub'. Cài: pip install requests pydub")
    voice_settings = {"stability": 0.3, "similarity_boost": 0.85}
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    payload = {"text": text, "model_id": model_id, "voice_settings": voice_settings}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise ApiError(resp.status_code, resp.text)
    return AudioSegment.from_file(io.BytesIO(resp.content), format="mp3")


# -------- ElevenLabs voice management (cleanup on quota) --------
def _is_voice_limit_error_message(msg: str) -> bool:
    try:
        data = json.loads(msg)
        # Newer API returns detail.status
        detail = data.get("detail") if isinstance(data, dict) else None
        if isinstance(detail, dict) and str(detail.get("status")) == "voice_limit_reached":
            return True
    except Exception:
        pass
    # Fallback: substring check
    return "voice_limit_reached" in (msg or "")


def eleven_list_voices(api_key: str) -> list:
    if requests is None:
        raise RuntimeError("Thiếu 'requests'.")
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    resp = requests.get(url, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise ApiError(resp.status_code, resp.text)
    data = resp.json() if resp.headers.get("content-type","" ).startswith("application/json") else {}
    return data.get("voices", []) if isinstance(data, dict) else []


def eleven_delete_voice(api_key: str, voice_id: str) -> bool:
    if requests is None:
        raise RuntimeError("Thiếu 'requests'.")
    url = f"https://api.elevenlabs.io/v1/voices/{voice_id}"
    headers = {"xi-api-key": api_key}
    resp = requests.delete(url, headers=headers, timeout=60)
    return resp.status_code in (200, 204)


def eleven_free_one_custom_voice(api_key: str, log_func: Optional[Callable[[str], None]] = None) -> bool:
    try:
        voices = eleven_list_voices(api_key)
    except Exception as e:
        if log_func:
            log_func(f"[TTS] Không lấy được danh sách voice để dọn quota: {e}")
        return False
    # Heuristic: delete the first non-premade (cloned/generated/custom) voice
    candidate_id = None
    for v in voices:
        category = str(v.get("category", "")).lower()
        vid = v.get("voice_id") or v.get("voiceID")
        if vid and category and category not in ("premade", "stock", "built-in", "professional"):
            candidate_id = vid
            break
    if not candidate_id and voices:
        # As a last resort, try deleting the last voice if metadata missing (avoid deleting premade if possible)
        for v in reversed(voices):
            vid = v.get("voice_id") or v.get("voiceID")
            category = str(v.get("category", "")).lower()
            if vid and category != "premade":
                candidate_id = vid
                break
    if not candidate_id:
        if log_func:
            log_func("[TTS] Không tìm thấy custom voice nào để xóa.")
        return False
    ok = eleven_delete_voice(api_key, candidate_id)
    if log_func:
        log_func(f"[TTS] Dọn bớt custom voice {'thành công' if ok else 'thất bại'} (voice_id={candidate_id}).")
    return ok


def speed_up_audio(audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
    if AudioSegment is None:
        return audio[:target_duration_ms]
    current_duration_ms = len(audio)
    if current_duration_ms <= target_duration_ms:
        return audio
    speed_factor = current_duration_ms / target_duration_ms
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        audio.export(tmp_in.name, format="wav")
        tmp_in_path = tmp_in.name
    tmp_out_path = tmp_in_path.replace(".wav", "_fast.wav")
    # Build chain atempo
    cmds = []
    remain = speed_factor
    while remain > 2.0:
        cmds.append("atempo=2.0")
        remain /= 2.0
    cmds.append(f"atempo={remain:.6f}")
    filter_arg = ",".join(cmds)
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-filter:a", filter_arg, tmp_out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        sped_up = AudioSegment.from_file(tmp_out_path, format="wav")
    except Exception:
        sped_up = AudioSegment.from_file(tmp_in_path, format="wav")[:target_duration_ms]
    finally:
        for p in (tmp_in_path, tmp_out_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
    return sped_up


def fit_to_duration(audio: AudioSegment, target_duration_ms: int) -> AudioSegment:
    if AudioSegment is None:
        return audio
    if len(audio) == target_duration_ms:
        return audio
    if len(audio) < target_duration_ms:
        return audio + AudioSegment.silent(duration=target_duration_ms - len(audio))
    sped = speed_up_audio(audio, target_duration_ms)
    if len(sped) > target_duration_ms:
        return sped[:target_duration_ms]
    if len(sped) < target_duration_ms:
        return sped + AudioSegment.silent(duration=target_duration_ms - len(sped))
    return sped


# ---------------- TXT chunking ----------------
MAX_CHARS_PER_REQUEST = 900
PAUSE_BETWEEN_CHUNKS_MS = 220


def split_text_into_chunks(text: str, max_chars: int = MAX_CHARS_PER_REQUEST) -> List[str]:
    text = re.sub(r"\r\n?", "\n", text).lstrip("\ufeff")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        sentences = re.split(r"(?<=[\.!?…])\s+", p)
        buf = ""
        for s in sentences:
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf += " " + s
            else:
                chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)
    if not chunks and text.strip():
        t = text.strip()
        for i in range(0, len(t), max_chars):
            chunks.append(t[i: i + max_chars])
    return chunks


# ---------------- GUI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Pipeline: Media → SRT → Script → Audio")
        self.configure(bg=BG_COLOR)
        # Mở rộng kích thước cửa sổ mặc định để nhìn thấy toàn bộ
        # các thành phần và nút bấm ngay khi khởi động.
        self.geometry("1200x860")
        self.minsize(1200, 820)

        # UI styles: bold titles and bold button texts
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", font=("Segoe UI", 9, "bold"))

        self.settings = load_settings()

        # AssemblyAI
        self.aai_api_file = tk.StringVar(value=self.settings.get("aai_api_file", ""))
        self.media_file = tk.StringVar(value=self.settings.get("media_file", ""))
        self.aai_lang = tk.StringVar(value=self.settings.get("aai_lang", "en"))

        # Gemini
        self.gem_api_file = tk.StringVar(value=self.settings.get("gem_api_file", ""))
        self.model_name = tk.StringVar(value=self.settings.get("gem_model", "gemini-2.0-flash"))
        self.output_format = tk.StringVar(value=self.settings.get("output_format", "Text"))
        self.target_lang = tk.StringVar(value=self.settings.get("target_lang", "Vietnamese"))
        self.lang_code = tk.StringVar(value=self.settings.get("lang_code", "vi"))
        self.locale = tk.StringVar(value=self.settings.get("locale", ""))
        self.override_rate = tk.StringVar(value=self.settings.get("override_rate", ""))

        # Files
        self.srt_file = tk.StringVar()
        self.script_file = tk.StringVar()
        # Batch directories (non-recursive)
        self.media_dir = tk.StringVar(value=self.settings.get("media_dir", ""))
        self.srt_dir = tk.StringVar()
        self.txt_dir = tk.StringVar()

        # ElevenLabs
        self.eleven_api_txt = tk.StringVar(value=self.settings.get("eleven_api_txt", ""))
        self.voice_id = tk.StringVar(value=self.settings.get("voice_id", ""))
        self.eleven_model = tk.StringVar(value=self.settings.get("eleven_model", MODEL_ELEVEN_DEFAULT))
        self.audio_format = tk.StringVar(value=self.settings.get("audio_format", "mp3"))
        self.rr_index = int(self.settings.get("eleven_rr_index", 0) or 0)

        # Batch directories (non-recursive) + unified pick fields
        self.media_dir = tk.StringVar(value=self.settings.get("media_dir", ""))
        self.srt_dir = tk.StringVar()
        self.txt_dir = tk.StringVar()
        self.media_pick = tk.StringVar(value=self.settings.get("media_pick", self.media_file.get() or self.media_dir.get()))
        self.srt_pick = tk.StringVar()
        self.txt_pick = tk.StringVar()

        # Language choices (Vietnamese labels)
        self._lang_choices = [
            ("Tiếng Anh", "English", "en"),
            ("Tiếng Việt", "Vietnamese", "vi"),
            ("Tiếng Tây Ban Nha", "Spanish", "es"),
            ("Tiếng Bồ đào nha", "Portuguese", "pt"),
            ("Tiếng Đức", "German", "de"),
            ("Tiếng Pháp", "French", "fr"),
            ("Tiếng Hàn", "Korean", "ko"),
            ("Tiếng Nhật", "Japanese", "ja"),
        ]
        self._lang_map_vi_to_en = {vi: en for (vi, en, code) in self._lang_choices}
        self._lang_map_vi_to_code = {vi: code for (vi, en, code) in self._lang_choices}
        self._locale_map = {
            "en": ["en-US", "en-UK"],
            "vi": ["vi-VN"],
            "es": ["es-ES", "es-MX"],
            "pt": ["pt-BR", "pt-PT"],
            "de": ["de-DE", "de-AT"],
            "fr": ["fr-FR", "fr-CA"],
            "ko": ["ko-KR"],
            "ja": ["ja-JP"],
        }
        # ensure locale is valid for current language
        current_code = self.lang_code.get()
        current_locales = self._locale_map.get(current_code, [])
        if self.locale.get() not in current_locales:
            self.locale.set(current_locales[0] if current_locales else "")
        self.lang_display = tk.StringVar(value=self.settings.get("lang_display", "Tiếng Việt"))

        # UI
        self._build_ui()

    # ---- UI ----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}
        # Root grid config
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Split layout: left config (1/3), right pipeline (2/3)
        container = ttk.PanedWindow(self, orient="horizontal")
        container.grid(row=0, column=0, sticky="nsew")

        # Left column: config
        left = ttk.Frame(container, padding="12 10 12 10")
        container.add(left, weight=1)
        sec_api = ttk.LabelFrame(left, text="Khu vực API Keys (cấu hình chung)")
        sec_api.pack(fill="x", **pad)
        r = ttk.Frame(sec_api)
        r.pack(fill="x", **pad)
        # Use grid so the Choose buttons are always visible
        ttk.Label(r, text="File API AAI (.txt):", width=22).grid(row=0, column=0, sticky="w", padx=(0,6))
        ttk.Entry(r, textvariable=self.aai_api_file).grid(row=0, column=1, sticky="we")
        ttk.Button(r, text="Chọn...", command=self.browse_aai_api).grid(row=0, column=2, padx=(6,0))
        ttk.Label(r, text="File API Gemini (.txt):", width=22).grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,0))
        ttk.Entry(r, textvariable=self.gem_api_file).grid(row=1, column=1, sticky="we", pady=(6,0))
        ttk.Button(r, text="Chọn...", command=self.browse_gem_api).grid(row=1, column=2, padx=(6,0), pady=(6,0))
        ttk.Label(r, text="File API ElevenLabs (.txt):", width=22).grid(row=2, column=0, sticky="w", padx=(0,6), pady=(6,0))
        ttk.Entry(r, textvariable=self.eleven_api_txt).grid(row=2, column=1, sticky="we", pady=(6,0))
        ttk.Button(r, text="Chọn...", command=self.browse_eleven_api).grid(row=2, column=2, padx=(6,0), pady=(6,0))
        r.grid_columnconfigure(1, weight=1)

        # Tuỳ chọn chung
        sec_common = ttk.LabelFrame(left, text="Tuỳ chọn chung")
        sec_common.pack(fill="x", **pad)
        r = ttk.Frame(sec_common); r.pack(fill="x", **pad)
        ttk.Label(r, text="Thư mục mặc định Input:", width=22).pack(side="left")
        self.default_input_dir = tk.StringVar(value=self.settings.get("default_input_dir", ""))
        ttk.Entry(r, textvariable=self.default_input_dir).pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(r, text="Chọn...", command=lambda: self._pick_dir_into(self.default_input_dir, "default_input_dir")).pack(side="left")
        r = ttk.Frame(sec_common); r.pack(fill="x", **pad)
        ttk.Label(r, text="Thư mục mặc định Output:", width=22).pack(side="left")
        self.default_output_dir = tk.StringVar(value=self.settings.get("default_output_dir", ""))
        ttk.Entry(r, textvariable=self.default_output_dir).pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(r, text="Chọn...", command=lambda: self._pick_dir_into(self.default_output_dir, "default_output_dir")).pack(side="left")

        # Right column: pipeline
        right = ttk.Frame(container, padding="12 10 12 10")
        container.add(right, weight=2)
        sec_aai = ttk.LabelFrame(right, text="Khối 1: Media → SRT")
        sec_aai.pack(fill="x", **pad)
        r = ttk.Frame(sec_aai); r.pack(fill="x", **pad)
        ttk.Label(r, text="Chọn file (video/audio) hoặc thư mục:", width=32).pack(side="left")
        ttk.Entry(r, textvariable=self.media_file).pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(r, text="Chọn file", command=self.browse_media).pack(side="left")
        ttk.Button(r, text="Chọn thư mục", command=self.browse_media_dir).pack(side="left", padx=(6,0))
        r = ttk.Frame(sec_aai); r.pack(fill="x", **pad)
        ttk.Label(r, text="Ngôn ngữ:").pack(side="left")
        ttk.Combobox(r, textvariable=self.aai_lang, values=["vi","en","es","fr","de","pt","ja","ko","zh"], width=12, state="readonly").pack(side="left", padx=(6,0))
        ttk.Button(r, text="Media → SRT", command=self.on_media_unified, width=18).pack(side="right")

        # Gemini section
        sec_g = ttk.LabelFrame(right, text="Khối 2: SRT → Kịch bản (Gemini)")
        sec_g.pack(fill="x", **pad)
        r = ttk.Frame(sec_g); r.pack(fill="x", **pad)
        ttk.Label(r, text="SRT Input: Chọn file / thư mục:", width=32).pack(side="left")
        ttk.Entry(r, textvariable=self.srt_file).pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(r, text="Chọn file", command=self.browse_srt).pack(side="left")
        ttk.Button(r, text="Chọn thư mục", command=self.browse_srt_dir).pack(side="left", padx=(6,0))
        r = ttk.Frame(sec_g); r.pack(fill="x", **pad)
        ttk.Label(r, text="Model:").pack(side="left")
        ttk.Entry(r, textvariable=self.model_name, width=28).pack(side="left")
        ttk.Label(r, text="Định dạng:").pack(side="left", padx=(12,0))
        ttk.Combobox(r, textvariable=self.output_format, values=["Text","Markdown","JSON"], width=12, state="readonly").pack(side="left")
        ttk.Label(r, text="Ngôn ngữ:").pack(side="left", padx=(12,0))
        cmb_lang = ttk.Combobox(
            r,
            textvariable=self.lang_display,
            values=[vi for (vi, en, code) in self._lang_choices],
            width=22,
            state="readonly",
        )
        cmb_lang.pack(side="left")

        ttk.Label(r, text="Locale:").pack(side="left", padx=(12,0))
        cmb_locale = ttk.Combobox(
            r,
            textvariable=self.locale,
            values=[""] + self._locale_map.get(self.lang_code.get(), []),
            width=12,
            state="readonly",
        )
        cmb_locale.pack(side="left")

        def _on_lang_selected(_=None):
            vi_name = self.lang_display.get()
            en_name = self._lang_map_vi_to_en.get(vi_name, "Vietnamese")
            code = self._lang_map_vi_to_code.get(vi_name, "vi")
            self.target_lang.set(en_name)
            self.lang_code.set(code)
            locales = self._locale_map.get(code, [])
            cmb_locale["values"] = [""] + locales
            if self.locale.get() not in locales:
                self.locale.set(locales[0] if locales else "")
            self.settings["lang_display"] = vi_name
            self.settings["target_lang"] = en_name
            self.settings["lang_code"] = code
            self.settings["locale"] = self.locale.get()
            save_settings(self.settings)

        def _on_locale_selected(_=None):
            self.settings["locale"] = self.locale.get()
            save_settings(self.settings)

        cmb_lang.bind("<<ComboboxSelected>>", _on_lang_selected)
        cmb_locale.bind("<<ComboboxSelected>>", _on_locale_selected)
        # sync current on load
        _on_lang_selected()
        r = ttk.Frame(sec_g); r.pack(fill="x", **pad)
        ttk.Button(r, text="SRT → Kịch bản", command=self.on_srt_unified, width=18).pack(side="left")
        ttk.Label(r, text="Tốc độ override (tùy chọn):").pack(side="left", padx=(18,0))
        ttk.Entry(r, textvariable=self.override_rate, width=10).pack(side="left")

        # ElevenLabs section
        sec_e = ttk.LabelFrame(right, text="Khối 3: Kịch bản/SRT → Audio (ElevenLabs)")
        sec_e.pack(fill="x", **pad)
        r = ttk.Frame(sec_e); r.pack(fill="x", **pad)
        ttk.Label(r, text="Voice ID:", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self.voice_id, width=28).pack(side="left")
        ttk.Label(r, text="Model:").pack(side="left", padx=(12,0))
        ttk.Combobox(r, textvariable=self.eleven_model, values=[MODEL_ELEVEN_DEFAULT, "eleven_multilingual_v2"], width=24, state="readonly").pack(side="left")
        # Ẩn lựa chọn định dạng âm thanh; mặc định mp3
        r = ttk.Frame(sec_e); r.pack(fill="x", **pad)
        ttk.Label(r, text="TXT/SRT Input: Chọn file / thư mục:", width=32).pack(side="left")
        ttk.Entry(r, textvariable=self.script_file).pack(side="left", fill="x", expand=True, padx=(0,6))
        ttk.Button(r, text="Chọn file", command=self.browse_script).pack(side="left")
        ttk.Button(r, text="Chọn thư mục", command=self.browse_txt_dir).pack(side="left", padx=(6,0))
        r = ttk.Frame(sec_e); r.pack(fill="x", **pad)
        ttk.Button(r, text="Kịch bản → Audio", command=self.on_txt_unified, width=18).pack(side="left")
        # Bottom global controls (always visible)
        ctrl = ttk.Frame(self, padding="6 6 6 6")
        ctrl.grid(row=1, column=0, sticky="ew")
        ttk.Button(ctrl, text="Chạy toàn bộ ▶", command=self.on_pipeline_full).grid(row=0, column=0)
        ttk.Button(ctrl, text="Tạm dừng ⏸", command=self.on_pause).grid(row=0, column=1, padx=(8,0))
        ttk.Button(ctrl, text="Dừng ■", command=self.on_stop).grid(row=0, column=2, padx=(8,0))
        self.queue_var = tk.StringVar(value="Hàng đợi: 0")
        ttk.Label(ctrl, textvariable=self.queue_var).grid(row=0, column=3, padx=(12,0))
        ttk.Button(ctrl, text="Mở thư mục đầu ra", command=self.open_output_folder).grid(row=0, column=5, padx=(8,8))
        self.progress = ttk.Progressbar(ctrl, orient="horizontal", mode="determinate")
        self.progress.grid(row=0, column=4, sticky="ew")
        ctrl.grid_columnconfigure(4, weight=1)
        ctrl.grid_rowconfigure(0, weight=1)

        # Log area (always visible)
        sec_log = ttk.LabelFrame(self, text="Nhật ký")
        sec_log.grid(row=2, column=0, sticky="nsew", **pad)
        sec_log.grid_rowconfigure(0, weight=1)
        sec_log.grid_columnconfigure(0, weight=1)
        self.log = tk.Text(sec_log, height=12, wrap="word")
        self.log.grid(row=0, column=0, sticky="nsew")

        self._log("Sẵn sàng. Chọn nguồn/thiết lập và bấm 'Chạy toàn bộ'.")

    # ---- Browse helpers ----
    def browse_aai_api(self):
        p = filedialog.askopenfilename(title="Chọn file AAI API", filetypes=[("Text","*.txt"),("All","*.*")])
        if p:
            self.aai_api_file.set(p)
            self.settings["aai_api_file"] = p
            save_settings(self.settings)

    def browse_media(self):
        p = filedialog.askopenfilename(title="Chọn file media", filetypes=[("Media","*.mp3 *.wav *.m4a *.mp4 *.mov *.mkv"),("All","*.*")])
        if p:
            self.media_file.set(p)
            self.settings["media_file"] = p
            save_settings(self.settings)

    def browse_media_dir(self):
        p = filedialog.askdirectory(title="Chọn thư mục Media")
        if p:
            self.media_dir.set(p)
            self.settings["media_dir"] = p
            save_settings(self.settings)

    def browse_gem_api(self):
        p = filedialog.askopenfilename(title="Chọn file Gemini API", filetypes=[("Text","*.txt"),("All","*.*")])
        if p:
            self.gem_api_file.set(p)
            self.settings["gem_api_file"] = p
            save_settings(self.settings)

    def browse_srt(self):
        p = filedialog.askopenfilename(title="Chọn file .srt", filetypes=[("SRT","*.srt"),("All","*.*")])
        if p:
            self.srt_file.set(p)

    def browse_srt_dir(self):
        p = filedialog.askdirectory(title="Chọn thư mục SRT")
        if p:
            self.srt_dir.set(p)

    def browse_script(self):
        p = filedialog.askopenfilename(title="Chọn file kịch bản .txt", filetypes=[("Text","*.txt"),("All","*.*")])
        if p:
            self.script_file.set(p)

    def browse_txt_dir(self):
        p = filedialog.askdirectory(title="Chọn thư mục TXT")
        if p:
            self.txt_dir.set(p)

    def browse_eleven_api(self):
        p = filedialog.askopenfilename(title="Chọn file ElevenLabs API (mỗi dòng 1 key)", filetypes=[("Text","*.txt"),("All","*.*")])
        if p:
            self.eleven_api_txt.set(p)
            self.settings["eleven_api_txt"] = p
            save_settings(self.settings)

    # ---- Log helper ----
    def _log(self, text: str):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    # ---- Utility ----
    def _read_first_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _ensure_requests(self):
        if requests is None:
            raise RuntimeError("Thiếu thư viện 'requests'. Cài: pip install requests")

    # helper to pick directory into a StringVar and save
    def _pick_dir_into(self, var: tk.StringVar, key: str):
        p = filedialog.askdirectory()
        if p:
            var.set(p)
            self.settings[key] = p
            save_settings(self.settings)

    # bottom control placeholders
    def on_pause(self):
        self._log("[Pipeline] Tạm dừng (chưa triển khai dừng job đang chạy)")

    def on_stop(self):
        self._log("[Pipeline] Dừng (chưa triển khai huỷ job đang chạy)")

    # ---- Actions ----
    def on_media_unified(self):
        # If media_file is a file -> single; else if media_dir set -> batch
        path = self.media_file.get().strip()
        if path and os.path.isfile(path):
            return self.on_media_to_srt()
        if self.media_dir.get().strip() and os.path.isdir(self.media_dir.get().strip()):
            return self.on_media_dir_pipeline()
        messagebox.showerror("Thiếu nguồn", "Hãy chọn file media hoặc thư mục media.")

    def on_srt_unified(self):
        path = self.srt_file.get().strip()
        if path and os.path.isfile(path):
            return self.on_srt_to_script()
        if self.srt_dir.get().strip() and os.path.isdir(self.srt_dir.get().strip()):
            return self.on_srt_dir_pipeline()
        messagebox.showerror("Thiếu SRT", "Hãy chọn file SRT hoặc thư mục SRT.")

    def on_txt_unified(self):
        path = self.script_file.get().strip()
        if path and os.path.isfile(path):
            return self.on_script_to_audio()
        if self.txt_dir.get().strip() and os.path.isdir(self.txt_dir.get().strip()):
            return self.on_txt_dir_pipeline()
        messagebox.showerror("Thiếu kịch bản", "Hãy chọn file TXT hoặc thư mục TXT.")
    def on_media_to_srt(self):
        api_path = self.aai_api_file.get().strip()
        media_path = self.media_file.get().strip()
        lang = self.aai_lang.get().strip() or "en"
        if not api_path or not os.path.isfile(api_path):
            messagebox.showerror("Thiếu API AAI", "Hãy chọn file API AssemblyAI (.txt)")
            return
        if not media_path or not os.path.isfile(media_path):
            messagebox.showerror("Thiếu media", "Hãy chọn file video/audio hợp lệ")
            return
        try:
            api_key = self._read_first_text_file(api_path)
        except Exception as e:
            messagebox.showerror("Lỗi API AAI", str(e)); return
        if not api_key:
            messagebox.showerror("API AAI trống", "File API không có nội dung"); return

        self._log("[AAI] Uploading media...")
        def task():
            try:
                upload_url = aai_upload(api_key, media_path)
                self._log("[AAI] Creating transcript...")
                tid = aai_request_transcript(api_key, upload_url, language_code=lang)
                self._log("[AAI] Polling transcript status...")
                _ = aai_poll_completed(api_key, tid)
                self._log("[AAI] Downloading SRT...")
                srt_text = aai_download_srt(api_key, tid)
                base = os.path.splitext(os.path.basename(media_path))[0]
                out_dir = get_output_dir(media_path)
                out_path = os.path.join(out_dir, f"{base}.srt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(srt_text)
                self.srt_file.set(out_path)
                self.settings["last_output_dir"] = out_dir
                save_settings(self.settings)
                self._log(f"[AAI] Done. {out_path}")
                messagebox.showinfo("Xong", f"Đã tạo phụ đề:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Lỗi AAI", str(e))
                self._log(f"[AAI] Lỗi: {e}")

        threading.Thread(target=task, daemon=True).start()

    # ---- Batch pipelines (non-recursive) ----
    def _confirm_overwrite(self, target_path: str) -> bool | None:
        if not os.path.exists(target_path):
            return True
        # Return True = overwrite, False = skip, None = cancel all
        return messagebox.askyesno("Tồn tại file", f"Đã tồn tại:\n{target_path}\n\nGhi đè? (Yes = Ghi đè, No = Bỏ qua)")

    def on_media_dir_pipeline(self):
        api_path = self.aai_api_file.get().strip()
        media_dir = self.media_dir.get().strip()
        lang = self.aai_lang.get().strip() or "en"
        if not api_path or not os.path.isfile(api_path):
            messagebox.showerror("Thiếu API AAI", "Hãy chọn file API AssemblyAI (.txt)"); return
        if not media_dir or not os.path.isdir(media_dir):
            messagebox.showerror("Thiếu thư mục", "Hãy chọn thư mục Media hợp lệ"); return
        try:
            api_key = self._read_first_text_file(api_path)
        except Exception as e:
            messagebox.showerror("Lỗi API AAI", str(e)); return
        if not api_key:
            messagebox.showerror("API AAI trống", "File API không có nội dung"); return

        media_exts = {".mp3",".wav",".m4a",".mp4",".mov",".mkv"}
        files = [os.path.join(media_dir, f) for f in os.listdir(media_dir) if os.path.splitext(f)[1].lower() in media_exts]
        if not files:
            messagebox.showinfo("Trống", "Không có file media phù hợp trong thư mục.")
            return
        self._log(f"[Batch AAI] Bắt đầu {len(files)} file...")

        def task():
            done = 0
            for media_path in files:
                try:
                    base = os.path.splitext(os.path.basename(media_path))[0]
                    out_dir = get_output_dir(media_path)
                    out_path = os.path.join(out_dir, f"{base}.srt")
                    decision = self._confirm_overwrite(out_path)
                    if decision is False:
                        self._log(f"[Batch AAI] Bỏ qua: {out_path}")
                        done += 1; continue
                    self._log(f"[Batch AAI] Upload: {os.path.basename(media_path)}")
                    upload_url = aai_upload(api_key, media_path)
                    tid = aai_request_transcript(api_key, upload_url, language_code=lang)
                    _ = aai_poll_completed(api_key, tid)
                    srt_text = aai_download_srt(api_key, tid)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(srt_text)
                    self.settings["last_output_dir"] = out_dir
                    save_settings(self.settings)
                    self._log(f"[Batch AAI] Done → {out_path}")
                except Exception as e:
                    self._log(f"[Batch AAI] Lỗi {os.path.basename(media_path)}: {e}")
                finally:
                    done += 1
            messagebox.showinfo("Xong", f"Đã xử lý Media → SRT: {done}/{len(files)} file")

        threading.Thread(target=task, daemon=True).start()

    def on_srt_dir_pipeline(self):
        gem_api_path = self.gem_api_file.get().strip()
        srt_dir = self.srt_dir.get().strip()
        if not gem_api_path or not os.path.isfile(gem_api_path):
            messagebox.showerror("Thiếu API Gemini", "Hãy chọn file API Gemini (.txt)"); return
        if not srt_dir or not os.path.isdir(srt_dir):
            messagebox.showerror("Thiếu thư mục", "Hãy chọn thư mục SRT hợp lệ"); return
        try:
            api_key = self._read_first_text_file(gem_api_path)
        except Exception as e:
            messagebox.showerror("Lỗi API Gemini", str(e)); return
        if not api_key:
            messagebox.showerror("API Gemini trống", "File API không có nội dung"); return
        files = [os.path.join(srt_dir, f) for f in os.listdir(srt_dir) if os.path.splitext(f)[1].lower() == ".srt"]
        if not files:
            messagebox.showinfo("Trống", "Không có file .srt trong thư mục."); return
        self._log(f"[Batch Gemini] Bắt đầu {len(files)} file...")

        model = self.model_name.get().strip() or "gemini-2.0-flash"
        fmt = (self.output_format.get().strip() or "Text").lower()
        lang_name = self.target_lang.get().strip() or "Vietnamese"
        lang_code = self.lang_code.get().strip() or "vi"
        locale = self.locale.get().strip()

        def task():
            processed = 0
            for srt_path in files:
                try:
                    base = os.path.splitext(os.path.basename(srt_path))[0]
                    out_dir = get_output_dir(srt_path)
                    out_path = os.path.join(out_dir, f"{base}_script_{lang_code}.txt")
                    decision = self._confirm_overwrite(out_path)
                    if decision is False:
                        self._log(f"[Batch Gemini] Bỏ qua: {out_path}")
                        processed += 1; continue
                    srt_text = self._read_first_text_file(srt_path)
                    duration = srt_duration_seconds_from_text(srt_text)
                    try:
                        override = float(self.override_rate.get().strip()) if self.override_rate.get().strip() else None
                    except Exception:
                        override = None
                    lb = length_budget(duration, lang_code, override)
                    prompt = build_prompt_text(srt_text, lang_name, locale, duration, lb, lang_code)
                    result = call_gemini(api_key, model, prompt, want_json=(fmt=="json"))
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(result)
                    self.settings["last_output_dir"] = out_dir
                    save_settings(self.settings)
                    self._log(f"[Batch Gemini] Done → {out_path}")
                except Exception as e:
                    self._log(f"[Batch Gemini] Lỗi {os.path.basename(srt_path)}: {e}")
                finally:
                    processed += 1
            messagebox.showinfo("Xong", f"Đã xử lý SRT → Kịch bản: {processed}/{len(files)} file")

        threading.Thread(target=task, daemon=True).start()

    def on_txt_dir_pipeline(self):
        txt_dir = self.txt_dir.get().strip()
        api_txt = self.eleven_api_txt.get().strip()
        voice_id = self.voice_id.get().strip()
        model_id = self.eleven_model.get().strip() or MODEL_ELEVEN_DEFAULT
        if not txt_dir or not os.path.isdir(txt_dir):
            messagebox.showerror("Thiếu thư mục", "Hãy chọn thư mục TXT hợp lệ"); return
        if not api_txt or not os.path.isfile(api_txt):
            messagebox.showerror("Thiếu API Eleven", "Hãy chọn file .txt chứa danh sách API"); return
        if not voice_id:
            messagebox.showerror("Thiếu Voice ID", "Hãy nhập Voice ID"); return
        if AudioSegment is None or requests is None:
            messagebox.showerror("Thiếu thư viện", "Cần requests + pydub và ffmpeg trong PATH"); return
        pool = self._read_eleven_keys(api_txt)
        if not pool:
            messagebox.showerror("API rỗng", "Không đọc được API"); return
        rotator = KeyRotator(pool)
        files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if os.path.splitext(f)[1].lower() == ".txt"]
        if not files:
            messagebox.showinfo("Trống", "Không có file .txt trong thư mục."); return
        self._log(f"[Batch TTS] Bắt đầu {len(files)} file...")

        def task():
            done = 0
            for txt_path in files:
                try:
                    base = os.path.splitext(os.path.basename(txt_path))[0]
                    out_dir = get_output_dir(txt_path)
                    out_path = os.path.join(out_dir, f"{base}.mp3")
                    decision = self._confirm_overwrite(out_path)
                    if decision is False:
                        self._log(f"[Batch TTS] Bỏ qua: {out_path}")
                        done += 1; continue
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        self._log(f"[Batch TTS] Rỗng: {txt_path}")
                        done += 1; continue
                    audio = self._tts_join_chunks(text, rotator, voice_id, model_id)
                    audio.export(out_path, format="mp3")
                    self.settings["last_output_dir"] = out_dir
                    save_settings(self.settings)
                    self._log(f"[Batch TTS] Done → {out_path}")
                except KeyExhaustedError as e:
                    self._log(f"[Batch TTS] Hết API tại {os.path.basename(txt_path)}: {e}")
                    break
                except ApiError as e:
                    self._log(f"[Batch TTS] Lỗi API {os.path.basename(txt_path)}: {e}")
                except Exception as e:
                    self._log(f"[Batch TTS] Lỗi {os.path.basename(txt_path)}: {e}")
                finally:
                    done += 1
            messagebox.showinfo("Xong", f"Đã xử lý TXT → Audio: {done}/{len(files)} file")


    def on_srt_to_script(self):
        gem_api_path = self.gem_api_file.get().strip()
        srt_path = self.srt_file.get().strip()
        if not gem_api_path or not os.path.isfile(gem_api_path):
            messagebox.showerror("Thiếu API Gemini", "Hãy chọn file API Gemini (.txt)")
            return
        if not srt_path or not os.path.isfile(srt_path):
            messagebox.showerror("Thiếu file SRT", "Hãy chọn file .srt hợp lệ")
            return
        try:
            api_key = self._read_first_text_file(gem_api_path)
        except Exception as e:
            messagebox.showerror("Lỗi API Gemini", str(e)); return
        if not api_key:
            messagebox.showerror("API Gemini trống", "File API không có nội dung"); return

        model = self.model_name.get().strip() or "gemini-2.0-flash"
        fmt = (self.output_format.get().strip() or "Text").lower()
        lang_name = self.target_lang.get().strip() or "Vietnamese"
        lang_code = self.lang_code.get().strip() or "vi"
        locale = self.locale.get().strip()
        try:
            srt_text = self._read_first_text_file(srt_path)
        except Exception as e:
            messagebox.showerror("Lỗi đọc SRT", str(e)); return
        duration = srt_duration_seconds_from_text(srt_text)
        try:
            override = float(self.override_rate.get().strip()) if self.override_rate.get().strip() else None
        except Exception:
            override = None
        lb = length_budget(duration, lang_code, override)
        want_json = (fmt == "json")
        if fmt == "text":
            prompt = build_prompt_text(srt_text, lang_name, locale, duration, lb, lang_code)
        else:
            # fallback to plain text prompt to keep narration-friendly output
            prompt = build_prompt_text(srt_text, lang_name, locale, duration, lb, lang_code)

        self._log("[Gemini] Đang tạo kịch bản...")
        def task():
            try:
                result = call_gemini(api_key, model, prompt, want_json)
                base = os.path.splitext(os.path.basename(srt_path))[0]
                out_dir = get_output_dir(srt_path)
                out_path = os.path.join(out_dir, f"{base}_script_{lang_code}.txt")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result)
                self.script_file.set(out_path)
                self.settings.update({
                    "gem_model": model,
                    "output_format": self.output_format.get().strip(),
                    "target_lang": self.target_lang.get().strip(),
                    "lang_code": lang_code,
                    "locale": self.locale.get().strip(),
                    "override_rate": self.override_rate.get().strip(),
                    "gem_api_file": gem_api_path,
                    "last_output_dir": out_dir,
                })
                save_settings(self.settings)
                self._log(f"[Gemini] Xong. {out_path}")
                messagebox.showinfo("Xong", f"Đã tạo kịch bản:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Lỗi Gemini", str(e))
                self._log(f"[Gemini] Lỗi: {e}")
        threading.Thread(target=task, daemon=True).start()

    def _read_eleven_keys(self, path: str) -> List[str]:
        keys: List[str] = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    k = line.strip()
                    if k and not k.startswith('#'):
                        keys.append(k)
        except Exception:
            pass
        return keys

    def _tts_join_chunks(self, text: str, rotator: KeyRotator, voice_id: str, model_id: str) -> AudioSegment:
        chunks = split_text_into_chunks(text)
        if not chunks:
            return AudioSegment.silent(duration=400)
        final_audio = AudioSegment.silent(duration=0)
        for idx, chunk in enumerate(chunks, start=1):
            seg = self._tts_with_rotator(chunk, voice_id, rotator, model_id, file_path="<txt>", idx=idx, total=len(chunks))
            final_audio += seg
            if idx != len(chunks):
                final_audio += AudioSegment.silent(duration=PAUSE_BETWEEN_CHUNKS_MS)
        return final_audio

    def _tts_with_rotator(self, text: str, voice_id: str, rotator: KeyRotator, model_id: str, file_path: str, idx: int, total: int) -> AudioSegment:
        attempts = 0
        last_err: Optional[ApiError] = None
        while attempts < max(1, rotator.count()):
            if rotator.count() == 0:
                raise KeyExhaustedError(file_path, idx, total)
            api_key = rotator.next()
            try:
                return text_to_speech(api_key, voice_id, text, model_id)
            except ApiError as e:
                last_err = e
                # Auto-handle voice_limit_reached: free one custom voice silently then retry immediately with same key
                if e.status_code == 400 and _is_voice_limit_error_message(e.message):
                    # Attempt cleanup once
                    try:
                        self._log("[TTS] Đã đạt giới hạn custom voice. Đang dọn bớt để tiếp tục...")
                        if eleven_free_one_custom_voice(api_key, log_func=self._log):
                            # After cleanup, try same key again once
                            try:
                                return text_to_speech(api_key, voice_id, text, model_id)
                            except ApiError as e2:
                                last_err = e2
                                # fall through to normal rotation rules
                    except Exception as _:
                        pass
                if e.status_code in (401, 403, 429):
                    rotator.remove(api_key)
                attempts += 1
                if rotator.count() == 0:
                    raise KeyExhaustedError(file_path, idx, total)
                continue
        if last_err:
            raise last_err
        raise ApiError(500, "Không thể tạo audio cho đoạn hiện tại")

    def on_script_to_audio(self):
        txt_path = self.script_file.get().strip()
        api_txt = self.eleven_api_txt.get().strip()
        voice_id = self.voice_id.get().strip()
        model_id = self.eleven_model.get().strip() or MODEL_ELEVEN_DEFAULT
        if not txt_path or not os.path.isfile(txt_path):
            messagebox.showerror("Thiếu kịch bản", "Hãy chọn file .txt hợp lệ"); return
        if not api_txt or not os.path.isfile(api_txt):
            messagebox.showerror("Thiếu API Eleven", "Hãy chọn file .txt chứa danh sách API"); return
        if not voice_id:
            messagebox.showerror("Thiếu Voice ID", "Hãy nhập Voice ID"); return
        if AudioSegment is None or requests is None:
            messagebox.showerror("Thiếu thư viện", "Cần requests + pydub và ffmpeg trong PATH"); return
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            messagebox.showerror("Lỗi đọc TXT", str(e)); return
        if not text:
            messagebox.showerror("TXT trống", "Không có nội dung"); return
        pool = self._read_eleven_keys(api_txt)
        if not pool:
            messagebox.showerror("API rỗng", "Không đọc được API"); return
        rotator = KeyRotator(pool)
        base = os.path.splitext(os.path.basename(txt_path))[0]
        out_dir = get_output_dir(txt_path)
        out_path = os.path.join(out_dir, f"{base}.mp3")

        self._log("[TTS] Đang tạo audio từ TXT...")
        def task():
            try:
                audio = self._tts_join_chunks(text, rotator, voice_id, model_id)
                audio.export(out_path, format="mp3")
                self.settings.update({
                    "eleven_api_txt": api_txt,
                    "voice_id": voice_id,
                    "eleven_model": model_id,
                    "audio_format": "mp3",
                    "last_output_dir": out_dir,
                })
                save_settings(self.settings)
                self._log(f"[TTS] Xong. {out_path}")
                messagebox.showinfo("Xong", f"Đã tạo audio:\n{out_path}")
            except KeyExhaustedError as e:
                self._log(str(e)); messagebox.showerror("Hết API", str(e))
            except ApiError as e:
                friendly = describe_api_error(e); self._log(friendly)
                messagebox.showerror("Lỗi API", f"{friendly}\n\nChi tiết:\n{e}")
            except Exception as e:
                self._log(f"[TTS] Lỗi: {e}"); messagebox.showerror("Lỗi TTS", str(e))
        threading.Thread(target=task, daemon=True).start()

    def on_srt_to_audio(self):
        srt_path = self.srt_file.get().strip()
        api_txt = self.eleven_api_txt.get().strip()
        voice_id = self.voice_id.get().strip()
        model_id = self.eleven_model.get().strip() or MODEL_ELEVEN_DEFAULT
        if not srt_path or not os.path.isfile(srt_path):
            messagebox.showerror("Thiếu SRT", "Hãy chọn file .srt hợp lệ"); return
        if not api_txt or not os.path.isfile(api_txt):
            messagebox.showerror("Thiếu API Eleven", "Hãy chọn file .txt chứa danh sách API"); return
        if not voice_id:
            messagebox.showerror("Thiếu Voice ID", "Hãy nhập Voice ID"); return
        if pysrt is None or AudioSegment is None or requests is None:
            messagebox.showerror("Thiếu thư viện", "Cần pysrt + requests + pydub và ffmpeg trong PATH"); return
        try:
            subs = pysrt.open(srt_path)
        except Exception as e:
            messagebox.showerror("Lỗi đọc SRT", str(e)); return
        pool = self._read_eleven_keys(api_txt)
        if not pool:
            messagebox.showerror("API rỗng", "Không đọc được API"); return
        rotator = KeyRotator(pool)
        base = os.path.splitext(os.path.basename(srt_path))[0]
        out_dir = get_output_dir(srt_path)
        out_path = os.path.join(out_dir, f"{base}_tts.mp3")

        self._log("[TTS] Đang tạo audio từ SRT (khớp timeline)...")
        def task():
            try:
                last_end = max((it.end.ordinal for it in subs), default=0)
                final_audio = AudioSegment.silent(duration=last_end)
                total = len(subs)
                for idx, sub in enumerate(subs, start=1):
                    text = sub.text.replace("\n", " ")
                    start_ms = sub.start.ordinal
                    end_ms = sub.end.ordinal
                    seg_dur = max(0, end_ms - start_ms)
                    seg = self._tts_with_rotator(text, voice_id, rotator, model_id, srt_path, idx, total)
                    fitted = fit_to_duration(seg, seg_dur)
                    final_audio = final_audio.overlay(fitted, position=start_ms)
                final_audio.export(out_path, format="mp3")
                self.settings.update({
                    "eleven_api_txt": api_txt,
                    "voice_id": voice_id,
                    "eleven_model": model_id,
                    "audio_format": "mp3",
                    "last_output_dir": out_dir,
                })
                save_settings(self.settings)
                self._log(f"[TTS] Xong. {out_path}")
                messagebox.showinfo("Xong", f"Đã tạo audio:\n{out_path}")
            except KeyExhaustedError as e:
                self._log(str(e)); messagebox.showerror("Hết API", str(e))
            except ApiError as e:
                friendly = describe_api_error(e); self._log(friendly)
                messagebox.showerror("Lỗi API", f"{friendly}\n\nChi tiết:\n{e}")
            except Exception as e:
                self._log(f"[TTS] Lỗi: {e}"); messagebox.showerror("Lỗi TTS", str(e))
        threading.Thread(target=task, daemon=True).start()

    def on_pipeline_full(self):
        # Chain: media→SRT then srt→script then script→audio
        def _after_media():
            if self.srt_file.get().strip():
                # Không cần lưu đường dẫn trung gian vào UI khi chạy tổng
                srt_path_prev = self.srt_file.get().strip()
                self.on_srt_to_script()
                # Next step scheduled after script success via simple delay check
                def _wait_script_then_tts():
                    # poll if script file exists
                    for _ in range(120):  # up to ~60s
                        p = self.script_file.get().strip()
                        if p and os.path.exists(p):
                            self.on_script_to_audio()
                            return
                        time.sleep(0.5)
                threading.Thread(target=_wait_script_then_tts, daemon=True).start()

        # First step
        def run_media_to_srt():
            self.on_media_to_srt()
            # monitor for srt
            def _wait_srt_then_script():
                for _ in range(240):  # up to ~20min if long transcription
                    p = self.srt_file.get().strip()
                    if p and os.path.exists(p):
                        _after_media()
                        return
                    time.sleep(5.0)
            threading.Thread(target=_wait_srt_then_script, daemon=True).start()
        run_media_to_srt()

    def open_output_folder(self):
        folder = self.settings.get("last_output_dir", "")
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("Mở thư mục", "Chưa có thư mục đầu ra hoặc thư mục không tồn tại.")
            return
        try:
            if os.name == "nt":
                os.startfile(folder)  # type: ignore
            elif os.name == "posix":
                import subprocess
                subprocess.run(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                import subprocess
                subprocess.run(["open", folder])
        except Exception:
            messagebox.showinfo("Mở thư mục", f"Vui lòng mở thủ công: {folder}")


def main():
    app = App()
    def _on_close():
        try:
            save_settings(app.settings)
        finally:
            app.destroy()
    app.protocol("WM_DELETE_WINDOW", _on_close)
    app.mainloop()


if __name__ == "__main__":
    main()


