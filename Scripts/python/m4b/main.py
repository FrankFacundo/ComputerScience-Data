#!/usr/bin/env python3
"""
m4b_inspect.py — Inspect M4B audiobooks

Features:
- Reads MP4/M4B tags (title, author, album, genre, year, track/disc, comments, etc.)
- Reads chapters (via mutagen when available; also via ffprobe)
- Extracts technical info (duration, bitrate, sample rate, channels, codec, container)
- Optionally exports embedded cover art
- Pretty JSON output; optionally write to file

Usage:
  python m4b_inspect.py /path/to/book.m4b
  python m4b_inspect.py book.m4b --export-cover cover.jpg --out book.json
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# --- Helpers -----------------------------------------------------------------


def which(cmd: str) -> Optional[str]:
    from shutil import which as _which

    return _which(cmd)


def run_ffprobe(path: str) -> Optional[Dict[str, Any]]:
    if not which("ffprobe"):
        return None
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return json.loads(out.decode("utf-8", "replace"))
    except Exception:
        return None


def fmt_time(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    try:
        s = float(seconds)
    except Exception:
        return None
    m, s = divmod(s, 60.0)
    h, m = divmod(m, 60.0)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def ensure_jsonable(obj):
    """Convert non-JSONable mutagen/ffprobe objects to plain types."""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", "replace")
        except Exception:
            return obj.hex()
    if isinstance(obj, (list, tuple)):
        return [ensure_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): ensure_jsonable(v) for k, v in obj.items()}
    # Fall back to string for unknown objects
    try:
        json.dumps(obj)  # test
        return obj
    except Exception:
        return str(obj)


# --- Mutagen (MP4/M4B) -------------------------------------------------------


def inspect_mutagen(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {"source": "mutagen", "ok": False}
    try:
        from mutagen.mp4 import MP4, MP4Cover
    except Exception as e:
        data["error"] = f"mutagen not available: {e}"
        return data

    try:
        mp4 = MP4(path)
    except Exception as e:
        data["error"] = f"Failed to parse MP4/M4B: {e}"
        return data

    info = getattr(mp4, "info", None)
    tags = getattr(mp4, "tags", None)

    # Technical info
    tech = {}
    if info:
        tech = {
            "duration_sec": getattr(info, "length", None),
            "duration_hms": fmt_time(getattr(info, "length", None)),
            "bitrate": getattr(info, "bitrate", None),
            "sample_rate": getattr(info, "sample_rate", None),
            "channels": getattr(info, "channels", None),
        }

    # Tags (human-friendly mapping for common atoms)
    # Many MP4 atoms are ©-prefixed; mutagen exposes them as such.
    keymap = {
        "©nam": "title",
        "©ART": "artist",
        "aART": "album_artist",
        "©alb": "album",
        "©gen": "genre",
        "©day": "year",
        "©wrt": "writer",
        "©cmt": "comment",
        "desc": "description",
        "ldes": "long_description",
        "©too": "encoder",
        "trkn": "track_number",
        "disk": "disc_number",
        "cpil": "compilation",
        "tmpo": "tempo",
        "pgap": "gapless",
        "tvsh": "show",
        "tvnn": "network",
        "tves": "episode",
        "tvsn": "season",
        "stik": "media_type",
        "covr": "cover",
    }

    meta: Dict[str, Any] = {}
    covers: List[Dict[str, Any]] = []

    if tags:
        # First pass: map known keys
        for k, v in tags.items():
            if k == "covr":
                # cover(s)
                for cov in v:
                    fmt = None
                    try:
                        fmt = (
                            "jpeg"
                            if cov.imageformat == MP4Cover.FORMAT_JPEG
                            else (
                                "png"
                                if cov.imageformat == MP4Cover.FORMAT_PNG
                                else "unknown"
                            )
                        )
                    except Exception:
                        pass
                    covers.append(
                        {
                            "bytes": len(cov) if hasattr(cov, "__len__") else None,
                            "format": fmt,
                        }
                    )
                meta[keymap.get(k, k)] = covers
            elif k in ("trkn", "disk"):
                # e.g. [(3, 20)]
                val = v[0] if isinstance(v, list) and v else v
                if isinstance(val, (list, tuple)) and len(val) >= 2:
                    meta[keymap.get(k, k)] = {"number": val[0], "total": val[1]}
                else:
                    meta[keymap.get(k, k)] = ensure_jsonable(v)
            else:
                meta[keymap.get(k, k)] = ensure_jsonable(v)

        # Keep unknown/freeform atoms too
        unknown = {}
        for k, v in tags.items():
            human = keymap.get(k)
            if not human:
                unknown[k] = ensure_jsonable(v)
        if unknown:
            meta["other_atoms"] = unknown

    # Chapters via mutagen
    chapters: List[Dict[str, Any]] = []
    chs = getattr(mp4, "chapters", None)
    if chs:
        # mutagen returns MP4Chapter objects with .title, .start_time, .end_time (units vary by version)
        for idx, ch in enumerate(chs, 1):
            title = getattr(ch, "title", f"Chapter {idx}")
            start = getattr(ch, "start_time", None)
            end = getattr(ch, "end_time", None)

            # Best-effort to treat times as seconds if they look like seconds; otherwise leave raw
            start_sec = float(start) if isinstance(start, (int, float)) else None
            end_sec = float(end) if isinstance(end, (int, float)) else None
            # If clearly too large to be seconds (e.g., > 10^6), assume milliseconds
            if start_sec is not None and start_sec > 1_000_000:
                start_sec = start_sec / 1000.0
            if end_sec is not None and end_sec > 1_000_000:
                end_sec = end_sec / 1000.0

            chapters.append(
                {
                    "index": idx,
                    "title": title,
                    "start_sec": start_sec,
                    "start_hms": fmt_time(start_sec) if start_sec is not None else None,
                    "end_sec": end_sec,
                    "end_hms": fmt_time(end_sec) if end_sec is not None else None,
                    "raw_start": start if start_sec is None else None,
                    "raw_end": end if end_sec is None else None,
                }
            )

    data.update(
        {
            "ok": True,
            "tech": tech,
            "metadata": meta,
            "chapters": chapters,
            "has_cover": bool(covers),
        }
    )
    return data


# --- Cover export -------------------------------------------------------------


def export_cover_mutagen(path: str, outpath: str) -> bool:
    try:
        from mutagen.mp4 import MP4
    except Exception:
        return False
    try:
        mp4 = MP4(path)
        tags = mp4.tags or {}
        covrs = tags.get("covr")
        if not covrs:
            return False
        # Save the first cover
        cov = covrs[0]
        with open(outpath, "wb") as f:
            f.write(bytes(cov))
        return True
    except Exception:
        return False


# --- Merge/normalize results --------------------------------------------------


def merge_results(mut: Dict[str, Any], ffp: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "file": {},
        "technical": {},
        "metadata": {},
        "chapters": [],
        "streams": [],
        "raw": {},
    }

    # File info
    res["file"] = {
        "path": mut.get("file_path"),
    }

    # Technical (prefer mutagen for duration; ffprobe has more details)
    mtech = mut.get("tech") or {}
    res["technical"].update(mtech)
    if ffp:
        fmt = ffp.get("format") or {}
        if "duration" in fmt and not res["technical"].get("duration_sec"):
            try:
                res["technical"]["duration_sec"] = float(fmt["duration"])
                res["technical"]["duration_hms"] = fmt_time(float(fmt["duration"]))
            except Exception:
                pass
        # Container and overall bitrate
        if "format_name" in fmt:
            res["technical"]["container"] = fmt.get("format_name")
        if "bit_rate" in fmt and not res["technical"].get("bitrate"):
            try:
                res["technical"]["bitrate"] = int(fmt["bit_rate"])
            except Exception:
                res["technical"]["bitrate"] = fmt.get("bit_rate")

    # Metadata
    res["metadata"].update(mut.get("metadata") or {})
    if ffp:
        # ffprobe puts tags under format.tags
        ftags = (ffp.get("format") or {}).get("tags") or {}
        # Don't overwrite existing mutagen-derived human names; add missing ones under "extra_ffprobe_tags"
        extras = {}
        for k, v in ftags.items():
            if k not in res["metadata"]:
                extras[k] = v
        if extras:
            res["metadata"]["extra_ffprobe_tags"] = extras

    # Chapters (prefer mutagen; if empty, use ffprobe)
    chapters = mut.get("chapters") or []
    if not chapters and ffp and "chapters" in ffp:
        chs = []
        for i, ch in enumerate(ffp["chapters"], 1):
            st = ch.get("start_time")
            et = ch.get("end_time")
            stf = float(st) if st is not None else None
            etf = float(et) if et is not None else None
            title = (ch.get("tags") or {}).get("title") or f"Chapter {i}"
            chs.append(
                {
                    "index": i,
                    "title": title,
                    "start_sec": stf,
                    "start_hms": fmt_time(stf),
                    "end_sec": etf,
                    "end_hms": fmt_time(etf),
                }
            )
        chapters = chs
    res["chapters"] = chapters

    # Streams (audio stream details)
    if ffp:
        for s in ffp.get("streams", []):
            if s.get("codec_type") != "audio":
                continue
            stream = {
                "index": s.get("index"),
                "codec": s.get("codec_name"),
                "profile": s.get("profile"),
                "sample_rate": s.get("sample_rate"),
                "channels": s.get("channels"),
                "channel_layout": s.get("channel_layout"),
                "bit_rate": s.get("bit_rate"),
                "duration_sec": float(s["duration"]) if s.get("duration") else None,
                "duration_hms": fmt_time(float(s["duration"]))
                if s.get("duration")
                else None,
            }
            res["streams"].append(stream)

    # Raw (for debugging/advanced users)
    res["raw"]["mutagen"] = mut
    if ffp:
        res["raw"]["ffprobe"] = ffp

    return res


# --- CLI ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Inspect M4B audiobook metadata and chapters."
    )
    ap.add_argument("m4b", help="Path to the .m4b file")
    ap.add_argument(
        "--export-cover",
        metavar="OUT_IMAGE",
        default="./conver.png",  # ← new
        help="Export embedded cover art to file (jpg/png)",
    )
    ap.add_argument(
        "--out",
        metavar="OUT_JSON",
        default="./out.json",  # ← new
        help="Write resulting JSON to file instead of stdout",
    )
    args = ap.parse_args()

    path = os.path.abspath(args.m4b)
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    mut = inspect_mutagen(path)
    mut["file_path"] = path

    ffp = run_ffprobe(path)

    result = merge_results(mut, ffp)

    if args.export_cover:
        ok = export_cover_mutagen(path, args.export_cover)
        result["cover_exported"] = ok
        result["cover_export_path"] = args.export_cover if ok else None

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(out_json)
        print(f"Wrote JSON to {args.out}")
    else:
        print(out_json)


if __name__ == "__main__":
    main()
