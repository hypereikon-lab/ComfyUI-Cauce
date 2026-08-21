"""Sample-exact audio track operations used by CAUCE nodes."""

from __future__ import annotations

from typing import Any

from .timebase import as_fraction, seconds_to_samples


def validate_audio(audio: dict[str, Any]) -> tuple[Any, int]:
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        raise ValueError("audio must contain waveform and a positive sample_rate")
    if getattr(waveform, "ndim", 0) != 3:
        raise ValueError("audio waveform must have shape [batch,channels,samples]")
    return waveform, sample_rate


def resample_audio(audio: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    waveform, source_rate = validate_audio(audio)
    sample_rate = int(sample_rate)
    if source_rate == sample_rate:
        return {"waveform": waveform, "sample_rate": source_rate}
    try:
        import torchaudio
    except ImportError as exc:  # pragma: no cover - ships in the H3 runtime
        raise RuntimeError("torchaudio is required to resample audio") from exc
    return {
        "waveform": torchaudio.functional.resample(waveform, source_rate, sample_rate),
        "sample_rate": sample_rate,
    }


def slice_audio(
    audio: dict[str, Any],
    *,
    start: object,
    duration: object,
    pad: bool = True,
) -> dict[str, Any]:
    import torch.nn.functional as functional

    waveform, sample_rate = validate_audio(audio)
    start_f = as_fraction(start)
    duration_f = as_fraction(duration)
    if start_f < 0 or duration_f <= 0:
        raise ValueError("audio slice requires start >= 0 and duration > 0")
    first = seconds_to_samples(start_f, sample_rate)
    count = seconds_to_samples(duration_f, sample_rate)
    if first >= waveform.shape[-1]:
        if not pad:
            raise ValueError("audio slice starts beyond the source waveform")
        result = waveform[..., :0]
    else:
        result = waveform[..., first : first + count]
    if result.shape[-1] < count:
        if not pad:
            raise ValueError("audio slice extends beyond the source waveform")
        result = functional.pad(result, (0, count - result.shape[-1]))
    return {"waveform": result, "sample_rate": sample_rate}


def empty_audio(duration: object, sample_rate: int = 32000, channels: int = 2):
    import torch

    count = seconds_to_samples(duration, int(sample_rate))
    if count < 1:
        raise ValueError("audio duration must produce at least one sample")
    return {
        "waveform": torch.zeros((1, int(channels), count), dtype=torch.float32),
        "sample_rate": int(sample_rate),
    }


def place_audio(
    track: dict[str, Any],
    clip: dict[str, Any],
    *,
    start: object,
    gain: float = 1.0,
    mode: str = "add",
) -> dict[str, Any]:
    import torch

    if mode not in {"add", "replace"}:
        raise ValueError("audio placement mode must be add or replace")
    track_waveform, sample_rate = validate_audio(track)
    clip = resample_audio(clip, sample_rate)
    clip_waveform, _ = validate_audio(clip)
    first = seconds_to_samples(start, sample_rate)
    if first < 0:
        raise ValueError("audio placement cannot start before zero")
    needed = first + int(clip_waveform.shape[-1])
    if needed > int(track_waveform.shape[-1]):
        raise ValueError("audio clip extends beyond the target track")

    channels = max(int(track_waveform.shape[1]), int(clip_waveform.shape[1]))
    if track_waveform.shape[1] == 1 and channels == 2:
        track_waveform = track_waveform.expand(-1, 2, -1)
    if clip_waveform.shape[1] == 1 and channels == 2:
        clip_waveform = clip_waveform.expand(-1, 2, -1)
    if track_waveform.shape[1] != clip_waveform.shape[1]:
        raise ValueError("audio track and clip channel counts are incompatible")

    result = track_waveform.clone()
    placed = clip_waveform.to(result) * float(gain)
    if mode == "replace":
        result[..., first:needed] = placed
    else:
        result[..., first:needed] += placed
    return {"waveform": result, "sample_rate": sample_rate}


def authoritative_audio(
    master_audio: dict[str, Any],
    *,
    start: object,
    duration: object,
) -> dict[str, Any]:
    """Return the exact master samples for the requested final media range."""

    return slice_audio(master_audio, start=start, duration=duration, pad=True)
