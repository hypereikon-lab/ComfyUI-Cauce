"""Exact MiniMax H3 visible-frame and audiovisual-latent geometry."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Literal


H3_FPS = Fraction(24, 1)
H3_AUDIO_LATENT_HZ = Fraction(40, 1)
H3_FRAME_STEP = 17
H3_FRAME_OFFSET = 5
H3_MIN_FRAMES = 5
H3_TRAINED_MIN_FRAMES = 124
H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

RoundMode = Literal["floor", "ceil", "nearest"]
Numberish = Fraction | int | float | str


def as_fraction(value: Numberish) -> Fraction:
    """Convert UI-friendly numeric values without inheriting float drift."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        return Fraction(int(value), 1)
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("time values must be finite")
        return Fraction(str(value))
    text = str(value).strip()
    if not text:
        raise ValueError("time values cannot be empty")
    return Fraction(text)


def floor_fraction(value: Fraction) -> int:
    return value.numerator // value.denominator


def ceil_fraction(value: Fraction) -> int:
    return -((-value.numerator) // value.denominator)


def round_fraction(value: Fraction) -> int:
    """Round to nearest integer, with halves away from zero.

    H3's legal 24->40 Hz conversions never land on an exact half, but using an
    explicit rule keeps every other media conversion reproducible.
    """

    sign = -1 if value < 0 else 1
    value = abs(value)
    whole, remainder = divmod(value.numerator, value.denominator)
    if remainder * 2 >= value.denominator:
        whole += 1
    return sign * whole


def quantize(value: Fraction, mode: RoundMode) -> int:
    if mode == "floor":
        return floor_fraction(value)
    if mode == "ceil":
        return ceil_fraction(value)
    if mode == "nearest":
        return round_fraction(value)
    raise ValueError(f"unsupported rounding mode: {mode}")


def seconds_to_frames(seconds: Numberish, fps: Numberish = H3_FPS, mode: RoundMode = "nearest") -> int:
    seconds_f = as_fraction(seconds)
    fps_f = as_fraction(fps)
    if seconds_f < 0:
        raise ValueError("seconds cannot be negative")
    if fps_f <= 0:
        raise ValueError("fps must be positive")
    return quantize(seconds_f * fps_f, mode)


def frames_to_seconds(frames: int, fps: Numberish = H3_FPS) -> Fraction:
    fps_f = as_fraction(fps)
    if int(frames) < 0:
        raise ValueError("frames cannot be negative")
    if fps_f <= 0:
        raise ValueError("fps must be positive")
    return Fraction(int(frames), 1) / fps_f


def fraction_payload(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def is_h3_frame_count(frames: int) -> bool:
    frames = int(frames)
    return frames >= H3_MIN_FRAMES and (frames - H3_FRAME_OFFSET) % H3_FRAME_STEP == 0


def h3_visual_latent_frames(pixel_frames: int) -> int:
    if not is_h3_frame_count(pixel_frames):
        raise ValueError("H3 pixel frames must lie on the 17k+5 grid")
    if pixel_frames <= 5:
        return 2
    return ((int(pixel_frames) - 5) // 17) * 5 + 2


def h3_audio_latent_frames(pixel_frames: int) -> int:
    if not is_h3_frame_count(pixel_frames):
        raise ValueError("H3 pixel frames must lie on the 17k+5 grid")
    return round_fraction(Fraction(int(pixel_frames), 1) / H3_FPS * H3_AUDIO_LATENT_HZ)


def visual_token_spans(token_count: int) -> tuple[tuple[int, int], ...]:
    if int(token_count) < 1:
        raise ValueError("token_count must be positive")
    cursor = 0
    spans: list[tuple[int, int]] = []
    for index in range(int(token_count)):
        end = cursor + H3_FRAME_PER_TOKEN[index % len(H3_FRAME_PER_TOKEN)]
        spans.append((cursor, end))
        cursor = end
    return tuple(spans)


def visual_token_count_for_span(pixel_frames: int) -> int:
    """Return token count when a frame span is exactly representable."""

    if int(pixel_frames) < 1:
        raise ValueError("pixel_frames must be positive")
    covered = 0
    tokens = 0
    while covered < int(pixel_frames):
        covered += H3_FRAME_PER_TOKEN[tokens % len(H3_FRAME_PER_TOKEN)]
        tokens += 1
    if covered != int(pixel_frames):
        raise ValueError(
            f"{pixel_frames} pixel frames do not end on an H3 visual-token boundary"
        )
    return tokens


def visual_span_for_tokens(token_count: int) -> int:
    spans = visual_token_spans(token_count)
    return spans[-1][1]


@dataclass(frozen=True)
class H3Shape:
    pixel_frames: int
    video_latent_frames: int
    audio_latent_frames: int
    duration: Fraction

    @classmethod
    def from_frames(cls, frames: int) -> "H3Shape":
        frames = int(frames)
        return cls(
            pixel_frames=frames,
            video_latent_frames=h3_visual_latent_frames(frames),
            audio_latent_frames=h3_audio_latent_frames(frames),
            duration=frames_to_seconds(frames),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "pixel_frames": self.pixel_frames,
            "video_latent_frames": self.video_latent_frames,
            "audio_latent_frames": self.audio_latent_frames,
            "duration": fraction_payload(self.duration),
            "duration_seconds": float(self.duration),
        }
