"""Sample-exact audio nodes."""

from __future__ import annotations

from ..cauce.audio import authoritative_audio, empty_audio, place_audio, slice_audio


class CauceEmptyAudioTrack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_seconds": (
                    "FLOAT",
                    {"default": 60.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
                "sample_rate": (["32000", "44100", "48000"], {"default": "32000"}),
                "channels": (["1", "2"], {"default": "2"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "build"
    CATEGORY = "CAUCE/Audio"

    def build(self, duration_seconds, sample_rate, channels):
        return (empty_audio(duration_seconds, int(sample_rate), int(channels)),)


class CauceAudioSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
                "pad_silence": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "sample_count")
    FUNCTION = "slice"
    CATEGORY = "CAUCE/Audio"

    def slice(self, audio, start_seconds, duration_seconds, pad_silence):
        result = slice_audio(
            audio,
            start=start_seconds,
            duration=duration_seconds,
            pad=pad_silence,
        )
        return result, int(result["waveform"].shape[-1])


class CaucePlaceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track": ("AUDIO",),
                "clip": ("AUDIO",),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "gain": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01},
                ),
                "mode": (["add", "replace"], {"default": "add"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "place"
    CATEGORY = "CAUCE/Audio"

    def place(self, track, clip, start_seconds, gain, mode):
        return (place_audio(track, clip, start=start_seconds, gain=gain, mode=mode),)


class CauceAuthoritativeAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "master_audio": ("AUDIO",),
                "start_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.001, "max": 99_999.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "extract"
    CATEGORY = "CAUCE/Audio"
    DESCRIPTION = "Return the exact master samples used for final muxing, independent of the AudioVAE."

    def extract(self, master_audio, start_seconds, duration_seconds):
        return (
            authoritative_audio(
                master_audio, start=start_seconds, duration=duration_seconds
            ),
        )


NODE_CLASS_MAPPINGS = {
    "CauceEmptyAudioTrack": CauceEmptyAudioTrack,
    "CauceAudioSlice": CauceAudioSlice,
    "CaucePlaceAudio": CaucePlaceAudio,
    "CauceAuthoritativeAudio": CauceAuthoritativeAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceEmptyAudioTrack": "CAUCE · Empty Audio Track",
    "CauceAudioSlice": "CAUCE · Exact Audio Slice",
    "CaucePlaceAudio": "CAUCE · Place Audio",
    "CauceAuthoritativeAudio": "CAUCE · Authoritative Audio",
}
