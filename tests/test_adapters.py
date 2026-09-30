import struct
import tempfile
import threading
import unittest

import numpy as np

from open_gptlive_poc.adapters.supertonic_speaker import SupertonicSpeaker, _resample_pcm16le
from pathlib import Path

from open_gptlive_poc.adapters.silero_vad import SileroVAD, resample_pcm16le
from open_gptlive_poc.adapters.faster_whisper import FasterWhisperASR


class FakeSegment:
    text = "hello local model"
    end = 2.0


class FakeModel:
    last_sample_count = 0
    language = None

    def transcribe(self, samples, language, vad_filter):
        self.last_sample_count = len(samples)
        self.language = language
        return iter([FakeSegment()]), object()


class FakeSupertonic:
    def __init__(self) -> None:
        self.voice_name = None

    def get_voice_style(self, voice_name: str):
        self.voice_name = voice_name
        return voice_name

    def synthesize(
        self,
        text: str,
        *,
        voice_style: str,
        lang: str,
        max_chunk_length: int,
        silence_duration: float,
    ):
        self.arguments = (text, voice_style, lang)
        return np.ones((1, 441), dtype=np.float32) * 0.5, np.array([0.01])


class AdapterTests(unittest.TestCase):
    def test_supertonic_maps_marin_and_resamples_to_pcm16le(self) -> None:
        model = FakeSupertonic()
        speaker = SupertonicSpeaker({"marin": "F1"}, tts=model)

        result = speaker.synthesize("Hello", "marin")

        self.assertEqual(model.voice_name, "F1")
        self.assertEqual(model.arguments, ("Hello", "F1", "na"))
        self.assertEqual(len(result), 240 * 2)
        self.assertAlmostEqual(struct.unpack_from("<h", result)[0], 16384, delta=2)

    def test_resampler_attenuates_frequencies_above_24khz_nyquist(self) -> None:
        time = np.arange(44_100) / 44_100
        waveform = 0.5 * np.sin(2 * np.pi * 15_000 * time)

        pcm = _resample_pcm16le(waveform)
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)

        self.assertEqual(samples.size, 24_000)
        self.assertLess(np.sqrt(np.mean(samples**2)), 200)

    def test_supertonic_stops_between_synthesis_chunks_when_cancelled(self) -> None:
        cancelled = threading.Event()

        class CancellingSupertonic(FakeSupertonic):
            calls = 0

            def synthesize(self, *args, **kwargs):
                self.calls += 1
                cancelled.set()
                return super().synthesize(*args, **kwargs)

        model = CancellingSupertonic()
        speaker = SupertonicSpeaker(tts=model)

        result = speaker.synthesize("First sentence. " + "next " * 80, "marin", cancelled)

        self.assertTrue(result)
        self.assertEqual(model.calls, 1)

    def test_resampler_changes_24khz_to_16khz(self) -> None:
        source = struct.pack("<3h", 0, 1000, 2000)

        result = resample_pcm16le(source, 24_000, 16_000)

        self.assertEqual(len(result), 4)
        self.assertEqual(struct.unpack("<2h", result), (0, 1500))

    def test_vad_emits_boundaries_and_honors_mute(self) -> None:
        with tempfile.NamedTemporaryFile() as model_file:
            scores = iter([0.9] + [0.1] * 22)
            vad = SileroVAD(model_file.name, pause_ms=700, infer=lambda samples: next(scores))
            chunk = b"\x00\x00" * 768

            events = vad.append_audio(chunk * 23)
            vad.set_muted(True)
            self.assertEqual(vad.append_audio(chunk), [])

        self.assertEqual([event.type for event in events], ["speech_started", "speech_stopped"])
        self.assertEqual(events[0].offset_ms, 0)

    def test_faster_whisper_returns_text_and_session_timestamps(self) -> None:
        with tempfile.NamedTemporaryFile() as model_file:
            model = FakeModel()
            asr = FasterWhisperASR(model_file.name, model=model)
            result = asr.transcribe(b"\x00\x00" * 24_000, start_ms=250)

        self.assertEqual(result.text, "hello local model")
        self.assertEqual((result.start_ms, result.end_ms), (250, 1250))
        self.assertEqual(model.language, "en")

    def test_faster_whisper_receives_16khz_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            (checkpoint / "tokenizer.json").touch()
            model = FakeModel()
            asr = FasterWhisperASR(str(checkpoint), model=model)

            asr.transcribe(b"\x00\x00" * 24_000, start_ms=0)

        self.assertEqual(model.last_sample_count, 16_000)

    def test_faster_whisper_rejects_incomplete_local_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "tokenizer.json"):
                FasterWhisperASR(directory)

    def test_adapters_reject_odd_pcm16_data(self) -> None:
        with tempfile.NamedTemporaryFile() as model_file:
            vad = SileroVAD(model_file.name, infer=lambda samples: 0.0)
            asr = FasterWhisperASR(model_file.name, model=FakeModel())

            with self.assertRaises(ValueError):
                vad.append_audio(b"odd")
            with self.assertRaises(ValueError):
                asr.transcribe(b"odd", start_ms=0)

    def test_muted_audio_advances_offsets_and_resets_speech(self) -> None:
        with tempfile.NamedTemporaryFile() as model_file:
            vad = SileroVAD(model_file.name, infer=lambda samples: 0.9)
            chunk = b"\x00\x00" * 768
            self.assertEqual(vad.append_audio(chunk)[0].offset_ms, 0)
            vad.set_muted(True)
            vad.append_audio(chunk)
            vad.set_muted(False)
            self.assertEqual(vad.append_audio(chunk)[0].offset_ms, 64)


if __name__ == "__main__":
    unittest.main()
