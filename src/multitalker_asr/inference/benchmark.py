import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from loguru import logger

from ..configs.streaming import ChunkPreset, preset_chunk_ms, preset_to_att_context
from .cache_aware import ChunkSwitcher


@dataclass
class LatencyMeasurement:
    preset: ChunkPreset
    chunk_ms: float
    samples_seconds: float
    wall_clock_seconds: float
    real_time_factor: float
    final_token_latency_ms: float

    @property
    def faster_than_realtime(self) -> bool:
        return self.real_time_factor < 1.0


@dataclass
class LatencyReport:
    measurements: List[LatencyMeasurement] = field(default_factory=list)

    def add(self, measurement: LatencyMeasurement) -> None:
        self.measurements.append(measurement)

    def median_rtf(self) -> Optional[float]:
        if not self.measurements:
            return None
        return statistics.median(m.real_time_factor for m in self.measurements)

    def as_dict(self) -> List[Dict]:
        return [
            {
                "preset": m.preset.value,
                "chunk_ms": m.chunk_ms,
                "wall_clock_seconds": m.wall_clock_seconds,
                "real_time_factor": m.real_time_factor,
                "final_token_latency_ms": m.final_token_latency_ms,
                "faster_than_realtime": m.faster_than_realtime,
            }
            for m in self.measurements
        ]


class LatencyBenchmark:
    """Sweep all chunk presets on a given audio source, measure RTF."""

    def __init__(
        self,
        switcher: ChunkSwitcher,
        transcribe_fn: Callable[[], str],
        audio_seconds: float,
        warmup_runs: int = 1,
        measurement_runs: int = 3,
    ):
        if audio_seconds <= 0:
            raise ValueError(f"audio_seconds must be > 0, got {audio_seconds}")
        if measurement_runs <= 0:
            raise ValueError(
                f"measurement_runs must be > 0, got {measurement_runs}"
            )
        self._switcher = switcher
        self._transcribe_fn = transcribe_fn
        self._audio_seconds = audio_seconds
        self._warmup_runs = max(0, warmup_runs)
        self._measurement_runs = measurement_runs

    def run(self, presets: Optional[List[ChunkPreset]] = None) -> LatencyReport:
        presets = presets or list(ChunkPreset)
        report = LatencyReport()
        for preset in presets:
            try:
                measurement = self._measure_preset(preset)
            except Exception as exc:
                logger.warning(f"Benchmark failed for {preset.value}: {exc}")
                continue
            report.add(measurement)
            logger.info(
                f"preset={preset.value} chunk_ms={measurement.chunk_ms:.0f} "
                f"rtf={measurement.real_time_factor:.3f}"
            )
        return report

    def _measure_preset(self, preset: ChunkPreset) -> LatencyMeasurement:
        self._switcher.apply(preset)
        for _ in range(self._warmup_runs):
            self._transcribe_fn()

        wall_times: List[float] = []
        last_token_times: List[float] = []
        for _ in range(self._measurement_runs):
            start = time.perf_counter()
            self._transcribe_fn()
            elapsed = time.perf_counter() - start
            wall_times.append(elapsed)
            last_token_times.append(elapsed * 1000.0)

        median_wall = statistics.median(wall_times)
        rtf = median_wall / self._audio_seconds
        return LatencyMeasurement(
            preset=preset,
            chunk_ms=preset_chunk_ms(preset),
            samples_seconds=self._audio_seconds,
            wall_clock_seconds=median_wall,
            real_time_factor=rtf,
            final_token_latency_ms=statistics.median(last_token_times),
        )
