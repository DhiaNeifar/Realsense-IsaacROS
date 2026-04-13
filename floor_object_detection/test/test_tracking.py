from dataclasses import dataclass

from floor_object_detection.tracking import TemporalDetectionTracker


@dataclass
class DummyCandidate:
    bbox: tuple[int, int, int, int]
    center_pixel: tuple[int, int]
    center_xyz: tuple[float, float, float] | None
    distance_m: float
    median_height_m: float
    width_m: float | None
    confidence: float


def make_candidate(x: int, y: int, z: float) -> DummyCandidate:
    return DummyCandidate(
        bbox=(x, y, 100, 80),
        center_pixel=(x + 50, y + 40),
        center_xyz=(0.1, 0.2, z),
        distance_m=z,
        median_height_m=0.12,
        width_m=0.25,
        confidence=0.80,
    )


def test_tracker_confirms_and_smooths_candidate():
    tracker = TemporalDetectionTracker(
        min_confirmed_frames=2,
        max_missed_frames=2,
        bbox_smoothing_alpha=0.5,
        center_smoothing_alpha=0.5,
        max_center_jump_px=100.0,
        max_depth_jump_m=0.5,
    )

    assert tracker.update(make_candidate(10, 20, 1.00)) is None
    tracked = tracker.update(make_candidate(14, 24, 1.05))

    assert tracked is not None
    assert tracked.track_id == 1
    assert tracked.stale is False
    assert tracked.distance_m > 1.00
    assert tracked.distance_m < 1.05


def test_tracker_bridges_short_dropout_then_expires():
    tracker = TemporalDetectionTracker(
        min_confirmed_frames=1,
        max_missed_frames=1,
        bbox_smoothing_alpha=0.5,
        center_smoothing_alpha=0.5,
        max_center_jump_px=100.0,
        max_depth_jump_m=0.5,
    )

    tracked = tracker.update(make_candidate(10, 20, 1.00))
    assert tracked is not None

    stale = tracker.update(None)
    assert stale is not None
    assert stale.stale is True

    expired = tracker.update(None)
    assert expired is None
