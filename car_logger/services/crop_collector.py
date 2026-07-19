"""Per-track crop collection for the multi-frame vote (Stage B spec,
Variant 1): crop #1 at confirmation, then up to reads_per_track-1 more
from later frames spaced spacing_s apart — consecutive frames are nearly
identical and would fail identically; spacing decorrelates the reads.
Hands the crop list to on_complete when full or when the track dies.

State is one bounded dict keyed by track_id, cleaned on completion and
on track death — it can never outgrow the tracker's own track list."""

import time

from car_logger.services.cropping import crop_to_jpeg


class _Collection(object):
    def __init__(self, event_id, first_crop, taken_at):
        self.event_id = event_id
        self.crops = [first_crop]
        self.last_taken_at = taken_at


class CropCollector(object):
    def __init__(self, on_complete, reads_per_track=3, spacing_s=0.4,
                 crop_fn=None, now=None):
        self._on_complete = on_complete
        self._reads_per_track = reads_per_track
        self._spacing_s = spacing_s
        self._crop_fn = crop_fn if crop_fn is not None else crop_to_jpeg
        self._now = now if now is not None else time.monotonic
        self._pending = {}

    def start(self, track_id, event_id, box, frame):
        """Take crop #1 at confirmation; register the collection.

        Completes immediately when the config asks for a single read."""
        crop = self._crop_fn(frame, box)
        if self._reads_per_track <= 1:
            self._on_complete(event_id, [crop])
            return
        self._pending[track_id] = _Collection(event_id, crop, self._now())

    def tick(self, live_tracks, frame):
        """Called once per pipeline tick with the tracker's live tracks."""
        live = dict((t.track_id, t) for t in live_tracks)
        for track_id in list(self._pending):
            collection = self._pending[track_id]
            track = live.get(track_id)
            if track is None:
                # track died — vote with what we have (student decision)
                del self._pending[track_id]
                self._on_complete(collection.event_id, collection.crops)
                continue
            if track.missed > 0:
                continue  # box is stale; cropping now would frame empty road
            if self._now() - collection.last_taken_at < self._spacing_s:
                continue
            collection.crops.append(self._crop_fn(frame, track.box))
            collection.last_taken_at = self._now()
            if len(collection.crops) >= self._reads_per_track:
                del self._pending[track_id]
                self._on_complete(collection.event_id, collection.crops)

    def drain(self):
        """Shutdown path: flush partial collections so their events still
        get a result (they become 'skipped' via the worker's own drain)."""
        for track_id in list(self._pending):
            collection = self._pending.pop(track_id)
            self._on_complete(collection.event_id, collection.crops)
