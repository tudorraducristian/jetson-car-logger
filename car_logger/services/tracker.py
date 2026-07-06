"""Simple greedy IoU tracker — the deduplication brain of the pipeline.

STUDENT DECISIONS (tune on real footage in Task 5):
- iou_threshold = 0.3 : too low glues two nearby cars into one track; too
  high breaks the track of a fast-moving car and duplicates its events.
- max_missed    = 5   : too low, one detector flicker splits a car into two
  events; too high, two cars passing the same spot within a short interval
  get merged into a single event.
- min_hits      = 5   : too low, shadows and reflections get logged as cars;
  too high, a car crossing the frame quickly never gets logged.
"""

from car_logger.services.geometry import iou


class Track(object):
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.hits = 1
        self.missed = 0
        self.emitted = False

    def is_confirmed(self, min_hits):
        return self.hits >= min_hits


class IoUTracker(object):
    def __init__(self, iou_threshold=0.3, max_missed=5, min_hits=5):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks = []

    def update(self, boxes):
        """Feed one frame's detection boxes; return the live Track list."""
        unmatched = list(range(len(boxes)))

        # Greedy: match each existing track to its best still-free box.
        for track in self.tracks:
            best_score = self.iou_threshold
            best_j = -1
            for j in unmatched:
                score = iou(track.box, boxes[j])
                if score >= best_score:
                    best_score = score
                    best_j = j
            if best_j >= 0:
                track.box = boxes[best_j]
                track.hits += 1
                track.missed = 0
                unmatched.remove(best_j)
            else:
                track.missed += 1

        # Births: any detection that matched nothing starts a new track.
        for j in unmatched:
            self.tracks.append(Track(self._next_id, boxes[j]))
            self._next_id += 1

        # Deaths: drop tracks unseen for more than max_missed frames.
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return self.tracks

    def new_confirmed_tracks(self):
        """Tracks that just became confirmed and have not emitted yet.

        Marks them emitted so a track produces exactly one event."""
        ready = []
        for t in self.tracks:
            if t.is_confirmed(self.min_hits) and not t.emitted:
                t.emitted = True
                ready.append(t)
        return ready
