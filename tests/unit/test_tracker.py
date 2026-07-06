from car_logger.services.tracker import IoUTracker


def test_overlapping_box_keeps_same_track_id():
    t = IoUTracker(iou_threshold=0.3, max_missed=5, min_hits=5)
    t.update([(0, 0, 10, 10)])
    first_id = t.tracks[0].track_id
    # a slightly shifted box that still overlaps a lot -> same track
    t.update([(1, 1, 11, 11)])
    assert len(t.tracks) == 1
    assert t.tracks[0].track_id == first_id
    assert t.tracks[0].hits == 2


def test_non_overlapping_boxes_make_two_tracks():
    t = IoUTracker()
    t.update([(0, 0, 10, 10), (100, 100, 110, 110)])
    assert len(t.tracks) == 2
    assert t.tracks[0].track_id != t.tracks[1].track_id


def test_track_dies_after_max_missed_frames():
    t = IoUTracker(max_missed=2)
    t.update([(0, 0, 10, 10)])          # born
    t.update([])                        # missed 1
    t.update([])                        # missed 2
    assert len(t.tracks) == 1           # still alive at == max_missed
    t.update([])                        # missed 3 -> death
    assert t.tracks == []


def test_confirmed_only_after_min_hits_and_emitted_once():
    t = IoUTracker(min_hits=3)
    box = (0, 0, 10, 10)
    t.update([box])
    assert t.new_confirmed_tracks() == []   # 1 hit
    t.update([box])
    assert t.new_confirmed_tracks() == []   # 2 hits
    t.update([box])
    confirmed = t.new_confirmed_tracks()     # 3 hits -> confirmed
    assert len(confirmed) == 1
    # a track emits exactly once, not every subsequent frame:
    t.update([box])
    assert t.new_confirmed_tracks() == []
