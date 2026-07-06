from car_logger.services.geometry import iou


def test_identical_boxes_iou_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_disjoint_boxes_iou_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_half_overlap_iou():
    # two 10x10 boxes overlapping in a 10x5 strip:
    # intersection = 50, union = 100 + 100 - 50 = 150 -> 1/3
    result = iou((0, 0, 10, 10), (0, 5, 10, 15))
    assert abs(result - (1.0 / 3.0)) < 1e-9


def test_zero_area_box_is_zero():
    assert iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0
