import pytest

from convert_openalpr_benchmarks import parse_annotation_line


def test_parse_annotation_line_takes_filename_and_last_field():
    line = "eu1.jpg\t396\t340\t203\t46\tM5XSX\n"
    assert parse_annotation_line(line) == ("eu1.jpg", "M5XSX")


def test_parse_annotation_line_rejects_garbage():
    with pytest.raises(ValueError):
        parse_annotation_line("not-an-annotation\n")
