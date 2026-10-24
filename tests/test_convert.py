import textwrap

from steelvision.data.voc_to_yolo import ParseReport, normalize_class, parse_voc_xml

VOC = textwrap.dedent(
    """\
    <annotation>
      <filename>img1.jpg</filename>
      <size><width>200</width><height>100</height><depth>3</depth></size>
      <object>
        <name>Scratch</name>
        <bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>70</ymax></bndbox>
      </object>
      <object>
        <name>scratch</name>
        <bndbox><xmin>150</xmin><ymin>0</ymin><xmax>250</xmax><ymax>50</ymax></bndbox>
      </object>
      <object>
        <name>dirt</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>0.5</xmax><ymax>0.5</ymax></bndbox>
      </object>
    </annotation>
    """
)


def test_normalize_class():
    assert normalize_class(" Rolled-in Scale ") == "rolled_in_scale"


def test_parse_voc_boxes_and_clipping(tmp_path):
    xml = tmp_path / "img1.xml"
    xml.write_text(VOC, encoding="utf-8")
    report = ParseReport()
    boxes = parse_voc_xml(xml, report)

    assert len(boxes) == 2
    first, second = boxes
    assert first.class_name == "scratch"
    assert abs(first.x_center - 0.30) < 1e-6
    assert abs(first.y_center - 0.45) < 1e-6
    assert abs(first.width - 0.50) < 1e-6
    assert abs(first.height - 0.50) < 1e-6
    # xmax=250 is clipped to the 200px image width.
    assert abs(second.width - 0.25) < 1e-6
    assert abs(second.x_center - 0.875) < 1e-6

    assert report.n_boxes == 2
    assert report.n_degenerate == 1
    assert report.class_counts == {"scratch": 2}
