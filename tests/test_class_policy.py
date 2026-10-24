from steelvision.data.prepare import apply_class_policy
from steelvision.data.voc_to_yolo import Sample, YoloBox


def _sample(*class_names):
    box = dict(x_center=0.5, y_center=0.5, width=0.1, height=0.1)
    return Sample(
        image=None,  # not used by the class policy
        xml=None,
        boxes=[YoloBox(class_name=c, **box) for c in class_names],
    )


def test_apply_class_policy_renames_and_drops():
    samples = [
        _sample("1_chongkong", "10_yaozhed"),
        _sample("d"),
        _sample("9_zhehen"),
    ]
    rename = {"1_chongkong": "punching", "10_yaozhed": "waist_folding", "9_zhehen": "crease"}
    observed, dropped = apply_class_policy(samples, rename, {"d"})

    assert observed == {"punching": 1, "waist_folding": 1, "crease": 1}
    assert dropped == {"d": 1}
    assert [b.class_name for b in samples[0].boxes] == ["punching", "waist_folding"]
    assert samples[1].boxes == []


def test_apply_class_policy_noop_without_config():
    samples = [_sample("scratch")]
    observed, dropped = apply_class_policy(samples, {}, set())
    assert observed == {"scratch": 1}
    assert dropped == {}
