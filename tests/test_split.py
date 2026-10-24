import numpy as np

from steelvision.data.split import SplitConfig, duplicate_groups, split_groups


def test_duplicate_groups_union_find():
    a = np.zeros((1, 8), dtype=bool)
    b = np.ones((1, 8), dtype=bool)  # distance 8 from a
    c = a.copy()  # exact duplicate of a
    groups = duplicate_groups(np.concatenate([a, b, c]), max_distance=2)
    members = sorted(sorted(g) for g in groups)
    assert members == [[0, 2], [1]]


def test_split_groups_balances_splits_and_keeps_groups_atomic():
    groups = [[i] for i in range(100)]
    class_of = [{"a"} if i % 2 == 0 else {"b"} for i in range(100)]
    cfg = SplitConfig(ratios={"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)

    assignment = split_groups(groups, class_of, cfg)
    sizes = {s: len(v) for s, v in assignment.items()}

    assert sum(sizes.values()) == 100
    assert sizes["train"] == max(sizes.values())
    assert 60 <= sizes["train"] <= 80
    assert 10 <= sizes["val"] <= 20
    assert 10 <= sizes["test"] <= 20
    assigned = [gi for idxs in assignment.values() for gi in idxs]
    assert sorted(assigned) == list(range(100))


def test_split_groups_keeps_duplicates_together():
    # One group of three near-duplicates must land in a single split.
    groups = [[0, 1, 2]] + [[i] for i in range(3, 20)]
    class_of = [{"a"} for _ in groups]
    cfg = SplitConfig(ratios={"train": 0.6, "val": 0.2, "test": 0.2}, seed=1)

    assignment = split_groups(groups, class_of, cfg)
    img_split = {
        image: split for split, gidxs in assignment.items() for gi in gidxs for image in groups[gi]
    }
    assert img_split[0] == img_split[1] == img_split[2]
