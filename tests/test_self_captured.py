import pytest

from steelvision.data.self_captured import (
    compute_label_consistency,
    panel_position_groups,
    validate_manifest,
)


def test_validate_manifest_and_grouping(tmp_path):
    csv_path = tmp_path / "manifest.csv"
    rows = ["file,session,panel,position,lighting,is_defective,label_status,notes"]
    # 6 panels x 3 lighting = 18 rows (9 OK, 9 defective -> 50% OK ratio)
    idx = 1
    for panel_idx in range(1, 7):
        is_def = "ok" if panel_idx <= 3 else "defective"
        for lt in ("A", "B", "C"):
            rows.append(
                f"IMG_{idx:04d}.jpg,2026-10-26-S1,P{panel_idx:02d},pos1,{lt},{is_def},qc-passed,"
            )
            idx += 1
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    rep = validate_manifest(
        csv_path,
        min_total_qc_passed=18,
        min_ok_ratio=0.40,
        min_ok_per_lighting=3,
    )
    assert rep.errors == []
    assert rep.n_rows == 18
    assert rep.n_qc_passed == 18
    assert rep.ok_ratio == pytest.approx(0.50)
    assert rep.meets_stopping_criteria is True

    groups = panel_position_groups(csv_path)
    assert len(groups) == 6
    assert all(len(files) == 3 for files in groups.values())


def test_label_consistency_passes_and_fails(tmp_path):
    dir_a = tmp_path / "pass_a"
    dir_b = tmp_path / "pass_b"
    dir_a.mkdir()
    dir_b.mkdir()

    # Image 1: OK in both passes (empty label file)
    (dir_a / "img1.txt").write_text("", encoding="utf-8")
    (dir_b / "img1.txt").write_text("", encoding="utf-8")

    # Image 2: closely matching boxes of class 0
    (dir_a / "img2.txt").write_text("0 0.50 0.50 0.40 0.40\n", encoding="utf-8")
    (dir_b / "img2.txt").write_text("0 0.51 0.50 0.40 0.39\n", encoding="utf-8")

    rep = compute_label_consistency(dir_a, dir_b)
    assert rep.n_images == 2
    assert rep.image_level_f1 == pytest.approx(1.0)
    assert rep.mean_box_iou > 0.85
    assert rep.passes_guide_targets is True
