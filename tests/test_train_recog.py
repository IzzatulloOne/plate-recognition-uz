"""Обучение распознавателя: то, что проверяется без модели и GPU."""

import json

import numpy as np

from tools import train_recog as tr
from tools.build_dataset import _lev


def test_encode_matches_ctc_layout():
    targets, lengths = tr.encode(["01A", "9"], "0123456789ABC")
    # класс 0 — blank, поэтому символ с позицией i в алфавите получает i + 1
    assert targets.tolist() == [1, 2, 11, 10]
    assert lengths.tolist() == [3, 1]


def test_eval_photos_never_reach_training(tmp_path, monkeypatch):
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"111": {"gt": "01A123BC"}, "222": {"gt": "4169AA70"}}))
    monkeypatch.setattr(tr, "EVAL_INDEX", ev)
    (tmp_path / "gt.txt").write_text(
        "images/111_01A123BC.jpg\t01A123BC\n"   # то же фото, что в эталоне
        "images/333_4169AA70.jpg\t4169AA70\n"   # другое фото той же машины
        "images/444_8815BA01.jpg\t8815BA01\n"   # чужой номер — остаётся
    )
    samples, skipped = tr.load_samples(tmp_path)
    assert [s["text"] for s in samples] == ["8815BA01"]
    assert samples[0]["cls"] == "trailer"
    assert skipped["в эталоне eval"] == 2


def test_augment_keeps_shape_and_type():
    img = np.random.default_rng(0).integers(0, 255, (40, 160, 3), dtype=np.uint8)
    for seed in range(20):
        out = tr.augment(img, np.random.default_rng(seed))
        assert out.shape == img.shape and out.dtype == np.uint8


def test_levenshtein():
    assert _lev("4169", "4169") == 0
    assert _lev("4189", "4169") == 1  # одна ошибка чтения — бокс засчитывается
    assert _lev("4196", "4169") == 2  # перестановка — это две замены
    assert _lev("40416", "4169") == 3  # так модель читала верх прицепа до дообучения
    assert _lev("", "70") == 2


def test_split_by_plate_text_keeps_same_car_together():
    from collections import Counter

    from tools.build_dataset import _split

    # одна машина на разных фото — одна и та же часть, иначе эталон «подсмотрен»
    assert len({_split("4169AA70") for _ in range(5)}) == 1
    plates = [f"{i:04d}AA{r:02d}" for i in range(2000) for r in (10, 40)]
    share = Counter(_split(p) for p in plates)
    assert 0.08 < share["test"] / len(plates) < 0.12
    assert 0.08 < share["val"] / len(plates) < 0.12
