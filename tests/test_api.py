"""Смоук-тесты: модель грузится, REST/WS отвечают, правила работают.

Модели грузятся один раз на сессию (медленно, ~1-2с), поэтому фикстура session-scoped.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import plate_rules


@pytest.fixture(scope="session")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _fake_plate(text: str = "01A123BC", w: int = 260, h: int = 60) -> np.ndarray:
    img = np.full((h, w, 3), 235, np.uint8)
    cv2.putText(img, text, (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    return img


# ------------------------------------------------------------------ правила
def test_normalize_and_formats():
    assert plate_rules.normalize(" 01 a 123-bc ") == "01A123BC"
    m = plate_rules.match("01A123BC")
    assert m.valid and m.format == "uz_private" and m.region_ok
    assert m.groups == ("01", "A", "123", "BC")

    m2 = plate_rules.match("01123ABC")
    assert m2.valid and m2.format == "uz_legal" and m2.plate_class == "legal"

    assert not plate_rules.match("ZZZ").valid


def test_repair_confusions():
    # O вместо 0 в позиции региона, S вместо 5 в цифрах
    m = plate_rules.repair("O1A1S3BC")
    assert m.valid and m.text == "01A153BC" and m.corrected


def test_dedup_repairs_ctc_duplicates():
    """CTC дублирует символы на широких пластинах — повторы снимаются по формату."""
    assert plate_rules.repair("01M00018461").text == "01M018461"  # вывод head_ctype
    assert plate_rules.repair("01M0021672").text == "01M021672"
    assert plate_rules.repair("01M00018461").corrected

    # законные повторы ломать нельзя
    m = plate_rules.match("01A007US")
    assert m.valid and m.text == "01A007US"
    assert plate_rules.repair("01A007US").text == "01A007US"


def test_pick_best_repairs_late_candidate():
    """Кандидат от второй головы приходит последним — его тоже надо чинить."""
    alts = [("XX", 0.9), ("01M00018461", 0.8)]
    text, _, m = plate_rules.pick_best(alts)
    assert text == "01M018461" and m.valid


def test_pick_best_prefers_valid_format():
    alts = [("01A1Z3BC", 0.90), ("01A123BC", 0.80)]
    text, conf, m = plate_rules.pick_best(alts)
    assert text == "01A123BC" and m.valid and conf == 0.80


def test_trailer_format_region_is_last():
    """4169 AA 70 — квадратный номер прицепа: регион в КОНЦЕ (сверено по platesmania)."""
    m = plate_rules.match("4169AA70")
    assert m.valid and m.format == "uz_trailer" and m.plate_class == "trailer"
    assert m.groups == ("4169", "AA", "70")
    assert m.region_ok, "код региона проверяется в последней группе, а не в первой"
    assert plate_rules.match("9999AA99").region_ok is False


def test_colored_9symbol_formats():
    """Цветные номера УЗ: регион + буква серии + 6 цифр (зелёные и жёлтые)."""
    green = plate_rules.match("01M018461")  # совместное или иностранное предприятие
    assert green.valid and green.format == "uz_joint" and green.plate_class == "joint"
    assert green.groups == ("01", "M", "018461")

    yellow = plate_rules.match("01H018461")  # транспорт иностранцев, жёлтый фон
    assert yellow.format == "uz_foreign" and yellow.plate_class == "foreign"

    other = plate_rules.match("01Z018461")  # другая серия того же скелета
    assert other.format == "uz_series6" and other.plate_class == "special"

    # не должен перебивать формат мототранспорта (7 символов, как на сайте)
    assert plate_rules.match("179FA01").format == "uz_moto"
    # «01 A 1234» был придуман без данных и на сайте не встречается
    assert plate_rules.match("01A1234").format is None


def test_letter_i_excluded():
    """В номерах УЗ нет 'I' — это видно и по весам head_anpr."""
    assert "I" not in plate_rules.LETTERS
    assert not plate_rules.match("01I123BC").valid


def test_pad_to_ratio_keeps_content():
    from app.core.recognizer import PLATE_AR, pad_to_ratio

    line = np.full((40, 60, 3), 200, np.uint8)
    line[10:30, 5:55] = 20
    out = pad_to_ratio(line)
    assert out.shape[0] == 40
    assert out.shape[1] == int(40 * PLATE_AR)
    assert (out[:, :60] == line).all(), "исходный кроп должен остаться слева без изменений"
    assert pad_to_ratio(np.zeros((10, 200, 3), np.uint8)).shape[1] == 200  # уже широкий


def test_layout_variants_single_line_first():
    from app.core.recognizer import layout_variants

    wide = np.full((30, 200, 3), 255, np.uint8)
    assert len(layout_variants(wide)) == 1, "широкий кроп не надо резать на строки"

    square = np.full((80, 100, 3), 255, np.uint8)
    square[10:30, 10:90] = 0  # верхняя строка
    square[50:70, 10:90] = 0  # нижняя строка
    variants = layout_variants(square)
    assert len(variants) == 2
    assert len(variants[0]) == 1, "первый вариант — всегда одна строка (на него откат)"
    assert len(variants[1]) == 2


def test_pick_variant_falls_back_to_single_line():
    from app.core.recognizer import RecogResult, _pick_variant

    single = (RecogResult(text="80110A", conf=0.5, chars=[]), [])
    двухстрочный_мусор = (RecogResult(text="2377385AJ1A", conf=0.9, chars=[]), [])
    valid = lambda t: plate_rules.match(t).valid  # noqa: E731

    # ни один вариант не валиден -> берём однострочный, даже если у него conf ниже
    assert _pick_variant([single, двухстрочный_мусор], valid)[0].text == "80110A"

    # если вторая раскладка дала валидный формат — она и побеждает
    good = (RecogResult(text="2602BA10", conf=0.6, chars=[]), [])
    assert _pick_variant([single, good], valid)[0].text == "2602BA10"


def test_classify_color():
    yellow = np.zeros((60, 200, 3), np.uint8)
    yellow[:] = (30, 210, 240)  # BGR ~ жёлтый
    assert plate_rules.classify_color(yellow)[0] == "yellow"

    green = np.zeros((60, 200, 3), np.uint8)
    green[:] = (60, 160, 40)
    assert plate_rules.classify_color(green)[0] == "green"

    white = np.full((60, 200, 3), 240, np.uint8)
    assert plate_rules.classify_color(white)[0] == "white"


# --------------------------------------------------------------------- трекер
def test_tracker_votes_and_ids():
    from app.core.tracker import PlateTracker

    tr = PlateTracker(iou_threshold=0.2, max_age=2)
    ids1 = tr.update([(10, 10, 110, 40)], ts=1.0)
    ids2 = tr.update([(12, 11, 112, 41)], ts=1.1)
    assert ids1 == ids2, "смещённый бокс должен остаться тем же треком"

    t = tr.get(ids1[0])
    t.vote("01A123BC", 0.9, True)
    t.vote("01A123BC", 0.8, True)
    t.vote("01A123B", 0.4, False)
    assert t.stable_text == "01A123BC"
    assert t.stable_votes == 2

    for i in range(4):  # трек должен умереть по max_age
        tr.update([], ts=2.0 + i)
    assert tr.get(ids1[0]) is None


# ------------------------------------------------------------------ REST / WS
def test_healthz(client):
    d = client.get("/healthz").json()
    assert d["status"] == "ok"
    assert d["num_class"] == 95
    assert set(d["heads"]) == {"anpr", "ctype"}
    assert d["charset_len"] == 94, "94 символа + blank = 95 классов"


def test_recognize_frame(client):
    frame = np.full((480, 640, 3), 120, np.uint8)
    r = client.post("/v1/recognize", files={"file": ("f.jpg", _jpeg(frame), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert "plates" in d and "timings" in d
    assert d["frame_size"] == [640, 480]


def test_recognize_plate_crop(client):
    r = client.post(
        "/v1/recognize/plate", files={"file": ("p.jpg", _jpeg(_fake_plate()), "image/jpeg")}
    )
    assert r.status_code == 200
    d = r.json()
    # charset верный => модель выдаёт только цифры и заглавные латинские
    assert d["raw_text"] == d["raw_text"].upper()
    assert all(c.isdigit() or c.isupper() for c in d["raw_text"])
    assert 0.0 <= d["raw_conf"] <= 1.0
    assert isinstance(d["alternatives"], list)


def test_bad_image_rejected(client):
    r = client.post("/v1/recognize", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_ws_recognize_roundtrip(client):
    frame = np.full((360, 640, 3), 100, np.uint8)
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_json({"type": "ping", "ts": 1})
        assert ws.receive_json()["type"] == "pong"

        ws.send_bytes(_jpeg(frame))
        msg = ws.receive_json()
        assert msg["type"] == "result"
        assert msg["frame_id"] == 1
        assert "timings" in msg and "plates" in msg

        ws.send_json({"type": "config", "overlay": True})
        assert ws.receive_json()["type"] == "config"


def test_ws_rejects_garbage(client):
    with client.websocket_connect("/ws/recognize") as ws:
        ws.send_bytes(b"garbage")
        msg = ws.receive_json()
        assert msg["type"] == "error"


def test_streams_lifecycle(client):
    assert client.get("/v1/streams").json() == []
    r = client.post("/v1/streams", json={"id": "nope", "source": "/does/not/exist.mp4"})
    assert r.status_code == 201  # поток создаётся, но с ошибкой подключения
    body = r.json()
    assert body["id"] == "nope" and not body["connected"]

    dup = client.post("/v1/streams", json={"id": "nope", "source": "x"})
    assert dup.status_code == 409

    assert client.delete("/v1/streams/nope").status_code == 200
    assert client.delete("/v1/streams/nope").status_code == 404


def test_events_endpoint(client):
    assert isinstance(client.get("/v1/events?limit=5").json(), list)
    assert "events" in client.get("/v1/stats").json()


# ------------------------------------------------------- установка на чистой машине


def test_fetch_yolo_default_matches_config():
    """`python -m tools.fetch_yolo` без аргументов обязан скачать то, что ждёт конфиг.

    Эти два значения жили в разных файлах и разъехались: скрипт качал
    `lpr-yolov8n-plate.pt`, а конфиг искал `yolo11n-plate.pt`. Локально это не
    всплывало — старые веса уже лежали, — зато свежая установка по README падала
    с FileNotFoundError на старте.
    """
    from app.config import Settings
    from tools.fetch_yolo import DEFAULT_MODELS, weights_name

    expected = Settings().detector_weights.name
    downloaded = {weights_name(m) for m in DEFAULT_MODELS}
    assert expected in downloaded, (
        f"конфиг ждёт {expected}, а fetch_yolo по умолчанию качает {sorted(downloaded)}"
    )


def _two_line_plate():
    """Синтетическая пластина прицепа: «8815» сверху, «BA 01» снизу."""
    import cv2
    import numpy as np

    img = np.full((120, 170, 3), 235, np.uint8)
    cv2.putText(img, "8815", (18, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
    cv2.putText(img, "BA 01", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
    return img


def test_unfold_lines_puts_bottom_line_right_of_top():
    from app.core.recognizer import unfold_lines

    crop = _two_line_plate()
    flat = unfold_lines(crop)
    assert flat is not crop
    h, w = flat.shape[:2]
    assert h < crop.shape[0] and w > crop.shape[1], "развёртка должна стать низкой и широкой"
    assert w / h > 2.6, "по пропорциям развёртка обязана походить на однострочную пластину"


def test_unfold_lines_leaves_single_line_plate_alone():
    import numpy as np

    from app.core.recognizer import unfold_lines, unfold_variants

    wide = np.full((30, 130, 3), 235, np.uint8)  # однострочная пластина, ar 4.3
    assert unfold_lines(wide) is wide
    assert unfold_variants(wide) == [[wide]]


def test_vehicle_model_canonical_names():
    from app.core.vehicle_models import canonical

    # одна машина под разными именами — один класс
    assert canonical("Chevrolet Cobalt") == canonical("Ravon R4") == "Cobalt"
    assert canonical("Chevrolet Lacetti") == canonical("Ravon Gentra") == "Lacetti"
    assert canonical("Daewoo Nexia") == canonical("Ravon Nexia R3") == "Nexia"
    assert canonical("Chevrolet Malibu") == "Malibu"
    assert canonical("Lada (VAZ) 2106") == canonical("Lada (VAZ) 2101") == "Жигули"
    assert canonical("Lada (VAZ) Largus") == "Lada Largus"
    assert canonical("Deepal (Shenlan) S7 (S07)") == "Deepal S7"
    assert canonical("BYD Song") == "BYD Song"
    assert canonical("Krone Profi Liner", trailer=True) == "Krone"
    assert canonical("License plate without vehicle") == ""
    assert canonical("") == ""


def test_pick_vehicle_takes_car_that_carries_the_plate():
    from app.core.vehicle import pick_vehicle

    truck = [0, 0, 1000, 800]  # тягач во весь кадр
    car = [600, 400, 900, 700]  # легковушка поверх него
    plate_on_car = [700, 600, 780, 620]
    # пластина внутри обеих рамок — берём самую тесную, то есть легковушку
    assert pick_vehicle([truck, car], plate_on_car) == car
    # центр пластины вне машин — модель не называем, а не гадаем по соседней
    assert pick_vehicle([car], [10, 10, 50, 30]) is None
    assert pick_vehicle([car], None) is None


def test_trailer_plate_looks_only_at_trucks():
    from app.core.vehicle import TRUCK, candidates, pick_vehicle

    car, trailer = [100, 300, 400, 600], [0, 0, 1000, 700]
    cars = [(car, 2), (trailer, TRUCK)]
    plate = [200, 500, 260, 540]  # центр внутри обеих рамок
    assert pick_vehicle(candidates(cars, trailer=False), plate) == car
    # номер прицепа: легковушка перед ним — не его машина
    assert pick_vehicle(candidates(cars, trailer=True), plate) == trailer


def test_detector_benchmark_matching_is_one_to_one():
    from tools.eval_detector import match

    gt = [[0, 0, 100, 40], [200, 0, 260, 50]]
    # две рамки на одну пластину засчитывают её один раз, вторая — ложная
    assert match(gt, [[2, 1, 101, 41], [0, 0, 98, 39]]) == {0}
    assert match(gt, [[202, 2, 259, 49], [0, 0, 100, 40]]) == {0, 1}
    assert match(gt, [[500, 500, 600, 540]]) == set()  # мимо — не попадание


def test_formats_measured_on_site_categories():
    from app.core import plate_rules

    # номера взяты из категорий сайта: мотоциклы, спецтехника, высшие органы
    moto = plate_rules.match("179FA01")
    assert moto.format == "uz_moto" and moto.groups[2] == "01" and moto.region_ok
    assert plate_rules.match("429NA10").format == "uz_moto"  # спецтехника — та же раскладка
    gov = plate_rules.match("PAA104")
    assert gov.format == "uz_gov" and gov.region_ok  # у высших органов региона нет
    assert plate_rules.match("01M016493").plate_class == "joint"


def test_moto_unfold_puts_region_after_letters():
    import cv2
    import numpy as np

    from app.core.recognizer import moto_unfold

    # сетка 2x2: «197» слева-сверху, «DA» слева-снизу, «01» справа-сверху
    img = np.full((120, 130, 3), 235, np.uint8)
    cv2.putText(img, "197", (6, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(img, "DA", (6, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(img, "01", (92, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    flat = moto_unfold(img)
    assert flat is not img
    h, w = flat.shape[:2]
    assert w / h > 2.5, "развёртка мото обязана стать низкой и широкой, как строка"


def test_moto_unfold_leaves_wide_plate_alone():
    import numpy as np

    from app.core.recognizer import moto_unfold

    wide = np.full((30, 130, 3), 235, np.uint8)  # однострочная, не сетка
    assert moto_unfold(wide) is wide
