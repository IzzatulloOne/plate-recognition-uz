"""Сбор через браузер без браузера: поддельная галерея с тем же пределом глубины."""

from urllib.parse import parse_qs, urlsplit

import pytest

from tools import fetch_browser as fb


class FakeGallery:
    """Ведёт себя как platesmania: глубже MAX_PAGES отдаёт последнюю страницу снова.

    Так и выглядел настоящий предел — не пустая страница, а повтор, поэтому
    прогон по прицепам упёрся в тысячу номеров из 4261.
    """

    PER_PAGE, MAX_PAGES = 10, 3

    def __init__(self, plates: list[tuple[str, str]]):
        self.plates = plates
        self.page_source = ""
        self.loads = 0
        self.prefixes: set[str] = set()

    def get(self, url: str) -> None:
        self.loads += 1
        q = parse_qs(urlsplit(url).query)
        prefix = q.get("nomer", [""])[0].rstrip("*")
        self.prefixes.add(prefix)
        hits = [p for p in self.plates if p[1].replace(" ", "").startswith(prefix)]
        if 0 < len(prefix) < 2:  # как настоящий сайт: короче двух символов не ищет
            hits = []
        pages = min(self.MAX_PAGES, -(-len(hits) // self.PER_PAGE))
        start = min(int(q.get("start", ["0"])[0]), max(pages - 1, 0))
        cards = "".join(
            f'<div class="panel panel-grey"><h4 class="text-center">Krone</h4>'
            f'<img src="https://img1.platesmania.com/261017/m/{pid}.jpg">'
            f'<img src="https://img1.platesmania.com/261017/inf/{pid}.png" alt="{gt}"></div>'
            for pid, gt in hits[start * self.PER_PAGE : (start + 1) * self.PER_PAGE]
        )
        self.page_source = f"<h1>License plates found <b>{len(hits)}</b></h1>{cards}"

    def get_cookies(self) -> list:
        return []

    def execute_script(self, _js: str) -> str:
        return "test-agent"


def test_page_url_and_counter():
    assert fb.page_url(1, ctype=6) == "https://platesmania.com/uz/gallery?ctype=6"
    assert fb.page_url(11, ctype=6) == "https://platesmania.com/uz/gallery.php?ctype=6&start=10"
    assert fb.found_total("License plates found <b>4.261</b>") == 4261
    assert fb.found_total("<html>капча</html>") is None


def test_fetch_slices_past_depth_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "DELAY", 0)
    monkeypatch.setattr(fb, "_ask", lambda msg: pytest.fail(f"разбудил человека: {msg}"))

    # 60 прицепов, а без нарезки сайт отдаёт 30. Фото заранее на диске — без сети.
    plates = [(str(1000 + d * 10 + i), f"{d}{i:03d} AA 70") for d in range(6) for i in range(10)]
    (tmp_path / "photos").mkdir()
    for pid, _ in plates:
        (tmp_path / "photos" / f"{pid}.jpg").write_bytes(b"jpg")

    site = FakeGallery(plates)
    items = fb.fetch(site, target=10_000, out_dir=tmp_path, ctype=6)

    assert len(items) == 60, "предел глубины не обойдён"
    assert {v["model"] for v in items.values()} == {"Krone"}
    # однозначные срезы сайт не ищет — первой ночью из-за них сбор вернул ноль
    assert all(len(m) != 1 for m in site.prefixes), sorted(site.prefixes)
    # Без детектора повторов скрипт листал бы повторы до max_pages=200;
    # проход по 100 двузначным срезам стоит по загрузке на пустой срез
    assert site.loads < 130, f"лишних загрузок страниц: {site.loads}"


def test_per_slice_spreads_collection_across_regions(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "DELAY", 0)
    monkeypatch.setattr(fb, "_ask", lambda msg: pytest.fail(f"разбудил человека: {msg}"))
    # три «региона» по 10 номеров; квота 4 на срез -> по 4 из каждого, а не 10+2 из первых
    plates = [(str(5000 + r * 100 + i), f"{r:02d}A{i:03d}BC")
              for r in (1, 40, 95) for i in range(10)]
    (tmp_path / "photos").mkdir()
    for pid, _ in plates:
        (tmp_path / "photos" / f"{pid}.jpg").write_bytes(b"jpg")

    items = fb.fetch(FakeGallery(plates), target=12, out_dir=tmp_path,
                     masks=[f"{i:02d}" for i in range(100)], per_slice=4)
    regions = sorted({v["gt"][:2] for v in items.values()})
    assert len(items) == 12 and regions == ["01", "40", "95"]


def test_unattended_never_waits_for_a_human(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "DELAY", 0)
    monkeypatch.setattr(fb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fb, "_ask", lambda msg: pytest.fail(f"ночью ждал ENTER: {msg}"))

    class Captcha(FakeGallery):
        def get(self, url: str) -> None:
            self.loads += 1
            self.page_source = "<html>Подтвердите, что вы человек</html>"

    site = Captcha([])
    items = fb.fetch(site, target=100, out_dir=tmp_path, ctype=6, unattended=True)
    # капчу не решаем: выждали, попробовали ещё раз и закончили этап сами
    assert items == {}
    assert site.loads <= 3


def test_shrink_keeps_small_photos_and_downsizes_big():
    import cv2
    import numpy as np

    big = cv2.imencode(".jpg", np.full((2400, 3200, 3), 120, np.uint8))[1].tobytes()
    small = cv2.imencode(".jpg", np.full((600, 800, 3), 120, np.uint8))[1].tobytes()
    out = cv2.imdecode(np.frombuffer(fb._shrink(big, 1600), np.uint8), cv2.IMREAD_COLOR)
    assert max(out.shape[:2]) == 1600 and out.shape[1] / out.shape[0] == 3200 / 2400
    assert fb._shrink(small, 1600) is small  # меньше предела — не трогаем
    assert fb._shrink(big, 0) is big  # 0 — оригинал


def test_network_drop_pauses_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(fb, "DELAY", 0)
    monkeypatch.setattr(fb.time, "sleep", lambda _s: None)
    monkeypatch.setattr(fb, "_ask", lambda msg: pytest.fail(f"разбудил человека: {msg}"))
    net = iter([False, False, True] + [True] * 100)  # Wi-Fi пропал на две проверки
    monkeypatch.setattr(fb, "_net_ok", lambda timeout=5: next(net))

    class Flaky(FakeGallery):
        fails = 2

        def get(self, url: str) -> None:
            if self.fails:  # первые загрузки — как при обрыве сети
                self.fails -= 1
                raise TimeoutError("страница не загрузилась")
            super().get(url)

    plates = [(str(9000 + i), f"{i:04d} AA 70") for i in range(10)]
    (tmp_path / "photos").mkdir()
    for pid, _ in plates:
        (tmp_path / "photos" / f"{pid}.jpg").write_bytes(b"jpg")
    items = fb.fetch(Flaky(plates), target=100, out_dir=tmp_path, ctype=6)
    assert len(items) == 10, "после возвращения сети сбор должен продолжиться"
