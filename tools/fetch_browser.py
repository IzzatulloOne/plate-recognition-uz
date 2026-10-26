"""Сбор выборки через живой браузер: platesmania закрылась от HTTP-клиента.

`tools.eval_uz` ходит на сайт через urllib и с сентября получает 403 на каждой
странице. Дело не в объёме запросов — между прогонами проходили недели. Сайт
поднял защиту, и urllib опознаётся как бот, чем он и является.

Живой Chromium проходит: капчу один раз решает человек, дальше страницы листает
скрипт внутри уже открытой сессии. Фото качаются обычным requests с куками и
User-Agent из браузера — тащить картинку кликами по DOM незачем.

Вторая причина, по которой это нужно: галерея умеет фильтровать по типу номера
прямо в URL. Прицепов в общем потоке 5.8%, и набрать их, листая всё подряд,
стоит сотен страниц. С фильтром — десятки.

    python -m tools.fetch_browser --probe                    # какой ctype что отдаёт
    python -m tools.fetch_browser -n 1500 --ctype 6          # прицепы, их 4261
    python -m tools.fetch_browser -n 400 --ctype 1 --region 115 --mask 40

Глубже десятой страницы сайт не пускает. Упёршись, скрипт сам режет выборку по
следующей цифре префикса номера и добирает остаток — каждому срезу достаётся
своя пагинация. Руками это же делается через --slice.

Индекс пишется в формате tools.eval_uz, поэтому собранное сразу идёт в дело:

    python -m tools.build_dataset --build
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
from itertools import product
from pathlib import Path

import requests

from app.core import plate_rules
from tools.eval_uz import BASE, ITEM_RE

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "uz"
CHROMIUM = "/usr/bin/chromium"
CHROMEDRIVER = "/usr/bin/chromedriver"
PAGE_TIMEOUT = 60  # с; меньше 120 с клиента Selenium — см. open_browser
#: профиль браузера между запусками: куки сайта живут, как в обычном браузере,
#: и капчу не приходится решать заново на каждый тип номера
PROFILE = ROOT / "data" / "chromium-profile"
DELAY = 1.5  # сек между страницами; человек листает примерно так же


def page_url(
    page: int, ctype: int | None = None, region: int | None = None, mask: str = ""
) -> str:
    """Адрес страницы галереи — ровно в той форме, какой сайт ссылается сам.

    В разметке пагинация выглядит как `gallery.php?&ctype=6&start=1`, где start —
    номер страницы минус один. Придуманная форма `gallery-11?ctype=6` работала
    до десятой страницы, а дальше отдавала «License plates found 0», и это
    выглядело как конец данных. Не выдумывать схему URL, когда страница
    приносит свою.

    Первая страница остаётся без start: её ссылку виджет пагинации не показывает
    (она активная), и проверить `start=0` не на чем.

    ctype — тип номера (см. --probe), region — код области (Фергана 115),
    mask — префикс номера, сайт ждёт его со звёздочкой.
    """
    params = [
        f"{k}={v}"
        for k, v in (("r", region), ("nomer", f"{mask}*" if mask else None), ("ctype", ctype))
        if v
    ]
    if page <= 1:
        return f"{BASE}/uz/gallery" + ("?" + "&".join(params) if params else "")
    return f"{BASE}/uz/gallery.php?" + "&".join([*params, f"start={page - 1}"])


#: сайт сам пишет, сколько номеров нашлось по фильтру: «License plates found <b>4.261</b>»
COUNTER_RE = re.compile(r"plates found\s*<b>([^<]+)</b>", re.I)


def found_total(body: str) -> int | None:
    """Сколько номеров у фильтра по мнению сайта. None — счётчика на странице нет.

    Это и отличает законно пустую выборку от капчи: раньше скрипт в обоих
    случаях будил человека.
    """
    m = COUNTER_RE.search(body)
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))  # тысячи разделены точками: 4.261
    return int(digits) if digits else None


def open_browser():
    """Chromium под управлением скрипта, но внешне обычный.

    Без disable-blink-features страница читает navigator.webdriver и видит
    автоматизацию ещё до капчи.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
    except ImportError:  # отдельный extra, в обычной установке его нет
        raise SystemExit('нет selenium: uv pip install -e ".[scrape]"') from None

    for path, what in ((CHROMIUM, "chromium"), (CHROMEDRIVER, "chromedriver")):
        if not Path(path).exists():
            raise SystemExit(f"нет {what} по пути {path} — поставь его или правь константу")

    opts = Options()
    opts.binary_location = CHROMIUM
    opts.add_argument(f"--user-data-dir={PROFILE}")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(service=Service(CHROMEDRIVER), options=opts)
    # Свой предел загрузки страницы короче 120 с клиента Selenium: при обрыве сети
    # иначе зависал сам клиент (ReadTimeoutError) и процесс падал — так пропал
    # дневной сбор, когда отвалился Wi-Fi. С пределом get() просто бросает исключение.
    driver.set_page_load_timeout(PAGE_TIMEOUT)
    return driver


def _net_ok(timeout: float = 5) -> bool:
    """Есть ли связь с сайтом — простое TCP-соединение, без запроса страницы."""
    import socket

    try:
        socket.create_connection(("platesmania.com", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def _wait_net(max_wait: float = 6 * 3600) -> bool:
    """Сеть пропала — ждать, пока вернётся (до max_wait). False — так и не вернулась."""
    if _net_ok():
        return True
    print("  сеть пропала — жду, проверяю раз в 30 с", flush=True)
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(30)
        if _net_ok():
            print(f"  сеть вернулась через {(time.time() - t0) / 60:.0f} мин", flush=True)
            return True
    return False


def _goto(driver, url: str, attempts: int = 5) -> bool:
    """driver.get, переживающий обрыв сети: ждём связь и повторяем ту же страницу."""
    for i in range(attempts):
        try:
            driver.get(url)
            return True
        except Exception as exc:  # TimeoutException, WebDriverException, ReadTimeoutError
            print(f"  страница не загрузилась ({type(exc).__name__}), "
                  f"попытка {i + 1}/{attempts}", flush=True)
            if not _wait_net():
                return False
            time.sleep(5)
    return False


def http_session(driver) -> requests.Session:
    """requests с куками браузера. Домен у кук не проставляем намеренно:

    фото лежат на img*.platesmania.com, а кука, выданная на platesmania.com,
    на поддомен бы не ушла.
    """
    s = requests.Session()
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"])
    s.headers["User-Agent"] = driver.execute_script("return navigator.userAgent;")
    return s


def _wait(message: str, unattended: bool) -> None:
    """Дождаться человека, а без присмотра — просто выждать.

    Капчу скрипт не решает ни в каком режиме: это проверка на человека, и обходить
    её мы не будем. Ночью можно только переждать — если сайт не отпустит, этап
    закончится сам, и цепочка пойдёт дальше с тем, что успели собрать.
    """
    if unattended:
        print(f"{message}\nБез присмотра: жду 2 минуты и пробую снова.", flush=True)
        time.sleep(120)
    else:
        _ask(message)


def _ask(message: str) -> None:
    print(f"\n{'=' * 62}\n{message}\nENTER здесь, когда готово.\n{'=' * 62}")
    input()


#: карточка машины в галерее; на ней же висит название модели
CARD_RE = re.compile(r"(?=<div[^>]*\bpanel-grey\b)")
MODEL_RE = re.compile(r"<h4[^>]*\btext-center\b[^>]*>(?P<model>.*?)</h4>", re.S)
TAGS_RE = re.compile(r"<[^>]+>")


def _plain(m: re.Match | None) -> str:
    """Текст из куска HTML: снять теги, расклеить сущности, схлопнуть пробелы."""
    if not m:
        return ""
    return " ".join(html.unescape(TAGS_RE.sub(" ", m.group(1))).split())


def listing(body: str) -> list[dict]:
    """Разбор страницы галереи по карточкам -> pid, номер, фото, модель авто.

    Текст номера сайт кладёт в alt «информера» — отрисованной пластины рядом с
    фото. Это готовый эталон, размечать руками ничего не нужно.

    Разбираем именно карточками, а не всю страницу разом: название модели иначе
    не привязать к нужной машине. Если разметка карточек изменится, кусок будет
    один на всю страницу — номера всё равно найдутся, модель просто останется
    пустой. Терять данные из-за переименованного класса нельзя.
    """
    out: list[dict] = []
    for card in CARD_RE.split(body):
        found = list(ITEM_RE.finditer(card))
        model = _plain(MODEL_RE.search(card)) if len(found) == 1 else ""
        for m in found:
            gt = plate_rules.normalize(html.unescape(m.group("gt")))
            if not gt:
                continue
            out.append(
                {
                    "pid": m.group("pid"),
                    "gt": gt,
                    "photo": m.group("photo").replace("/m/", "/o/"),
                    "model": model,
                }
            )
    return out


def dump_page(driver, url: str, dst: Path) -> None:
    """Сохранить HTML в соседней вкладке, не сбивая обход галереи.

    Нужно, чтобы доставать новые поля (цвет, год) по настоящей разметке,
    а не угадывать селекторы.
    """
    main = driver.current_window_handle
    driver.switch_to.new_window("tab")
    try:
        driver.get(url)
        time.sleep(DELAY)
        dst.write_text(driver.page_source, encoding="utf-8")
        print(f"  дамп: {dst}", flush=True)
    finally:
        driver.close()
        driver.switch_to.window(main)


def probe(driver, ctypes: range, pages: int = 1) -> None:
    """Разложить по форматам первые страницы каждого ctype.

    Нумерация типов у сайта своя и нигде не описана. Угадывать её не нужно —
    дешевле посмотреть, что приходит.
    """
    for ct in ctypes:
        mix: dict[str, int] = {}
        for page in range(1, pages + 1):
            _goto(driver, page_url(page, ctype=ct))
            time.sleep(DELAY)
            for row in listing(driver.page_source):
                fmt = plate_rules.match(row["gt"]).format or "не опознан"
                mix[fmt] = mix.get(fmt, 0) + 1
        total = sum(mix.values())
        top = ", ".join(f"{k} {v}" for k, v in sorted(mix.items(), key=lambda kv: -kv[1]))
        print(f"ctype={ct:<3} всего {total:<4} {top or '— пусто'}", flush=True)


#: Короче двух символов сайт номер не ищет: nomer=0* отдаёт ноль, nomer=01* — 18 949.
#: Однозначные срезы первой ночью вернули пустоту на всех трёх этапах сбора.
MIN_PREFIX = 2
#: третий символ у частных — буква серии (01 A 123 BC), у юрлиц — цифра
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _shrink(jpeg: bytes, max_side: int) -> bytes:
    """Ужать фото до max_side по длинной стороне; 0 — оставить как есть.

    Оригинал в среднем весит 727 КБ, и полный сбор в 40 тысяч фото на диск не
    влезает. Учим на 960 (детектор) и 100x32 (распознаватель), так что 1600
    по длинной стороне детали номера сохраняет.
    """
    if not max_side:
        return jpeg
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None or max(img.shape[:2]) <= max_side:
        return jpeg
    h, w = img.shape[:2]
    k = max_side / max(h, w)
    img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else jpeg


def _deeper(mask: str, depth: int, limit: int) -> list[tuple[str, int]]:
    """Срезы следующего уровня. Пусто, если глубже нельзя.

    Короткий префикс сразу добивается цифрами до двух символов (у всех номеров
    УЗ первые два символа — цифры: код региона или начало номера прицепа),
    дальше — плюс один символ из цифр и букв.
    """
    if depth >= limit:
        return []
    if len(mask) < MIN_PREFIX:
        need = MIN_PREFIX - len(mask)
        return [(mask + "".join(t), depth + 1) for t in product("0123456789", repeat=need)]
    return [(mask + ch, depth + 1) for ch in ALPHABET]


def _split_note(label: str, mask: str, extra: list) -> list:
    print(
        f"{label}добираю срезами {mask}0* … {mask}9*" if extra else f"{label}глубже резать некуда",
        flush=True,
    )
    return extra


def fetch(
    driver,
    target: int,
    out_dir: Path,
    formats: set[str] | None = None,
    ctype: int | None = None,
    region: int | None = None,
    masks: list[str] | None = None,
    max_pages: int = 200,
    start_page: int = 1,
    dump_html: Path | None = None,
    auto_slice: int = 2,
    unattended: bool = False,
    per_slice: int = 0,
    max_side: int = 0,
) -> dict:
    """Обойти галерею и сложить фото с эталонным текстом в out_dir.

    masks — префиксы номера, по одному проходу на каждый. Нужны, если у фильтра
    больше номеров, чем отдаёт пагинация: срез по префиксу даёт каждому куску
    свой запас страниц.

    auto_slice — на сколько цифр углубляться самостоятельно, упёршись в предел
    пагинации. Иначе прогон обрывался бы, а капчу пришлось бы решать заново.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "photos").mkdir(exist_ok=True)
    index = out_dir / "index.json"
    items: dict = json.loads(index.read_text()) if index.exists() else {}
    seen = set(items)
    print(f"в {out_dir} уже {len(items)} шт.")

    session = http_session(driver)
    skipped, dumped = 0, False

    queue: list[tuple[str, int]] = [(m, 0) for m in (masks or [""])]
    while queue and len(items) < target:
        mask, depth = queue.pop(0)
        page, empty_pages, walked, slice_new = start_page, 0, 0, 0
        shown: set[tuple[str, ...]] = set()  # какие страницы этот срез уже видел
        label = f"[{mask}*] " if mask else ""

        while len(items) < target and page < start_page + max_pages:
            if per_slice and slice_new >= per_slice:
                break  # квота среза: иначе весь сбор ушёл бы в первые регионы по порядку
            url = page_url(page, ctype, region, mask)
            if not _goto(driver, url):
                print(f"{label}сайт не открывается и после ожидания сети — заканчиваю",
                      flush=True)
                queue.clear()
                break
            time.sleep(DELAY)
            body = driver.page_source
            rows = listing(body)
            total = found_total(body)

            if page == start_page and total is not None:
                print(f"{label}сайт насчитал {total} номеров по фильтру", flush=True)

            if not rows:
                # Три разные причины, и сайт сам говорит, какая именно.
                if total == 0:
                    print(f"{label}по фильтру пусто", flush=True)
                    break
                if total is None:
                    # Не галерея вовсе: капча, ошибка или разлогин. Решает человек.
                    empty_pages += 1
                    if empty_pages >= 2:
                        print(f"{label}страница снова не галерея — заканчиваю")
                        break
                    _wait(f"Страница {page} не похожа на галерею. Если это капча — реши её.",
                          unattended)
                    session = http_session(driver)
                    continue
                # Счётчик есть и он больше нуля, а строк нет: глубже сайт не пускает.
                # Берём остаток срезами по следующей цифре префикса — у каждого
                # среза своя пагинация, и вместе они покрывают весь фильтр.
                print(f"{label}страница {page} пуста при {total} в фильтре — предел глубины")
                queue = queue + _split_note(label, mask, _deeper(mask, depth, auto_slice))
                break
            empty_pages = 0

            # Чаще предел выглядит иначе: сайт не отдаёт пустую страницу, а
            # повторяет последнюю разрешённую. Прогон по прицепам после тысячи
            # номеров ещё сто страниц подряд получал «+0» и до нарезки не дошёл.
            # Повтор страницы и есть сигнал; счётчик говорит, конец это или предел.
            sig = tuple(r["pid"] for r in rows)
            if sig in shown:
                if total and total - walked >= len(rows):
                    print(f"{label}после {walked} из {total} сайт повторяет страницы — предел")
                    queue = queue + _split_note(label, mask, _deeper(mask, depth, auto_slice))
                else:
                    print(f"{label}страницы кончились", flush=True)
                break
            shown.add(sig)
            walked += len(rows)

            found = 0
            for row in rows:
                pid, gt, photo = row["pid"], row["gt"], row["photo"]
                if pid in seen:
                    continue
                seen.add(pid)
                if formats and plate_rules.match(gt).format not in formats:
                    skipped += 1
                    continue

                dst = out_dir / "photos" / f"{pid}.jpg"
                if not dst.exists():
                    try:
                        r = session.get(photo, headers={"Referer": url}, timeout=20)
                        r.raise_for_status()
                    except Exception as exc:
                        # обрыв сети — дождаться и повторить; иначе фото пропало бы зря
                        if not _net_ok() and _wait_net():
                            try:
                                r = session.get(photo, headers={"Referer": url}, timeout=20)
                                r.raise_for_status()
                            except Exception as exc2:
                                print(f"  {pid}: фото не скачалось ({exc2})")
                                continue
                        else:
                            print(f"  {pid}: фото не скачалось ({exc})")
                            continue
                    dst.write_bytes(_shrink(r.content, max_side))
                    time.sleep(DELAY / 3)

                items[pid] = {
                    "gt": gt,
                    "photo": photo,
                    "file": str(dst.relative_to(out_dir)),
                    "model": row["model"],
                }
                found += 1
                slice_new += 1
                print(f"  {pid}: {gt:<10} {row['model']}", flush=True)

                if dump_html and not dumped:
                    dumped = True
                    dump_html.mkdir(parents=True, exist_ok=True)
                    (dump_html / "gallery.html").write_text(body, encoding="utf-8")
                    print(f"  дамп: {dump_html / 'gallery.html'}", flush=True)
                    dump_page(driver, f"{BASE}/uz/nomer{pid}", dump_html / f"nomer{pid}.html")
                if len(items) >= target or (per_slice and slice_new >= per_slice):
                    break

            print(
                f"{label}страница {page}: +{found}, всего {len(items)} "
                f"(мимо фильтра {skipped})",
                flush=True,
            )
            index.write_text(json.dumps(items, ensure_ascii=False, indent=1))  # не терять прогресс
            page += 1
            time.sleep(DELAY)

    index.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"\nвыборка: {len(items)} шт. в {out_dir}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-n", "--target", type=int, default=300,
                    help="сколько номеров должно стать в индексе всего")
    ap.add_argument("--add", type=int, metavar="N",
                    help="сколько добавить к уже собранному (вместо -n)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="куда класть выборку")
    ap.add_argument("--formats", nargs="+", metavar="FMT", help="оставить только эти форматы")
    ap.add_argument("--ctype", type=int, help="тип номера в нумерации сайта (см. --probe)")
    ap.add_argument("--region", type=int, help="код области (Фергана 115)")
    ap.add_argument("--mask", default="", help="префикс номера, например 40")
    ap.add_argument(
        "--slice",
        type=int,
        default=0,
        metavar="N",
        help="резать выборку по N цифрам префикса (обойти предел глубины пагинации)",
    )
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument(
        "--dump-html",
        type=Path,
        metavar="DIR",
        help="сохранить HTML галереи и одной карточки — чтобы доставать новые поля по факту",
    )
    ap.add_argument("--probe", action="store_true", help="какой ctype какие форматы отдаёт")
    ap.add_argument("--max-side", type=int, default=0, metavar="PX",
                    help="ужимать фото до PX по длинной стороне (0 — оригинал)")
    ap.add_argument("--per-slice", type=int, default=0, metavar="N",
                    help="не больше N новых номеров с одного среза — регионы поровну")
    ap.add_argument("--unattended", action="store_true",
                    help="без присмотра: не ждать ENTER; на капче выждать и закончить этап")
    ap.add_argument("--probe-max", type=int, default=9, help="до какого ctype пробовать")
    args = ap.parse_args()

    if args.add:
        index = args.out / "index.json"
        have = len(json.loads(index.read_text())) if index.exists() else 0
        args.target = have + args.add

    known = {f.name for f in plate_rules.PLATE_FORMATS}
    formats = set(args.formats) if args.formats else None
    if formats and (bad := formats - known):
        ap.error(f"неизвестные форматы: {', '.join(sorted(bad))}. Есть: {', '.join(sorted(known))}")

    if args.slice:
        digits = max(args.slice, MIN_PREFIX - len(args.mask))
        masks = [args.mask + "".join(t) for t in product("0123456789", repeat=digits)]
    else:
        masks = [args.mask]
    if args.slice:
        print(f"срезов по префиксу: {len(masks)} ({masks[0]}* … {masks[-1]}*)")

    driver = open_browser()
    try:
        _goto(driver, page_url(args.start_page, args.ctype, args.region, args.mask))
        if not args.unattended:
            _ask("Открылся браузер. Реши капчу, если она есть, и дождись галереи.")
        if args.probe:
            probe(driver, range(1, args.probe_max + 1))
            return 0
        fetch(
            driver,
            target=args.target,
            out_dir=args.out,
            formats=formats,
            ctype=args.ctype,
            region=args.region,
            masks=masks,
            max_pages=args.max_pages,
            start_page=args.start_page,
            dump_html=args.dump_html,
            unattended=args.unattended,
            per_slice=args.per_slice,
            max_side=args.max_side,
        )
    finally:
        driver.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
