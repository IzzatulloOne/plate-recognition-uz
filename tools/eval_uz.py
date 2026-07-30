"""Замер точности на реальных узбекских номерах из галереи platesmania.com/uz.

Сайт отдаёт вместе с фото машины «информер» — отрисованную пластину, в alt которой
лежит точный текст номера. Это готовый ground truth: скачиваем N фото + подписи и
считаем точность детекции и распознавания без ручной разметки.

    python -m tools.eval_uz --fetch 40          # скачать выборку в data/eval/
    python -m tools.eval_uz                     # прогнать пайплайн и посчитать метрики
    python -m tools.eval_uz --fetch 40 --run
    python -m tools.eval_uz --run --save-fails  # сохранить кропы ошибок для разбора

Скрипт бережно относится к сайту: пауза между запросами, только то, что нужно.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path

import cv2

from app.config import settings
from app.core import plate_rules

BASE = "https://platesmania.com"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
INDEX = EVAL_DIR / "index.json"
DELAY = 1.0  # сек между запросами


def _get(url: str, referer: str = BASE, attempts: int = 3) -> bytes:
    """GET с ретраями: длинные проходы по галерее спотыкаются о временные сбои DNS."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Referer": referer, "Accept-Language": "ru,en;q=0.9"},
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except Exception as exc:  # сеть, DNS, 5xx — пробуем ещё раз
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** (attempt + 1))
    raise last if last else RuntimeError(f"не скачалось: {url}")


#: у части картинок между src и alt стоит loading="lazy", поэтому между атрибутами
#: должно быть [^>]*, а не жёсткая последовательность — иначе теряется 80% элементов
ITEM_RE = re.compile(
    r'<img[^>]+src="(?P<photo>https://img\d+\.platesmania\.com/\d+/m/(?P<pid>\d+)\.jpg)"[^>]*>'
    r".*?"
    r'<img[^>]+src="(?P<inf>https://img\d+\.platesmania\.com/\d+/inf/[^"]+\.png)"'
    r'[^>]*\salt="(?P<gt>[^"]+)"',
    re.S,
)


def informer_color(url: str, pid: str) -> tuple[str, float]:
    """Цвет пластины по «информеру» — отрисованному номеру (маленький PNG).

    Отрисовка передаёт цвет фона точно, поэтому это дешёвый способ узнать тип номера,
    не скачивая большое фото: жёлтые и зелёные встречаются редко, и перебирать
    страницы галереи по цвету иначе слишком дорого.
    """
    import cv2
    import numpy as np

    try:
        raw = _get(url, referer=f"{BASE}/uz/nomer{pid}")
    except Exception:
        return "unknown", 0.0
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "unknown", 0.0
    return plate_rules.classify_color(img)


def fetch(
    target: int,
    out_dir: Path | None = None,
    orig: bool = True,
    colors: set[str] | None = None,
    max_pages: int = 40,
) -> dict:
    """Скачивает выборку (фото + текст номера) со страниц галереи.

    colors — брать только номера с таким цветом фона (например {"yellow", "green"}).
    Цвет определяется по информеру, поэтому большие фото качаются только для нужных.
    out_dir по умолчанию data/eval; tools.build_dataset передаёт свою папку.
    """
    out_dir = Path(out_dir) if out_dir else EVAL_DIR
    index = out_dir / "index.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "photos").mkdir(exist_ok=True)
    items: dict = json.loads(index.read_text()) if index.exists() else {}
    seen_pids = set(items)
    color_stats: dict[str, int] = {}

    page = 1
    misses = 0
    while len(items) < target and page <= max_pages:
        url = f"{BASE}/uz/gallery" + (f"-{page}" if page > 1 else "")
        print(f"страница {page}: {url}", flush=True)
        try:
            body = _get(url).decode("utf-8", "replace")
            misses = 0
        except Exception as exc:
            # одна упавшая страница не должна обрывать длинный проход по галерее
            misses += 1
            print(f"  не удалось ({misses}/5): {exc}", flush=True)
            if misses >= 5:
                print("  слишком много подряд — останавливаюсь")
                break
            page += 1
            time.sleep(DELAY * 3)
            continue
        found = 0
        for m in ITEM_RE.finditer(body):
            pid, gt = m.group("pid"), plate_rules.normalize(html.unescape(m.group("gt")))
            if pid in seen_pids or not gt:
                continue
            seen_pids.add(pid)

            color, frac = "", 0.0
            if colors:
                color, frac = informer_color(m.group("inf"), pid)
                color_stats[color] = color_stats.get(color, 0) + 1
                time.sleep(DELAY / 4)
                if color not in colors:
                    continue
                print(f"  {pid}: {gt} — {color} ({frac:.2f})", flush=True)

            photo = m.group("photo").replace("/m/", "/o/") if orig else m.group("photo")
            dst = out_dir / "photos" / f"{pid}.jpg"
            if not dst.exists():
                try:
                    dst.write_bytes(_get(photo, referer=f"{BASE}/uz/nomer{pid}"))
                except Exception as exc:
                    print(f"  {pid}: фото не скачалось ({exc})")
                    continue
                time.sleep(DELAY / 2)
            items[pid] = {"gt": gt, "photo": photo, "file": str(dst.relative_to(out_dir))}
            if color:
                items[pid]["informer_color"] = color
            found += 1
            if len(items) >= target:
                break
        print(f"  добавлено {found}, всего {len(items)}", flush=True)
        index.write_text(json.dumps(items, ensure_ascii=False, indent=1))  # не терять прогресс
        page += 1
        time.sleep(DELAY)

    index.write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print(f"\nвыборка: {len(items)} шт. в {out_dir}")
    if color_stats:
        print("цвета просмотренных информеров:", dict(sorted(color_stats.items())))
    return items


def run(
    save_fails: bool = False,
    overrides: dict | None = None,
    quiet: bool = False,
    eval_dir: Path | None = None,
) -> dict:
    """Прогоняет пайплайн по выборке и возвращает метрики.

    overrides — точечная подмена настроек (детектор, imgsz, conf) без правки .env,
    чтобы сравнивать варианты в одинаковых условиях.
    """
    from app.core.pipeline import ANPRPipeline

    eval_dir = Path(eval_dir) if eval_dir else EVAL_DIR
    index = eval_dir / "index.json"
    if not index.exists():
        print(f"нет выборки в {eval_dir} — сначала: python -m tools.eval_uz --fetch 40")
        return {}
    items = json.loads(index.read_text())
    cfg = settings.model_copy(update=overrides) if overrides else settings
    pipe = ANPRPipeline(cfg)
    fails_dir = eval_dir / "fails"
    if save_fails:
        fails_dir.mkdir(exist_ok=True)

    n = det = raw_ok = fin_ok = fmt_ok = retried = 0
    per_class: dict[str, list[int]] = {}
    per_color: dict[str, list[int]] = {}
    rows = []
    t0 = time.time()
    for v in items.values():
        path = eval_dir / v["file"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        n += 1
        gt = v["gt"]
        gt_class = plate_rules.match(gt).plate_class
        res = pipe.process_frame(img)
        retried += bool(res.retried)
        stat = per_class.setdefault(gt_class, [0, 0])
        stat[0] += 1
        cstat = per_color.setdefault(v.get("informer_color", "?"), [0, 0])
        cstat[0] += 1

        if not res.plates:
            rows.append((gt, "—", "—", "нет детекции", ""))
            continue
        det += 1
        crops = getattr(res, "_crops", [])
        i = max(range(len(res.plates)), key=lambda k: res.plates[k].det_conf)
        p = res.plates[i]
        raw_ok += plate_rules.normalize(p.raw_text) == gt
        hit = plate_rules.normalize(p.text) == gt
        fin_ok += hit
        fmt_ok += p.valid
        stat[1] += hit
        cstat[1] += hit
        rows.append((gt, p.raw_text, p.text, p.format or "-", p.color))
        if save_fails and not hit and i < len(crops):
            cv2.imwrite(str(fails_dir / f"{gt}_read_{p.text or 'empty'}.jpg"), crops[i])

    dt = time.time() - t0
    pct = lambda a, b: f"{a}/{b} = {100 * a / max(b, 1):.1f}%"  # noqa: E731
    if not quiet:
        print(f"\n{'GT':12s} {'greedy':13s} {'итог':13s} {'формат':11s} цвет")
        for gt, raw, fin, fmt, col in sorted(rows):
            mark = "OK" if plate_rules.normalize(fin) == gt else "!!"
            print(f"{gt:12s} {raw:13s} {fin:13s} {fmt:11s} {col:7s} {mark}")

    print(
        f"\n=== {Path(cfg.detector_weights).name} imgsz={cfg.det_imgsz} conf={cfg.det_conf} "
        f"retry={cfg.det_retry_imgsz or 'off'}"
    )
    print(f"  выборка {n} фото, {dt / max(n, 1):.2f} с/фото, вторых проходов {retried}")
    print(f"  детекция пластины : {pct(det, n)}")
    print(f"  greedy как есть   : {pct(raw_ok, n)}")
    print(f"  после правил      : {pct(fin_ok, n)}")
    print(f"  формат распознан  : {pct(fmt_ok, n)}")
    print("  по классам номеров:")
    for cls, (tot, ok) in sorted(per_class.items()):
        print(f"    {cls:9s} {pct(ok, tot)}")
    if set(per_color) - {"?"}:
        print("  по цвету пластины (из информера):")
        for col, (tot, ok) in sorted(per_color.items()):
            print(f"    {col:9s} {pct(ok, tot)}")
    if save_fails:
        print(f"  кропы ошибок: {fails_dir}")
    return {
        "detector": Path(cfg.detector_weights).name,
        "imgsz": cfg.det_imgsz,
        "conf": cfg.det_conf,
        "retry": cfg.det_retry_imgsz,
        "n": n,
        "detected": det,
        "text_ok": fin_ok,
        "greedy_ok": raw_ok,
        "format_ok": fmt_ok,
        "retried": retried,
        "sec_per_photo": round(dt / max(n, 1), 3),
        "by_class": {k: tuple(v) for k, v in per_class.items()},
        "by_color": {k: tuple(v) for k, v in per_color.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fetch", type=int, metavar="N", help="скачать N примеров")
    ap.add_argument("--run", action="store_true", help="прогнать пайплайн и посчитать метрики")
    ap.add_argument("--medium", action="store_true", help="качать превью 460x350 вместо оригиналов")
    ap.add_argument("--save-fails", action="store_true")
    ap.add_argument("--detector", help="путь к весам YOLO вместо ANPR_DETECTOR_WEIGHTS")
    ap.add_argument("--out", help="папка выборки (по умолчанию data/eval)")
    ap.add_argument(
        "--colors",
        help="брать только номера этих цветов, напр. yellow,green "
        "(цвет определяется по информеру, большие фото качаются только для них)",
    )
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--conf", type=float)
    ap.add_argument("--no-retry", action="store_true", help="без второго прохода в 1536")
    ap.add_argument(
        "--compare",
        nargs="+",
        metavar="WEIGHTS",
        help="сравнить несколько детекторов на одной выборке в одинаковых условиях",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else EVAL_DIR
    if args.fetch:
        fetch(
            args.fetch,
            out_dir=out_dir,
            orig=not args.medium,
            colors={c.strip() for c in args.colors.split(",")} if args.colors else None,
            max_pages=args.max_pages,
        )

    over: dict = {}
    if args.detector:
        over["detector_weights"] = Path(args.detector)
    if args.imgsz:
        over["det_imgsz"] = args.imgsz
    if args.conf is not None:
        over["det_conf"] = args.conf
    if args.no_retry:
        over["det_retry_imgsz"] = 0

    if args.compare:
        results = []
        for w in args.compare:
            results.append(
                run(
                    overrides={**over, "detector_weights": Path(w)},
                    quiet=True,
                    eval_dir=out_dir,
                )
            )
        print(f"\n{'детектор':28s} {'детекция':>10s} {'текст':>10s} {'формат':>10s} {'с/фото':>8s}")
        for r in results:
            if not r:
                continue
            n = r["n"]
            print(
                f"{r['detector']:28s} {r['detected']}/{n:<8d} {r['text_ok']}/{n:<8d} "
                f"{r['format_ok']}/{n:<8d} {r['sec_per_photo']:>8.2f}"
            )
        return 0

    if args.run or not args.fetch:
        return 0 if run(save_fails=args.save_fails, overrides=over, eval_dir=out_dir) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
