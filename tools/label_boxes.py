"""Разметка рамок номеров для эталона детектора: страница в браузере, без установки.

    python -m tools.label_boxes --prepare     # выбрать фото, детектор предложит рамки
    python -m tools.label_boxes               # http://127.0.0.1:8765 — размечать

Детектор заранее рисует свои рамки пунктиром: клик — принять, пропущенные —
рисовать мышью, лишние — выделить и Delete. Размечать нужно ВСЕ номера на фото,
не только прицеп: эталон меряет и пропуски, и ложные срабатывания.

Зачем руками: все автоматические рамки подтверждала наша же модель, и мерить
детектор только на них — значит не увидеть того, чего не видит она сама.

Фото — из части test (эталон eval-v2), в обучение они не попадают никогда.
Рамки — в формате YOLO: data/det-bench/labels/<pid>.txt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "eval-v2"
OUT = ROOT / "data" / "det-bench"
ITEMS = OUT / "items.json"
PID_RE = re.compile(r"^\d{1,12}$")


UZ_INDEX = ROOT / "data" / "uz" / "index.json"


def prepare(n_trailers: int, n_other: int, n_moto: int = 0, seed: int = 0) -> None:
    """Выбрать фото и записать предложения детектора.

    Трейлеры и прочее берём из eval-v2 (уже отложенная часть). Мото там всего 25,
    поэтому их (`n_moto`) добираем из всей собранной выборки data/uz. Любое фото,
    попавшее сюда, build_dataset исключает из обучения — эталон остаётся честным.

    Можно звать повторно: уже добавленные фото пропускаются.
    """
    import cv2

    from app.config import settings
    from app.core import plate_rules
    from app.core.detector import PlateDetector

    bench = json.loads((BENCH / "index.json").read_text())
    rng = random.Random(seed)

    def of(src: dict, base: Path, *classes: str) -> list[tuple[str, dict, Path]]:
        return [(p, v, base) for p, v in src.items()
                if plate_rules.match(v["gt"]).plate_class in classes]

    trailers = of(bench, BENCH, "trailer")
    others = [(p, v, BENCH) for p, v in bench.items()
              if plate_rules.match(v["gt"]).plate_class not in ("trailer", "moto", "special")]
    moto = of(json.loads(UZ_INDEX.read_text()), UZ_INDEX.parent, "moto", "special")
    chosen = (rng.sample(trailers, min(n_trailers, len(trailers)))
              + rng.sample(moto, min(n_moto, len(moto)))
              + rng.sample(others, min(n_other, len(others))))
    rng.shuffle(chosen)
    pick = [(p, (base / v["file"]).resolve(), v["gt"]) for p, v, base in chosen]

    det = PlateDetector(weights=settings.detector_weights, device=settings.device)
    items = json.loads(ITEMS.read_text()) if ITEMS.exists() else {}
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for i, (pid, path, gt) in enumerate(pick, 1):
        if pid in items:
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        # Низкий порог и крупный кадр: лишнюю рамку удалить быстрее, чем нарисовать
        sug = [[d.x1, d.y1, d.x2, d.y2, round(d.conf, 2)]
               for d in det.detect(img, imgsz=1536, conf=0.10)]
        items[pid] = {"file": str(path), "w": w, "h": h, "gt": gt,
                      "suggest": sug, "done": False}
        if i % 20 == 0:
            print(f"  {i}/{len(pick)}", flush=True)
            ITEMS.write_text(json.dumps(items, ensure_ascii=False))
    ITEMS.write_text(json.dumps(items, ensure_ascii=False))
    print(f"готово: {len(items)} фото -> {ITEMS}")


def _load_boxes(pid: str, w: int, h: int) -> list[list[int]] | None:
    f = OUT / "labels" / f"{pid}.txt"
    if not f.exists():
        return None
    out = []
    for line in f.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = (float(x) for x in line.split())
            out.append([round((cx - bw / 2) * w), round((cy - bh / 2) * h),
                        round((cx + bw / 2) * w), round((cy + bh / 2) * h)])
    return out


def _save_boxes(pid: str, item: dict, boxes: list[list[float]]) -> None:
    w, h = item["w"], item["h"]
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
        y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
        if x2 - x1 >= 3 and y2 - y1 >= 3:
            lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    (OUT / "labels" / f"{pid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


class Handler(BaseHTTPRequestHandler):
    items: dict = {}

    def log_message(self, *_a) -> None:  # не засорять терминал строкой на каждый запрос
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def _pid(self, prefix: str) -> str | None:
        pid = self.path[len(prefix):]
        return pid if PID_RE.match(pid) and pid in self.items else None

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            self._json([{"pid": p, "gt": v["gt"], "done": v["done"]}
                        for p, v in self.items.items()])
        elif self.path.startswith("/api/item/") and (pid := self._pid("/api/item/")):
            v = self.items[pid]
            self._json({"pid": pid, "w": v["w"], "h": v["h"], "gt": v["gt"],
                        "suggest": v["suggest"], "boxes": _load_boxes(pid, v["w"], v["h"])})
        elif self.path.startswith("/img/") and (pid := self._pid("/img/")):
            self._send(200, Path(self.items[pid]["file"]).read_bytes(), "image/jpeg")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if not (self.path.startswith("/api/save/") and (pid := self._pid("/api/save/"))):
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        _save_boxes(pid, self.items[pid], body.get("boxes", []))
        self.items[pid]["done"] = True
        # Пишем через файл на диске, а не свою память: если items.json пополнили
        # извне (prepare добавил мото), его записи переживут сохранение.
        on_disk = json.loads(ITEMS.read_text()) if ITEMS.exists() else {}
        on_disk[pid] = self.items[pid]
        ITEMS.write_text(json.dumps(on_disk, ensure_ascii=False))
        self._json({"ok": True, "done": sum(v["done"] for v in on_disk.values())})


PAGE = r"""<!doctype html><html lang="ru"><meta charset="utf-8">
<title>Разметка номеров</title>
<style>
 html,body{margin:0;height:100%;background:#111;color:#ddd;font:14px system-ui,sans-serif}
 #bar{display:flex;gap:16px;align-items:center;padding:8px 12px;background:#1c1c1c}
 #bar b{color:#fff} #gt{font:600 18px monospace;color:#ffd54a}
 #help{color:#888;font-size:12px;margin-left:auto}
 canvas{display:block;margin:0 auto;cursor:crosshair}
</style>
<div id="bar"><span id="pos"></span><span id="gt"></span><span id="state"></span>
<span id="help">клик по пунктиру — принять · тянуть мышью — новая рамка ·
клик по рамке + Delete — удалить · A — принять все ·
Enter / → — сохранить и дальше · ← — назад</span></div>
<canvas id="c"></canvas>
<script>
const c = document.getElementById('c'), g = c.getContext('2d');
let list = [], i = 0, item = null, img = new Image();
let boxes = [], sug = [], sel = -1, k = 1, drag = null;
const $ = id => document.getElementById(id);
async function start() {
  list = await (await fetch('/api/list')).json();
  i = Math.max(0, list.findIndex(x => !x.done));
  load();
}
async function load() {
  item = await (await fetch('/api/item/' + list[i].pid)).json();
  boxes = (item.boxes || []).map(b => b.slice(0, 4));
  sug = item.boxes ? [] : item.suggest.map(b => b.slice(0, 5));
  sel = -1;
  img = new Image();
  img.onload = () => { fit(); draw(); };
  img.src = '/img/' + item.pid;
  const done = list.filter(x => x.done).length;
  $('pos').innerHTML = `<b>${i + 1}</b> / ${list.length} · размечено ${done}`;
  $('gt').textContent = item.gt;
  $('state').textContent = list[i].done ? '✓ сохранено' : '';
}
function fit() {
  k = Math.min((innerWidth - 8) / item.w, (innerHeight - 48) / item.h);
  c.width = Math.round(item.w * k); c.height = Math.round(item.h * k);
}
function rect(b, color, dash, lw) {
  g.setLineDash(dash); g.strokeStyle = color; g.lineWidth = lw;
  g.strokeRect(b[0] * k, b[1] * k, (b[2] - b[0]) * k, (b[3] - b[1]) * k);
}
function draw() {
  g.drawImage(img, 0, 0, c.width, c.height);
  sug.forEach(b => rect(b, '#ffd54a', [6, 4], 2));
  boxes.forEach((b, j) => rect(b, j === sel ? '#ff5252' : '#4caf50', [], 2.5));
  if (drag && drag.now) rect([drag.x, drag.y, drag.now[0], drag.now[1]], '#4fc3f7', [], 1.5);
}
const inside = (b, x, y) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
const pt = e => {
  const r = c.getBoundingClientRect();
  return [(e.clientX - r.left) / k, (e.clientY - r.top) / k];
};
c.onmousedown = e => { const [x, y] = pt(e); drag = {x, y, now: null}; };
c.onmousemove = e => { if (drag) { drag.now = pt(e); draw(); } };
c.onmouseup = e => {
  const [x, y] = pt(e), d = drag; drag = null;
  if (d && d.now && Math.abs(x - d.x) * k > 4 && Math.abs(y - d.y) * k > 4) {
    boxes.push([Math.min(d.x, x), Math.min(d.y, y), Math.max(d.x, x), Math.max(d.y, y)]);
    sel = boxes.length - 1;
  } else {
    const s = sug.findIndex(b => inside(b, x, y));
    if (s >= 0) { boxes.push(sug[s].slice(0, 4)); sug.splice(s, 1); sel = boxes.length - 1; }
    else sel = boxes.findIndex(b => inside(b, x, y));
  }
  draw();
};
async function save() {
  const r = await (await fetch('/api/save/' + item.pid, {method: 'POST',
    headers: {'Content-Type': 'application/json'}, body: JSON.stringify({boxes})})).json();
  list[i].done = true;
  return r;
}
addEventListener('keydown', async e => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && sel >= 0) {
    boxes.splice(sel, 1); sel = -1; draw();
  } else if (e.key === 'a' || e.key === 'ф') {
    boxes.push(...sug.map(b => b.slice(0, 4))); sug = []; draw();
  } else if (e.key === 'Enter' || e.key === 'ArrowRight') {
    await save();
    if (i < list.length - 1) i++;
    load();
  } else if (e.key === 'ArrowLeft' && i > 0) {
    i--; load();
  }
});
addEventListener('resize', () => { if (item) { fit(); draw(); } });
start();
</script></html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepare", action="store_true", help="выбрать фото и предложить рамки")
    ap.add_argument("--trailers", type=int, default=120)
    ap.add_argument("--other", type=int, default=60)
    ap.add_argument("--moto", type=int, default=0, help="добрать N мото из data/uz в эталон")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.prepare:
        prepare(args.trailers, args.other, args.moto)
        return 0
    if not ITEMS.exists():
        print("сначала: python -m tools.label_boxes --prepare")
        return 2
    Handler.items = json.loads(ITEMS.read_text())
    done = sum(v["done"] for v in Handler.items.values())
    print(f"http://127.0.0.1:{args.port}  — фото {len(Handler.items)}, размечено {done}")
    print("Ctrl+C — выход, размеченное уже сохранено")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
