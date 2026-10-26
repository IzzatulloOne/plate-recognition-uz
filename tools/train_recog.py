"""Дообучение распознавателя на кропах из tools.build_dataset.

    python -m tools.train_recog --data data/uz/recog --limit 64 --epochs 1   # проверка на CPU
    python -m tools.train_recog --data /kaggle/input/anpr-uz/recog --out /kaggle/working

Устроено так, чтобы обученное сразу работало в сервисе:

- двухстрочные кропы прицепов разворачиваются в одну строку той же функцией,
  что и при чтении (recognizer.unfold_lines): модель учится ровно на том, что
  увидит в работе;
- препроцесс — тот же prep_gray, что у PlateRecognizer: серый, 100x32, [-1, 1];
- учится всё, кроме head_ctype, и чекпоинт сохраняется в формате
  best_accuracy.pth — load_model() грузит его без правок;
- классы номеров уравнены по весу: прицепов в выборке больше, но модель не
  должна разучиться читать однострочные;
- лучшая эпоха — по средней точности по классам, а не по общей: иначе рост на
  прицепах закрыл бы провал на частных.

Эталонная выборка tools.eval_uz в обучение не попадает никогда: на ней меряется
72/80, и модель, видевшая экзамен, дала бы красивую цифру, которая ничего не значит.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from app.config import settings
from app.core import plate_rules
from app.core.recognizer import PlateRecognizer, moto_unfold, prep_gray, unfold_lines

ROOT = Path(__file__).resolve().parent.parent
EVAL_INDEX = ROOT / "data" / "eval" / "index.json"
#: класс -> как развернуть квадратный кроп в одну строку. Прицеп — верх над низом,
#: мото/спец — сетка 2x2 (регион в правом окне). Обучение и чтение режут одной
#: функцией, иначе модель увидит в работе не то, на чём училась.
UNFOLD = {"trailer": unfold_lines, "moto": moto_unfold, "special": moto_unfold}

#: Сила аугментации на класс. Где реальных фото десятки тысяч (частные, юрлица) —
#: крутить незачем, своё разнообразие уже есть. Где их сотни (мото, спец, высшие
#: органы) — крутим сильно: это единственный способ дать модели разнообразие,
#: пока больше реальных на сайте просто нет. Множитель масштабирует и силу
#: искажений, и вероятность каждого из них.
AUG_STRENGTH = {
    "private": 0.5, "legal": 0.5,
    "trailer": 1.0, "joint": 0.8, "foreign": 0.8,
    "moto": 1.6, "special": 1.6, "gov": 1.6,
    "other": 1.0,
}
DEFAULT_STRENGTH = 1.0


def split_of(plate: str) -> str:
    """То же разбиение, что у tools.build_dataset._split: по тексту номера."""
    from tools.build_dataset import _split

    return _split(plate)


def eval_holdout() -> tuple[set[str], set[str]]:
    """pid и тексты эталонной выборки eval_uz. Текст — потому что та же машина
    на другом снимке для модели почти тот же пример."""
    if not EVAL_INDEX.exists():
        return set(), set()
    ev = json.loads(EVAL_INDEX.read_text())
    return set(ev), {v["gt"] for v in ev.values()}


def load_samples(data: Path, limit: int = 0) -> tuple[list[dict], Counter]:
    """gt.txt (`путь<TAB>ТЕКСТ`) -> образцы с классом номера и частью train/val."""
    hold_pids, hold_texts = eval_holdout()
    skipped: Counter = Counter()
    out: list[dict] = []
    for line in (data / "gt.txt").read_text().splitlines():
        if not line.strip():
            continue
        rel, text = line.split("\t")
        pid = Path(rel).stem.split("_")[0]
        if pid in hold_pids or text in hold_texts:
            skipped["в эталоне eval"] += 1
            continue
        part = split_of(text)
        if part == "test":  # эталон eval-v2 — даже если кроп как-то попал в данные
            skipped["в эталоне eval-v2"] += 1
            continue
        cls = plate_rules.match(text).plate_class or "other"
        out.append({"path": data / rel, "text": text, "cls": cls, "part": part})
    if limit:
        random.Random(0).shuffle(out)
        out = out[:limit]
    return out, skipped


# ---------------------------------------------------------------- аугментации
def augment(img: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """Искажения, которые кроп встречает в работе; strength масштабирует их силу.

    Главное — дрожание бокса: детектор каждый кадр режет пластину чуть иначе,
    и модель не должна зависеть от точных полей. strength=0.5 — вдвое мягче и
    реже, strength=1.6 — сильнее и чаще (для редких классов, см. AUG_STRENGTH).
    """
    h, w = img.shape[:2]
    shift = 0.06 * strength  # дрожание бокса
    warp = 0.03 * strength   # перспектива
    prob = min(0.45 * strength, 0.9)  # вероятность каждого из блюра/шума/сжатия
    pad = int(max(h, w) * 0.1) + 2
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    dx1, dx2 = rng.uniform(-shift, shift, 2) * w
    dy1, dy2 = rng.uniform(-shift, shift, 2) * h
    src = np.float32([[pad + dx1, pad + dy1], [pad + w + dx2, pad + dy1],
                      [pad + w + dx2, pad + h + dy2], [pad + dx1, pad + h + dy2]])
    src += rng.uniform(-warp, warp, (4, 2)).astype(np.float32) * np.float32([w, h])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    out = cv2.warpPerspective(big, cv2.getPerspectiveTransform(src, dst), (w, h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    amp = 0.3 * strength
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(1 - amp, 1 + amp),
                              beta=rng.uniform(-30 * strength, 30 * strength))
    if rng.random() < prob:  # мелкая пластина издалека: потеря разрешения
        s = rng.uniform(max(0.25, 0.8 - 0.4 * strength), 0.8)
        small = cv2.resize(out, (max(8, int(w * s)), max(8, int(h * s))))
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < prob:
        k = int(rng.choice([3, 5, 7])) if strength > 1 else int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < prob:
        q = int(rng.integers(max(15, 75 - int(60 * strength)), 75))
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


class CropDataset(Dataset):
    def __init__(self, samples: list[dict], h: int, w: int, train: bool, seed: int = 0):
        self.samples, self.h, self.w, self.train = samples, h, w, train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = cv2.imread(str(s["path"]))
        if self.train:
            rng = np.random.default_rng((self.seed, i, int(time.time_ns() % 1_000_000)))
            aug = augment(img, rng, AUG_STRENGTH.get(s["cls"], DEFAULT_STRENGTH))
            fold = UNFOLD.get(s["cls"])
            if fold and fold(aug) is aug:
                aug = img  # искажение сбило раскладку — учим на чистом кропе
            img = aug
        fold = UNFOLD.get(s["cls"])
        if fold:
            img = fold(img)
        x = prep_gray(img, self.h, self.w, settings.preprocess, settings.contrast_boost)
        return torch.from_numpy(x)[None], s["text"], s["cls"]


def collate(batch):
    xs, texts, classes = zip(*batch, strict=True)
    return torch.stack(xs), list(texts), list(classes)


# ---------------------------------------------------------------- обучение
def encode(texts: list[str], charset: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Тексты -> плоский тензор индексов (класс 0 — blank) и длины."""
    idx = [charset.index(ch) + 1 for t in texts for ch in t]
    return torch.tensor(idx, dtype=torch.long), torch.tensor([len(t) for t in texts])


@torch.no_grad()
def evaluate(model, loader, decoder, device) -> dict:
    model.eval()
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for x, texts, classes in loader:
        probs = model(x.to(device))["anpr"].float().softmax(2).cpu().numpy()
        for p, t, c in zip(probs, texts, classes, strict=True):
            hits[c][0] += decoder.greedy(p).text == t
            hits[c][1] += 1
    per = {c: round(ok / n, 4) for c, (ok, n) in sorted(hits.items())}
    macro = sum(per.values()) / len(per) if per else 0.0
    return {"macro": round(macro, 4), "per_class": per,
            "counts": {c: n for c, (_, n) in sorted(hits.items())}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "uz" / "recog")
    ap.add_argument("--weights", type=Path, default=settings.recognizer_weights,
                    help="с какого чекпоинта начинать")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "recog")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="взять N образцов — проверка запуска")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true", help="без fp16 (на CPU выключено и так)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    samples, skipped = load_samples(args.data, args.limit)
    train = [s for s in samples if s["part"] == "train"]
    val = [s for s in samples if s["part"] == "val"]
    if args.limit and not val:  # на крошечной проверке валидация — сам train
        val = train

    # Двухстрочный кроп, который не режется на строки, в работе прочитается как
    # одна строка и не прочитается вовсе. Учить на нём — учить шуму.
    def usable(s: dict) -> bool:
        fold = UNFOLD.get(s["cls"])
        if fold is None:
            return True
        img = cv2.imread(str(s["path"]))
        return img is not None and fold(img) is not img

    before = len(train)
    train = [s for s in train if usable(s)]
    val = [s for s in val if usable(s)]
    print(f"образцов: train {len(train)}, val {len(val)}; "
          f"без раскладки выброшено {before - len(train)} (train); "
          f"пропущено: {dict(skipped) or 'ничего'}")
    print("train по классам:", dict(Counter(s["cls"] for s in train)))
    print("val   по классам:", dict(Counter(s["cls"] for s in val)))
    if not any(s["cls"] not in UNFOLD for s in train):
        print("\n!!! однострочных номеров в обучении нет — модель разучится их читать.\n"
              "!!! Доберите частные и юрлица: tools.fetch_browser --ctype 1 / --ctype 2\n")

    device = torch.device(args.device)
    rec = PlateRecognizer(args.weights, device=str(device), two_line=False)
    model, decoder = rec.model, rec.decoder
    for name, p in model.named_parameters():
        p.requires_grad_(not name.startswith("head_ctype"))
    h, w = rec.info.img_h, rec.info.img_w

    # Каждый класс номеров весит одинаково, сколько бы его ни было в выборке
    cls_n = Counter(s["cls"] for s in train)
    weights = [1.0 / cls_n[s["cls"]] for s in train]
    sampler = WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    dl_train = DataLoader(CropDataset(train, h, w, True, args.seed), batch_size=args.batch,
                          sampler=sampler, num_workers=args.workers, collate_fn=collate,
                          drop_last=len(train) > args.batch)
    dl_val = DataLoader(CropDataset(val, h, w, False), batch_size=args.batch * 2,
                        num_workers=args.workers, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = max(1, args.epochs * len(dl_train))
    warm = min(len(dl_train), steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    args.out.mkdir(parents=True, exist_ok=True)
    from tools import provenance

    provenance.write(args.out / "params.json", {
        "что": "дообучение распознавателя (tools.train_recog)",
        "параметры": vars(args),
        "исходные веса": {"файл": args.weights, "sha256": provenance.sha256(args.weights)},
        "данные": {"gt.txt sha256": provenance.sha256(args.data / "gt.txt"),
                   "train": dict(Counter(s["cls"] for s in train)),
                   "val": dict(Counter(s["cls"] for s in val)),
                   "разбиение": "tools.build_dataset._split: по тексту номера, test = eval-v2"},
        "препроцесс": {"preprocess": settings.preprocess, "contrast_boost": settings.contrast_boost,
                       "двухстрочные": "recognizer.unfold_lines"},
    })
    base = evaluate(model, dl_val, decoder, device)
    print(f"\nдо обучения: {base}", flush=True)
    log = [{"epoch": 0, **base}]
    best = base["macro"]
    torch.save(model.state_dict(), args.out / "best_accuracy.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, texts, _ in dl_train:
            x = x.to(device, non_blocking=True)
            targets, lengths = encode(texts, decoder.charset)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)["anpr"]
            # CTC считаем в fp32: в половинной точности он теряет устойчивость
            log_probs = logits.float().log_softmax(2).permute(1, 0, 2)
            t_len = torch.full((x.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, t_len, lengths)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

        res = evaluate(model, dl_val, decoder, device)
        mark = ""
        if res["macro"] > best:
            best = res["macro"]
            torch.save(model.state_dict(), args.out / "best_accuracy.pth")
            mark = "  <- лучшая"
        log.append({"epoch": epoch, "loss": round(total / max(n, 1), 4), **res})
        (args.out / "metrics.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"эпоха {epoch:3d}  loss {total / max(n, 1):.4f}  macro {res['macro']:.3f}  "
              f"{res['per_class']}  {time.time() - t0:.0f} с{mark}", flush=True)

    best_row = max(log, key=lambda r: r["macro"])
    print(f"\nбыло:  {base['per_class']}  macro {base['macro']}")
    print(f"стало: {best_row['per_class']}  macro {best_row['macro']} (эпоха {best_row['epoch']})")
    print(f"чекпоинт: {args.out / 'best_accuracy.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
