"""Метки моделей машин с platesmania -> классы уровня «Nexia, Cobalt, Malibu».

Сайт пишет одну и ту же машину по-разному: Ravon R4 — это Cobalt, Ravon Gentra
и Chevrolet Gentra — Lacetti, Nexia бывает и Daewoo, и Chevrolet. Поколения и
комплектации не различаем намеренно — нужна модель, а не паспорт машины.

Правила:
- узбекские GM (Chevrolet, Daewoo, Ravon) — имя модели без марки;
- Жигули 2101–2107 — один класс: на фото их и человек не всегда различит;
- прицепы — по марке (Schmitz, Krone, Kögel), модель прицепа сзади не видна;
- остальное — «Марка Модель» без уточнений в скобках: «Lada (VAZ) Largus» ->
  «Lada Largus».
"""

from __future__ import annotations

import re

GM_BRANDS = ("Chevrolet", "Daewoo", "Ravon")

#: синонимы внутри GM: регулярка по имени модели (без марки) -> класс
_GM_ALIASES = (
    (re.compile(r"^(Cobalt|R4)\b"), "Cobalt"),
    (re.compile(r"^(Lacetti|Gentra)\b"), "Lacetti"),
    (re.compile(r"^Nexia\b"), "Nexia"),
    (re.compile(r"^(Spark|R2)\b"), "Spark"),
    (re.compile(r"^Matiz\b"), "Matiz"),
)
_ZHIGULI = re.compile(r"^Lada \(VAZ\) 210[1-7]\b")
_PARENS = re.compile(r"\s*\([^)]*\)")
_NO_VEHICLE = "License plate without vehicle"


def canonical(raw: str, trailer: bool = False) -> str:
    """Исходная строка сайта -> класс модели. Пустая строка — метки нет."""
    raw = " ".join((raw or "").split())
    if not raw or raw.startswith(_NO_VEHICLE):
        return ""
    if trailer:
        return raw.split()[0]
    if _ZHIGULI.match(raw):
        return "Жигули"
    brand, _, rest = raw.partition(" ")
    if brand in GM_BRANDS and rest:
        for pattern, name in _GM_ALIASES:
            if pattern.match(rest):
                return name
        return _PARENS.sub("", rest).strip()
    return _PARENS.sub("", raw).strip()
