#!/usr/bin/env bash
# Обновление до свежего коммита с откатом, если сервис не поднялся.
#
#   sudo bash /opt/anpr/app/deploy/update.sh
#
# Именно этот скрипт позже станет инструментом агента: он делает ровно одно
# понятное действие, возвращает внятный код возврата и сам чинит за собой.

set -euo pipefail

ROOT="${ROOT:-/opt/anpr}"
APP="$ROOT/app"
VENV="$ROOT/venv"
SERVICE_USER="anpr"
HEALTH_TIMEOUT=120

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "запускать через sudo"

wait_healthy() {
    for _ in $(seq 1 $((HEALTH_TIMEOUT / 2))); do
        curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1 && return 0
        sleep 2
    done
    return 1
}

BEFORE="$(sudo -u "$SERVICE_USER" git -C "$APP" rev-parse HEAD)"
log "текущий коммит: ${BEFORE:0:8}"

sudo -u "$SERVICE_USER" git -C "$APP" fetch --quiet origin
AFTER="$(sudo -u "$SERVICE_USER" git -C "$APP" rev-parse origin/main)"

if [ "$BEFORE" = "$AFTER" ]; then
    log "обновлений нет"
    wait_healthy || die "сервис не отвечает, хотя код не менялся: journalctl -u anpr -n 50"
    exit 0
fi

log "обновляю до ${AFTER:0:8}"
sudo -u "$SERVICE_USER" git -C "$APP" merge --ff-only origin/main

# Зависимости переставляем только если менялся pyproject: pip -e занимает минуты,
# а девяносто девять правок из ста его не затрагивают.
if ! sudo -u "$SERVICE_USER" git -C "$APP" diff --quiet "$BEFORE" "$AFTER" -- pyproject.toml; then
    log "pyproject изменился, обновляю зависимости"
    sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q -e "$APP"
fi

log "перезапуск"
systemctl restart anpr

if wait_healthy; then
    printf '\n\033[32mОбновлено: %s -> %s\033[0m\n' "${BEFORE:0:8}" "${AFTER:0:8}"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────── откат
printf '\n\033[31mСервис не поднялся за %s с. Откатываюсь.\033[0m\n' "$HEALTH_TIMEOUT" >&2
journalctl -u anpr -n 30 --no-pager >&2

sudo -u "$SERVICE_USER" git -C "$APP" reset --hard "$BEFORE"
sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q -e "$APP"
systemctl restart anpr

if wait_healthy; then
    die "откат на ${BEFORE:0:8} удался, сервис работает на прежней версии"
fi
die "откат не помог, сервис лежит. Смотри journalctl -u anpr -n 100"
