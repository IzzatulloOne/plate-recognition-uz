#!/usr/bin/env bash
# Первичная установка ANPR на чистый Ubuntu 22.04/24.04 (x86_64 или ARM64).
# Идемпотентен: повторный запуск не ломает уже установленное.
#
#   sudo WEIGHTS_URL=https://github.com/.../best_accuracy.pth bash setup.sh
#
# WEIGHTS_URL необязателен. Без него скрипт остановится и подскажет команду scp.

set -euo pipefail

REPO="${REPO:-https://github.com/IzzatulloOne/plate-recognition-uz.git}"
ROOT="${ROOT:-/opt/anpr}"
APP="$ROOT/app"
VENV="$ROOT/venv"
SERVICE_USER="anpr"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[31mОШИБКА: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "запускать через sudo"

# ─────────────────────────────────────────────────────────────── системные пакеты
log "системные пакеты"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-dev git curl nginx ufw

# ────────────────────────────────────────────────────────────────────────── swap
# Пик потребления при установке торча выше, чем в работе. На машине с 4 ГБ
# без swap pip иногда убивается OOM-киллером на распаковке колеса.
if [ ! -f /swapfile ] && [ "$(free -m | awk '/^Mem:/{print $2}')" -lt 6000 ]; then
    log "swap 2 ГБ (мало RAM, страхуемся от OOM при установке)"
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap -q /swapfile
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

# ────────────────────────────────────────────────────────── пользователь и код
id "$SERVICE_USER" &>/dev/null || {
    log "пользователь $SERVICE_USER"
    useradd --system --create-home --home-dir "$ROOT" --shell /usr/sbin/nologin "$SERVICE_USER"
}
mkdir -p "$ROOT"
chown "$SERVICE_USER:$SERVICE_USER" "$ROOT"

if [ -d "$APP/.git" ]; then
    log "обновляю код"
    sudo -u "$SERVICE_USER" git -C "$APP" pull --ff-only
else
    log "клонирую $REPO"
    sudo -u "$SERVICE_USER" git clone --depth 1 "$REPO" "$APP"
fi

# ────────────────────────────────────────────────────────────────── окружение
[ -d "$VENV" ] || {
    log "виртуальное окружение"
    sudo -u "$SERVICE_USER" python3 -m venv "$VENV"
}

# ГЛАВНОЕ МЕСТО ВСЕГО СКРИПТА.
# По умолчанию pip тянет torch со сборкой под CUDA: +4.9 ГБ пакетов nvidia и
# triton, которые на сервере без видеокарты не нужны вообще. На x86 их отсекает
# отдельный индекс pytorch. На ARM их и так нет — там колёса изначально CPU-only.
log "torch (CPU-сборка)"
ARCH="$(uname -m)"
if [ "$ARCH" = "x86_64" ]; then
    sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q --upgrade pip
    sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
    log "  архитектура $ARCH — колёса с PyPI уже без CUDA"
    sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q --upgrade pip
fi

log "остальные зависимости"
sudo -u "$SERVICE_USER" "$VENV/bin/pip" install -q -e "$APP"

# ─────────────────────────────────────────────────────────────────────── веса
log "веса детектора"
sudo -u "$SERVICE_USER" bash -c "cd '$APP' && '$VENV/bin/python' -m tools.fetch_yolo"

if [ ! -f "$APP/best_accuracy.pth" ]; then
    if [ -n "${WEIGHTS_URL:-}" ]; then
        log "скачиваю распознаватель (188 МБ)"
        sudo -u "$SERVICE_USER" curl -fL --progress-bar -o "$APP/best_accuracy.pth" "$WEIGHTS_URL"
    else
        die "нет best_accuracy.pth.
  Залей его как ассет GitHub Release (там лимит 2 ГБ, в отличие от 100 МБ в репозитории)
  и перезапусти с WEIGHTS_URL=..., либо скопируй вручную с локальной машины:

    scp best_accuracy.pth <user>@<ip>:/tmp/
    sudo mv /tmp/best_accuracy.pth $APP/ && sudo chown $SERVICE_USER: $APP/best_accuracy.pth"
    fi
fi
# 188 МБ — если файл заметно меньше, это страница с ошибкой, а не веса
SIZE=$(stat -c%s "$APP/best_accuracy.pth")
[ "$SIZE" -gt 100000000 ] || die "best_accuracy.pth весит $SIZE байт — качалось не то"

# ───────────────────────────────────────────────────────────────────────── .env
if [ ! -f "$APP/.env" ]; then
    log ".env (ключ генерируется случайно)"
    CPUS="$(nproc)"
    API_KEY="$(openssl rand -hex 24)"
    sudo -u "$SERVICE_USER" tee "$APP/.env" >/dev/null <<EOF
# Сгенерировано setup.sh. Доступ закрыт ключом: заголовок X-API-Key.
ANPR_API_KEY=$API_KEY
ANPR_CORS_ORIGINS=*

# Потоки под фактическое число ядер сервера
ANPR_TORCH_THREADS=$CPUS
ANPR_WORKERS=2

# На сервере снапшоты быстро съедают диск. События остаются, картинки нет.
ANPR_SAVE_SNAPSHOTS=false
EOF
    chmod 600 "$APP/.env"
    echo "$API_KEY" > "$ROOT/api_key.txt"
    chmod 600 "$ROOT/api_key.txt"
fi

# ─────────────────────────────────────────────────────────────────── systemd
# Каталог создаём заранее: юнит запускается с ProtectSystem=strict, где писать
# можно только в data/. Создать саму папку под таким режимом приложение уже не смогло бы.
mkdir -p "$APP/data/snapshots"
chown -R "$SERVICE_USER:$SERVICE_USER" "$APP/data"

log "systemd"
sed -e "s|@ROOT@|$ROOT|g" -e "s|@USER@|$SERVICE_USER|g" \
    "$APP/deploy/anpr.service" > /etc/systemd/system/anpr.service
systemctl daemon-reload
systemctl enable --now anpr

# ───────────────────────────────────────────────────────────────────── nginx
log "nginx"
cp "$APP/deploy/nginx.conf" /etc/nginx/sites-available/anpr
ln -sf /etc/nginx/sites-available/anpr /etc/nginx/sites-enabled/anpr
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# ─────────────────────────────────────────────────────────────────────── ufw
log "firewall"
ufw allow OpenSSH >/dev/null
ufw allow 80/tcp >/dev/null
ufw --force enable >/dev/null

# ─────────────────────────────────────────────────────────────────── проверка
log "жду подъёма (модели грузятся ~11 с, на слабом vCPU до минуты)"
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8000/healthz >/dev/null 2>&1; then
        printf '\n\033[32mГотово.\033[0m\n\n'
        curl -s http://127.0.0.1:8000/healthz; echo
        IP="$(curl -s -m 5 ifconfig.me || echo '<ip>')"
        cat <<EOF

  Ключ:     $(cat "$ROOT/api_key.txt")
  Проверка: curl -X POST http://$IP/v1/recognize \\
              -H "X-API-Key: \$(cat $ROOT/api_key.txt)" -F "file=@car.jpg"
  Логи:     journalctl -u anpr -f
  Обновить: bash $APP/deploy/update.sh
EOF
        exit 0
    fi
    sleep 2
done
die "сервис не поднялся за 2 минуты, смотри: journalctl -u anpr -n 50"
