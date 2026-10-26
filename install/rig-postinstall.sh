#!/usr/bin/env bash
# Донастройка rig после autoinstall (Ubuntu 26.10). Autoinstall кладёт копию в /opt/rig/.
# Запуск на rig:  sudo bash /opt/rig/rig-postinstall.sh
# Можно запускать повторно.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "запускать через sudo"; exit 1; }
. /etc/os-release
[[ "$VERSION_ID" == "26.10" ]] || { echo "ожидалась Ubuntu 26.10, а тут $VERSION_ID"; exit 1; }

TARGET_USER="${SUDO_USER:-al}"
NVIDIA_BRANCH="${NVIDIA_BRANCH:-580}"   # open-драйвер: 5090 требует open, 3090 (Ampere) с ним тоже работает
export DEBIAN_FRONTEND=noninteractive

echo "== обновления"
apt-get update
apt-get -y full-upgrade

echo "== драйвер NVIDIA $NVIDIA_BRANCH-open (подписанные модули из архива Ubuntu)"
if ! nvidia-smi >/dev/null 2>&1; then
  apt-get install -y "nvidia-driver-$NVIDIA_BRANCH-open" \
    "linux-modules-nvidia-$NVIDIA_BRANCH-open-generic"
  need_reboot=1
fi
systemctl enable nvidia-persistenced 2>/dev/null || true

echo "== uv"
sudo -u "$TARGET_USER" -i sh -c 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'

echo "== /models"
mkdir -p /models
chown "$TARGET_USER:$TARGET_USER" /models

echo "== Tailscale"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh

sensors-detect --auto >/dev/null 2>&1 || true

echo
[[ -n "${need_reboot:-}" ]] && echo "Драйвер поставлен — нужен sudo reboot."
echo "Дальше:"
echo "  nvidia-smi -L                 # UUID карт -> docs/03-software.md"
echo "  sudo tailscale up --ssh       # имя машины: rig"
