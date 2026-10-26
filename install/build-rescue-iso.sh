#!/usr/bin/env bash
# Собирает спасательный ISO для rig: без клавиатуры включает сеть по USB4 с Mac, задаёт пароль al,
# пишет диагностику и выключает ПК. Диск не форматирует (см. rescue.template.yaml).
# Запуск на Mac:  install/build-rescue-iso.sh [исходный.iso]   (нужны xorriso и openssl)
# Пароль al: install/rescue.env (в .gitignore); если файла нет — создаётся со случайным паролем.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
dl="$HOME/Downloads/ubuntu-iso"
src="${1:-$(ls -t "$dl"/ubuntu-26.10*-live-server-amd64.iso | head -1)}"
out="$dl/rig-rescue-$(basename "$src" | sed -E 's/^ubuntu-([0-9.]+(-beta)?)-.*/\1/').iso"
gh_user="${GH_USER:-sanasol}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

env_file="$here/rescue.env"
[[ -f "$env_file" ]] || { echo "RIG_PASSWORD=$(openssl rand -base64 12 | tr -d '/+=')" >"$env_file"; chmod 600 "$env_file"; }
# shellcheck source=/dev/null
. "$env_file"
hash="$(openssl passwd -6 "$RIG_PASSWORD")"

{ cat "$HOME/.ssh/id_rsa.pub"; curl -fsS "https://github.com/$gh_user.keys"; echo; } \
  | grep -E '^ssh-' | awk '!seen[$2]++' >"$work/authorized_keys"
[[ -s "$work/authorized_keys" ]] || { echo "нет SSH-ключей"; exit 1; }

sed "s|@PASS_HASH@|$hash|" "$here/rescue.template.yaml" >"$work/autoinstall.yaml"

xorriso -osirrox on -indev "$src" -extract /boot/grub/grub.cfg "$work/grub.orig" 2>/dev/null
chmod u+w "$work/grub.orig"
{
  echo 'set timeout=3'
  echo 'set default=0'
  echo 'menuentry "RESCUE rig: USB4 network + password (no disk wipe)" {'
  echo '	set gfxpayload=keep'
  echo '	linux	/casper/vmlinuz ---'
  echo '	initrd	/casper/initrd'
  echo '}'
  grep -v '^set timeout=' "$work/grub.orig"
} >"$work/grub.cfg"

rm -f "$out"
xorriso -indev "$src" -outdev "$out" \
  -map "$work/autoinstall.yaml" /autoinstall.yaml \
  -map "$work/grub.cfg" /boot/grub/grub.cfg \
  -map "$here/rescue.sh" /rig/rescue.sh \
  -map "$work/authorized_keys" /rig/authorized_keys \
  -boot_image any replay 2>&1 | grep -E 'Writing to|FAILURE|SORRY' || true

echo "готово: $out"
echo "пароль al: в $env_file"
