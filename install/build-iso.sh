#!/usr/bin/env bash
# Собирает ISO для установки rig без экрана и кабеля: autoinstall.yaml + Wi-Fi + меню GRUB с автозапуском.
# Запуск на Mac:  install/build-iso.sh [исходный.iso]   (нужны xorriso и python3: brew install xorriso)
# Wi-Fi берётся из install/wifi.env (в .gitignore), пакеты wpasupplicant — из $DEBS.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
dl="$HOME/Downloads/ubuntu-iso"
src="${1:-$(ls -t "$dl"/ubuntu-26.10*-live-server-amd64.iso | head -1)}"
debs="${DEBS:-$dl/rig-debs-26.10}"
out="$dl/rig-autoinstall-$(basename "$src" | sed -E 's/^ubuntu-([0-9.]+(-beta)?)-.*/\1/').iso"
gh_user="${GH_USER:-sanasol}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Контрольная сумма с releases.ubuntu.com.
ver="$(basename "$src" | sed -E 's/^ubuntu-([0-9]+\.[0-9]+).*/\1/')"
curl -fsS "https://releases.ubuntu.com/$ver/SHA256SUMS" | grep -F "$(basename "$src")" | sed 's/ \*/  /' \
  | (cd "$(dirname "$src")" && shasum -a 256 -c --quiet)

ls "$debs"/wpasupplicant_*.deb "$debs"/libpcsclite1_*.deb "$debs"/iw_*.deb >/dev/null

# shellcheck source=/dev/null
. "$here/wifi.env"
: "${WIFI_SSIDS:?}" "${WIFI_PASSWORD:?}" "${WIFI_COUNTRY:?}"
WIFI_SSIDS="$WIFI_SSIDS" WIFI_PASSWORD="$WIFI_PASSWORD" python3 -I -c '
import json, os
pw = json.dumps(os.environ["WIFI_PASSWORD"])
for s in dict.fromkeys(os.environ["WIFI_SSIDS"].split("|")):
    print("          %s: {password: %s}" % (json.dumps(s), pw))
' >"$work/wifi"

# Ключи: локальный id_rsa.pub + опубликованные на GitHub, без дублей.
{ cat "$HOME/.ssh/id_rsa.pub"; curl -fsS "https://github.com/$gh_user.keys"; echo; } \
  | grep -E '^ssh-' | awk '!seen[$2]++' | sed 's/^/          - /' >"$work/keys"
[[ -s "$work/keys" ]] || { echo "нет SSH-ключей"; exit 1; }

awk -v keys="$work/keys" -v wifi="$work/wifi" -v cc="$WIFI_COUNTRY" '
  /^@SSH_KEYS@$/ { while ((getline l < keys) > 0) print l; next }
  /^@WIFI@$/     { while ((getline l < wifi) > 0) print l; next }
  { gsub(/@WIFI_COUNTRY@/, cc); print }
' "$here/autoinstall.template.yaml" >"$work/autoinstall.yaml"

xorriso -osirrox on -indev "$src" -extract /boot/grub/grub.cfg "$work/grub.orig" 2>/dev/null
chmod u+w "$work/grub.orig"
{
  echo 'set timeout=5'
  echo 'set default=0'
  echo 'menuentry "AUTOINSTALL rig (Wi-Fi) - wipes Samsung 990 PRO" {'
  echo '	set gfxpayload=keep'
  echo "	linux	/casper/vmlinuz autoinstall cfg80211.ieee80211_regdom=$WIFI_COUNTRY ---"
  echo '	initrd	/casper/initrd'
  echo '}'
  grep -v '^set timeout=' "$work/grub.orig"
} >"$work/grub.cfg"

rm -f "$out"
xorriso -indev "$src" -outdev "$out" \
  -map "$work/autoinstall.yaml" /autoinstall.yaml \
  -map "$work/grub.cfg" /boot/grub/grub.cfg \
  -map "$here/rig-postinstall.sh" /rig/rig-postinstall.sh \
  -map "$debs" /rig/debs \
  -boot_image any replay 2>&1 | grep -E 'Writing to|FAILURE|SORRY' || true

echo "готово: $out"
echo "ключей: $(wc -l <"$work/keys" | tr -d ' '), вариантов SSID: $(wc -l <"$work/wifi" | tr -d ' ')"
