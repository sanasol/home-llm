#!/bin/sh
# Лимиты мощности по модели карты (индексы плавают, особенно у OCuLink): 5090 -> 460 W, 3090 -> 300 W.
nvidia-smi -pm 1 >/dev/null
nvidia-smi --query-gpu=uuid,name --format=csv,noheader | while IFS=, read -r uuid name; do
  case "$name" in
    *5090*) nvidia-smi -i "$uuid" -pl 460 ;;
    *3090*) nvidia-smi -i "$uuid" -pl 300 ;;
  esac
done
