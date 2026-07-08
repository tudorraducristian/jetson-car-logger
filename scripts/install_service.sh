#!/usr/bin/env bash
# Install and enable the car-logger systemd service + daily-restart timer.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../deployment" && pwd)"
sudo cp "$SRC/car-logger.service" /etc/systemd/system/
sudo cp "$SRC/car-logger-restart.service" /etc/systemd/system/
sudo cp "$SRC/car-logger-restart.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now car-logger.service
sudo systemctl enable --now car-logger-restart.timer
sudo systemctl status car-logger.service --no-pager
