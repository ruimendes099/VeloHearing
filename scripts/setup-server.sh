#!/usr/bin/env bash
# Rebuild the VeloHearing orchestration server on a fresh Ubuntu 24.04 host.
# Run as root. Safe to re-run.
#
# Does NOT set rui's password, install SSH keys or change sshd_config —
# do those by hand, and only disable root/password login after key login works.
set -euo pipefail

USER_NAME="${USER_NAME:-rui}"
REPO_URL="${REPO_URL:-https://github.com/ruimendes099/VeloHearing.git}"
ADMIN_IPS="${ADMIN_IPS:-}"   # space-separated IPs fail2ban should never ban

[[ $EUID -eq 0 ]] || { echo "run as root" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "== packages"
apt-get update -qq
apt-get install -y -qq git ffmpeg python3-venv python3-pip build-essential htop \
    ufw fail2ban unattended-upgrades

echo "== user $USER_NAME"
id "$USER_NAME" &>/dev/null || adduser --disabled-password --gecos "" "$USER_NAME"
usermod -aG sudo "$USER_NAME"
install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "/home/$USER_NAME/.ssh"
[[ -f /home/$USER_NAME/.ssh/authorized_keys ]] || \
    install -m 600 -o "$USER_NAME" -g "$USER_NAME" /dev/null "/home/$USER_NAME/.ssh/authorized_keys"

echo "== firewall"
ufw allow 22/tcp comment 'SSH'
ufw default deny incoming
ufw default allow outgoing
ufw --force enable

echo "== fail2ban"
# Ubuntu 24.04 logs sshd under ssh.service, not sshd.service
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime  = 1h
findtime = 10m
maxretry = 5
backend  = systemd
ignoreip = 127.0.0.1/8 ::1 $ADMIN_IPS

[sshd]
enabled = true
journalmatch = _SYSTEMD_UNIT=ssh.service + _COMM=sshd
EOF
systemctl enable fail2ban
systemctl restart fail2ban

echo "== automatic security updates"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
systemctl enable --now unattended-upgrades

echo "== repo + venv"
REPO_DIR="/home/$USER_NAME/VeloHearing"
[[ -d $REPO_DIR/.git ]] || sudo -u "$USER_NAME" git clone -q "$REPO_URL" "$REPO_DIR"
sudo -u "$USER_NAME" python3 -m venv "$REPO_DIR/.venv"
sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$USER_NAME" "$REPO_DIR/.venv/bin/pip" install -q -r "$REPO_DIR/requirements.txt"

echo "== done"
echo "next: set $USER_NAME's password, add SSH key, verify key login, then harden sshd"
