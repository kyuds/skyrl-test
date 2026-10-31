#!/bin/bash
# VM startup script (runs as root at every boot; from Charlie's guide, made idempotent): raise the open-file
# limits Ray and vLLM need. They apply to sessions opened after the first reboot, which 02_setup_node.sh does.
add() { grep -qxF "$1" "$2" 2>/dev/null || echo "$1" >> "$2"; }
add "DefaultLimitNOFILE=infinity" /etc/systemd/system.conf
add "DefaultLimitNOFILE=infinity" /etc/systemd/user.conf
add "*                soft    nofile          unlimited" /etc/security/limits.conf
add "*                hard    nofile          unlimited" /etc/security/limits.conf
