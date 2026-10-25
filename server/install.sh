#!/bin/sh
# Server setup, run with sudo (safe to rerun): Caddy serves serve.py's output folder at https://$HOST (certificate
# handled by Caddy) to the public page's origin, and systemd keeps serve.py running (server/cg-live.service).
#
#     sudo sh server/install.sh            # Caddy only
#     sudo sh server/install.sh service    # Caddy, then install and start the publisher
set -eu
HOST=live.christopherjlee.ca
IP=13.140.225.13
ORIGIN=https://cg-531d2447.christopherjlee.ca
DIR=/srv/cg-live
HERE=$(cd "$(dirname "$0")" && pwd)

if [ "$(dig +short A "$HOST" | tail -n 1)" != "$IP" ]; then
	echo "DNS for $HOST doesn't point at $IP yet. Wait a few minutes after saving the Porkbun record and rerun."
	exit 1
fi

command -v caddy >/dev/null || { apt-get update -qq && apt-get install -y caddy; }
install -d -o chris -g chris -m 755 "$DIR"

cat > /etc/caddy/Caddyfile <<EOF
$HOST {
	root * $DIR
	file_server {
		hide .*
	}
	encode zstd gzip
	header {
		Access-Control-Allow-Origin $ORIGIN
		Cache-Control no-store
		-Server
	}
}
EOF
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
systemctl reload caddy || systemctl restart caddy
echo "Caddy: https://$HOST serves $DIR"

if [ "${1:-}" = "service" ]; then
	install -m 644 "$HERE/cg-live.service" /etc/systemd/system/cg-live.service
	systemctl daemon-reload
	systemctl enable --now cg-live
	systemctl restart cg-live
	echo "Publisher: running (journalctl -u cg-live -f to watch)"
fi
