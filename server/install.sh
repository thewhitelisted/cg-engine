#!/bin/sh
# Server setup, run with sudo (safe to rerun). Caddy serves the site from serve.py's folder ($DIR: the page, web/app.js,
# api/state.json) with automatic HTTPS, proxies the push stream (api/events) to serve.py on 127.0.0.1:$API_PORT, and lets
# the page's own domain call the API. The page's domain ($SITE) is served here once its DNS points at this server
# (until then it stays on GitHub Pages and calls the API on $HOST). systemd keeps serve.py running (cg-live.service).
#
#     sudo sh server/install.sh            # Caddy only
#     sudo sh server/install.sh service    # Caddy, then install and (re)start the publisher
set -eu
HOST=live.christopherjlee.ca
SITE=cg-531d2447.christopherjlee.ca
IP=13.140.225.13
DIR=/srv/cg-live
API_PORT=8787
HERE=$(cd "$(dirname "$0")" && pwd)

points_here() { [ "$(dig +short A "$1" | tail -n 1)" = "$IP" ]; }
if ! points_here "$HOST"; then
	echo "DNS for $HOST doesn't point at $IP yet. Wait a few minutes after saving the Porkbun record and rerun."
	exit 1
fi
HOSTS="$HOST"
if points_here "$SITE"; then
	HOSTS="$HOST, $SITE"
else
	echo "Note: $SITE still points elsewhere (GitHub Pages); serving only $HOST. Rerun after its DNS change."
fi

command -v caddy >/dev/null || { apt-get update -qq && apt-get install -y caddy; }
install -d -o chris -g chris -m 755 "$DIR"

cat > /etc/caddy/Caddyfile <<EOF
$HOSTS {
	root * $DIR
	@events path /api/events*
	handle @events {
		reverse_proxy 127.0.0.1:$API_PORT {
			flush_interval -1
		}
	}
	handle {
		encode zstd gzip
		file_server {
			hide .*
		}
	}
	@api path /api/*
	@static not path /api/*
	header {
		Access-Control-Allow-Origin https://$SITE
		-Server
	}
	header @api Cache-Control no-store
	header @static Cache-Control no-cache
}
EOF
caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null
ufw allow 80/tcp >/dev/null
ufw allow 443/tcp >/dev/null
systemctl reload caddy || systemctl restart caddy
echo "Caddy: https://${HOSTS%%,*} serves $DIR; api/events -> 127.0.0.1:$API_PORT (sites: $HOSTS)"

if [ "${1:-}" = "service" ]; then
	install -m 644 "$HERE/cg-live.service" /etc/systemd/system/cg-live.service
	systemctl daemon-reload
	systemctl enable --now cg-live
	systemctl restart cg-live
	echo "Publisher: running (journalctl -u cg-live -f to watch)"
fi
