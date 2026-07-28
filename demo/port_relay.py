#!/usr/bin/env python3
"""Dependency-free TCP relay: login-node 127.0.0.1:LPORT -> TARGET_HOST:TPORT.

The chat server runs on a compute node (see serve.sh), but VS Code's port
forwarding and the Simple Browser only see the login node's localhost. Run this
on the login node to bridge the two:

    python demo/port_relay.py 8000 pcl-zen4 8000

Then open http://localhost:8000 (VS Code will offer to forward port 8000).
"""
import socket
import sys
import threading

LPORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
THOST = sys.argv[2] if len(sys.argv) > 2 else "localhost"
TPORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8000


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def _handle(client: socket.socket) -> None:
    try:
        upstream = socket.create_connection((THOST, TPORT))
    except OSError as exc:
        client.close()
        print(f"relay: upstream {THOST}:{TPORT} failed: {exc}", flush=True)
        return
    threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
    threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", LPORT))
    srv.listen(64)
    print(f"relay: http://127.0.0.1:{LPORT}  ->  {THOST}:{TPORT}  (Ctrl+C to stop)",
          flush=True)
    try:
        while True:
            client, _ = srv.accept()
            _handle(client)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
