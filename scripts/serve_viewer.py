"""Serve only viewer assets and generated results on the loopback interface."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        resolved=(ROOT/path.lstrip('/')).resolve()
        allowed=any(resolved.is_relative_to(ROOT/name) for name in ('viewer','results'))
        if not allowed:
            self.send_error(404);return
        super().do_GET()

    def do_HEAD(self):
        path=unquote(urlsplit(self.path).path)
        resolved=(ROOT/path.lstrip('/')).resolve()
        if not any(resolved.is_relative_to(ROOT/name) for name in ('viewer','results')):
            self.send_error(404);return
        super().do_HEAD()

if __name__=='__main__':
    print('Open http://127.0.0.1:8765/viewer/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8765),partial(Handler,directory=str(ROOT))).serve_forever()
