from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


def main():
    host = "0.0.0.0"
    port = int(__import__("os").environ.get("EEG_DEMO_HTTP_PORT", "51730"))
    TCPServer.allow_reuse_address = True

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    with TCPServer((host, port), Handler) as httpd:
        print(f"HTTP server on http://{host}:{port} (serving ./frontend)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
