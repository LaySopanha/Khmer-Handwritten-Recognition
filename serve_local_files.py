#!/usr/bin/env python3
"""
Small helper HTTP server with permissive CORS headers for feeding local images
to Label Studio. Similar to LayoutLMv3 project helper.
"""

import argparse
import os
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Tuple


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that adds wildcard CORS headers."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a directory over HTTP for Label Studio local files."
    )
    parser.add_argument(
        "--directory",
        "-d",
        default=".",
        help="Directory to expose (defaults to current working directory).",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host/IP to bind (default: 0.0.0.0)."
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8080, help="Port to listen on (default: 8080)."
    )
    return parser.parse_args()


def run_server(bind: Tuple[str, int], directory: str) -> None:
    os.chdir(directory)
    handler = partial(CORSRequestHandler, directory=".")
    httpd = HTTPServer(bind, handler)
    host, port = bind
    print(f"Serving {os.path.abspath(directory)} at http://{host}:{port}/ (Ctrl+C to stop)")
    httpd.serve_forever()


def main() -> None:
    args = parse_args()
    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"{directory} is not a directory")
    run_server((args.host, args.port), directory)


if __name__ == "__main__":
    main()
