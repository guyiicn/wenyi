"""Live-server regressions for silent-after-body and trickle LLM hangs.

A mock origin that accepts TLS or plaintext, reads the whole request, then neither
writes response headers nor closes the socket used to hang v1255: join waited on
SSLSocket.read and client.close() never ran. These tests use no real API keys.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from wenyi_core.config import Config
from wenyi_core.llm.factory import build_client
from wenyi_core.llm.retrying import ProviderRequestError

OK_BODY = json.dumps(
    {
        "id": "local",
        "object": "chat.completion",
        "created": 0,
        "model": "x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()

_LARGE_BODY = "x" * (400 * 1024)
_STALL_SERVER = r"""
from __future__ import annotations

import json
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

OK_BODY = json.dumps(
    {
        "id": "local",
        "object": "chat.completion",
        "created": 0,
        "model": "x",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()


def read_http_request(conn: socket.socket) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return buf
        buf += chunk
    header, rest = buf.split(b"\r\n\r\n", 1)
    length = 0
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    body = rest
    while len(body) < length:
        chunk = conn.recv(min(65536, length - len(body)))
        if not chunk:
            break
        body += chunk
    return header + b"\r\n\r\n" + body


def main() -> None:
    mode, count_path = sys.argv[1], Path(sys.argv[2])
    certfile = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    keyfile = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(64)
    sys.stdout.write(f"{server.getsockname()[1]}\n")
    sys.stdout.flush()
    ctx = None
    if certfile and keyfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
    lock = threading.Lock()

    def handle(conn: socket.socket) -> None:
        try:
            if ctx is not None:
                conn = ctx.wrap_socket(conn, server_side=True)
            read_http_request(conn)
            with lock:
                current = int(count_path.read_text(encoding="utf-8") or "0")
                count_path.write_text(str(current + 1), encoding="utf-8")
            if mode == "ok":
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    + f"Content-Length: {len(OK_BODY)}\r\n\r\n".encode()
                    + OK_BODY
                )
                return
            if mode == "trickle":
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                while True:
                    conn.sendall(b"1\r\n \r\n")
                    time.sleep(1)
                return
            while True:
                time.sleep(60)
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
"""


def _count_fds() -> int:
    fd_dir = Path("/proc/self/fd")
    if fd_dir.is_dir():
        return len(list(fd_dir.iterdir()))
    return -1


def _make_tls_files(tmp_path: Path) -> tuple[Path, Path]:
    cert = tmp_path / "stall-cert.pem"
    key = tmp_path / "stall-key.pem"
    subprocess.check_call(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-nodes",
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert, key


def _start_stall_server(
    tmp_path: Path, mode: str, *, tls: bool
) -> tuple[subprocess.Popen[str], int, Path, Path | None]:
    count_path = tmp_path / f"stall-{mode}-{'tls' if tls else 'plain'}-count"
    count_path.write_text("0", encoding="utf-8")
    cert = key = None
    args = [sys.executable, "-c", _STALL_SERVER, mode, str(count_path)]
    if tls:
        cert, key = _make_tls_files(tmp_path)
        args.extend([str(cert), str(key)])
    else:
        args.extend(["", ""])
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdout is not None
    port = int(proc.stdout.readline().strip())
    return proc, port, count_path, cert


def _make_client(url: str, *, timeout: float, max_retries: int, max_concurrency: int = 2):
    os.environ["WENYI_STALL_KEY"] = "local-dummy-not-a-real-key"
    cfg = Config.from_dict(
        {
            "language": {"source": "en", "target": "zh"},
            "llm": {
                "providers": {
                    "stall": {
                        "kind": "openai-compatible",
                        "base_url": url,
                        "api_key_env": "WENYI_STALL_KEY",
                        "timeout": timeout,
                        "max_retries": max_retries,
                        "max_concurrency": max_concurrency,
                    }
                },
                "models": {"m": {"provider": "stall", "model": "x", "max_output_tokens": 32}},
                "tiers": {"strong": "m", "cheap": "m", "fast": "m"},
            },
            "paths": {"state_dir": "/tmp/wenyi-silent-after-body-state"},
        }
    )
    return build_client(cfg, credentials={"stall": "local-dummy-not-a-real-key"})


def _call(client, body: str) -> None:
    client.complete([{"role": "user", "content": body}], operation="polish.body")


@pytest.mark.parametrize("tls", [False, True], ids=["plain", "tls"])
def test_silent_after_body_large_request_times_out_and_retries(
    tmp_path: Path, tls: bool, monkeypatch: pytest.MonkeyPatch
):
    """Read the body, write nothing, keep the socket. Application timeout must fire."""
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "silent", tls=tls)
        if tls:
            assert cert is not None
            monkeypatch.setenv("SSL_CERT_FILE", str(cert))
            monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        scheme = "https" if tls else "http"
        client = _make_client(f"{scheme}://127.0.0.1:{port}/v1", timeout=1.0, max_retries=1)
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            _call(client, _LARGE_BODY)
        elapsed = time.monotonic() - started
        assert elapsed < 8.0, f"silent-after-body hung for {elapsed:.2f}s"
        assert elapsed >= 1.0
        error = caught.value
        assert error.reason == "timeout"
        assert error.resumable is True
        accepts = int(count_path.read_text(encoding="utf-8"))
        assert accepts == 2, f"expected initial attempt plus one retry, got {accepts}"
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize("tls", [False, True], ids=["plain", "tls"])
def test_silent_after_body_concurrency_two_does_not_hang(
    tmp_path: Path, tls: bool, monkeypatch: pytest.MonkeyPatch
):
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "silent", tls=tls)
        if tls:
            assert cert is not None
            monkeypatch.setenv("SSL_CERT_FILE", str(cert))
            monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        scheme = "https" if tls else "http"
        client = _make_client(
            f"{scheme}://127.0.0.1:{port}/v1",
            timeout=1.0,
            max_retries=0,
            max_concurrency=2,
        )
        errors: list[BaseException | None] = [None, None]
        started = time.monotonic()

        def run(index: int) -> None:
            try:
                _call(client, _LARGE_BODY)
            except BaseException as error:
                errors[index] = error

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=8.0)
        elapsed = time.monotonic() - started
        assert all(not thread.is_alive() for thread in threads), "concurrent request hung"
        assert elapsed < 8.0, f"concurrency-2 hung for {elapsed:.2f}s"
        assert all(isinstance(error, ProviderRequestError) for error in errors)
        assert all(error.reason == "timeout" and error.resumable for error in errors)  # type: ignore[union-attr]
        assert int(count_path.read_text(encoding="utf-8")) == 2
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.parametrize("mode", ["silent", "trickle"])
def test_stall_modes_converge_threads_and_fds(tmp_path: Path, mode: str):
    """Keep the previous silent/trickle convergence contract: no thread or fd growth."""
    proc = None
    try:
        proc, port, _count_path, _cert = _start_stall_server(tmp_path, mode, tls=False)
        client = _make_client(f"http://127.0.0.1:{port}/v1", timeout=1.0, max_retries=0)
        with pytest.raises(ProviderRequestError):
            _call(client, "hi")
        time.sleep(0.4)
        threads_before = threading.active_count()
        fds_before = _count_fds()
        for _ in range(5):
            with pytest.raises(ProviderRequestError):
                _call(client, "hi")
        time.sleep(0.4)
        threads_after = threading.active_count()
        fds_after = _count_fds()
        llm_threads = sum(1 for thread in threading.enumerate() if thread.name == "llm-request")
        assert threads_after <= threads_before + 1
        assert llm_threads == 0
        if fds_before >= 0 and fds_after >= 0:
            assert fds_after <= fds_before + 2
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
