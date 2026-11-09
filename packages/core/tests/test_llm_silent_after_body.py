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
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from wenyi_core.agents.review_actions import ReviewActionLoop
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
                current = int(count_path.read_text(encoding="utf-8") or "0") + 1
                count_path.write_text(str(current), encoding="utf-8")
            if (
                mode == "ok"
                or (mode == "ok_then_silent" and current == 1)
                or (mode == "ok_two_then_silent" and current <= 2)
            ):
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


def _make_client(
    url: str,
    *,
    timeout: float,
    max_retries: int,
    max_concurrency: int = 2,
    max_output_tokens: int = 32,
):
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
                "models": {
                    "m": {
                        "provider": "stall",
                        "model": "x",
                        "max_output_tokens": max_output_tokens,
                    }
                },
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


def _call_json(client, body: str, operation: str) -> str:
    return client.complete(
        [{"role": "user", "content": body}],
        operation=operation,
        json_mode=True,
    )


def _seed_cached_json_object(client) -> None:
    """Match production: json_mode_source=cached, json_object, then the hang."""
    from wenyi_core.llm.providers.openai_compatible import OpenAICompatibleOptions

    adapter = client.adapter("stall")
    key = (
        "x",
        json.dumps(OpenAICompatibleOptions().model_dump(mode="json"), sort_keys=True),
    )
    adapter._json_modes[key] = "json_object"


def test_json_object_scan_times_out_while_verify_retries(tmp_path: Path):
    """Concurrency 2: review.verify times out and retries while a large cached
    json_object review.scan is in flight. Both must finish inside the retry bound.

    Production v1257: verify's APITimeoutError retried; scan emitted
    llm_json_mode_selected (cached json_object) and then hung 21 minutes with no
    httpx line, no watchdog, and no attempt 2.
    """
    proc = None
    try:
        proc, port, count_path, _cert = _start_stall_server(tmp_path, "silent", tls=False)
        client = _make_client(
            f"http://127.0.0.1:{port}/v1",
            timeout=1.0,
            max_retries=1,
            max_concurrency=2,
        )
        _seed_cached_json_object(client)
        errors: list[BaseException | None] = [None, None]

        def verify() -> None:
            try:
                _call_json(client, "small verify", "review.verify")
            except BaseException as error:
                errors[0] = error

        def scan() -> None:
            time.sleep(0.15)
            try:
                _call_json(client, _LARGE_BODY, "review.scan")
            except BaseException as error:
                errors[1] = error

        threads = [
            threading.Thread(target=verify, name="review-verify"),
            threading.Thread(target=scan, name="review-scan"),
        ]
        started = time.monotonic()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=12.0)
        elapsed = time.monotonic() - started
        assert all(not thread.is_alive() for thread in threads), (
            f"json_object concurrent request hung after {elapsed:.2f}s"
        )
        assert elapsed < 12.0, f"json_object scan/verify hung for {elapsed:.2f}s"
        assert all(isinstance(error, ProviderRequestError) for error in errors)
        assert all(error.reason == "timeout" and error.resumable for error in errors)  # type: ignore[union-attr]
        assert int(count_path.read_text(encoding="utf-8")) >= 3
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


def test_tls_accept_without_response_headers_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """TLS handshake and HTTP body complete; server never writes headers.

    Production v1257 py-spy: llm-request blocked in SSLSocket.read →
    httpcore _receive_response_headers; main stayed in _wait_until. The
    application deadline must raise a retryable timeout.
    """
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "silent", tls=True)
        assert cert is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        client = _make_client(
            f"https://127.0.0.1:{port}/v1",
            timeout=1.0,
            max_retries=0,
        )
        _seed_cached_json_object(client)
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            _call_json(client, _LARGE_BODY, "review.scan")
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"TLS header stall hung for {elapsed:.2f}s"
        assert elapsed >= 1.0
        assert caught.value.reason == "timeout"
        assert caught.value.resumable is True
        accepted = int((count_path.read_text(encoding="utf-8") or "0").strip() or "0")
        assert accepted <= 1
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


def test_verify_success_then_scan_tls_header_stall_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """stall-135214: review.verify completed, then review.scan hung after
    llm_json_mode_selected (cached json_object) with no completed/retry event.
    Server accepts TLS, replies once, then never writes headers.
    """
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "ok_then_silent", tls=True)
        assert cert is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        client = _make_client(
            f"https://127.0.0.1:{port}/v1",
            timeout=1.0,
            max_retries=0,
            max_concurrency=2,
        )
        _seed_cached_json_object(client)
        verify_text = _call_json(client, "small verify", "review.verify")
        assert verify_text
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            _call_json(client, _LARGE_BODY, "review.scan")
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"scan after verify hung for {elapsed:.2f}s"
        assert elapsed >= 1.0
        assert caught.value.reason == "timeout"
        assert caught.value.resumable is True
        assert int(count_path.read_text(encoding="utf-8")) == 2
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


class _LoopTrace:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}

    def load(self, agent_id: str) -> dict[str, Any] | None:
        return deepcopy(self.data.get(agent_id))

    def save(self, agent_id: str, snapshot: dict[str, Any]) -> None:
        self.data[agent_id] = deepcopy(snapshot)

    def log_event(self, event: str, **data: Any) -> None:
        return None


class _LoopEvidence:
    def execute(self, request: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    def evidence_refs(self, value: Any) -> set[str]:
        return set()


def test_verify_scan_then_cached_verify_tls_header_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """stall-143323: PID 339625 v1273. review.verify and review.scan both
    completed with cached json_object, then a later review.verify hung after
    llm_json_mode_selected with ESTAB and main still in _wait_until.
    """
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "ok_two_then_silent", tls=True)
        assert cert is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        monkeypatch.setenv("WENYI_LLM_HARD_DEADLINE_S", "2")
        client = _make_client(
            f"https://127.0.0.1:{port}/v1",
            timeout=300.0,
            max_retries=0,
            max_concurrency=2,
            max_output_tokens=65536,
        )
        _seed_cached_json_object(client)
        assert _call_json(client, "small verify", "review.verify")
        assert _call_json(client, "small scan", "review.scan")
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            _call_json(client, "x" * 124000, "review.verify")
        elapsed = time.monotonic() - started
        assert elapsed < 6.0, f"cached verify after scan hung for {elapsed:.2f}s"
        assert elapsed >= 1.5
        assert caught.value.reason == "timeout"
        assert caught.value.resumable is True
        assert int(count_path.read_text(encoding="utf-8")) == 3
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


def _spawn_child_count() -> int:
    """Count this process's multiprocessing.spawn children (isolated attempts)."""
    my_pid = os.getpid()
    count = 0
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = None
            for line in (entry / "status").read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid != my_pid:
                continue
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if b"multiprocessing.spawn" in cmdline or b"spawn_main" in cmdline:
            count += 1
    return count


def test_autofix_verify_action_loop_tls_header_stall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """stall-150106 (v1273 PID 341458), stall-151327 (v1274 PID 363675),
    stall-152118 (v1274 PID 368348 result-11), stall-152743 (v1274 PID
    370971 result-12).

    AutofixVerification.fix_job hung in ReviewActionLoop after
    llm_json_mode_selected (cached json_object, autofix.verify). autofix.fix
    and earlier autofix.verify calls had completed; socket stayed ESTAB. v1274
    sliced SSL polls to 1s and capped at 240s, but MainThread/watchdog still
    sat in _wait_until. Isolation must let the parent return and SIGKILL the
    child so a GIL-holding SSL poll cannot pin the writer process.
    """
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "ok_two_then_silent", tls=True)
        assert cert is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        monkeypatch.setenv("WENYI_LLM_HARD_DEADLINE_S", "2")
        client = _make_client(
            f"https://127.0.0.1:{port}/v1",
            timeout=300.0,
            max_retries=0,
            max_output_tokens=65536,
        )
        _seed_cached_json_object(client)
        assert _call_json(client, "small fix", "autofix.fix")
        assert _call_json(client, "small verify", "autofix.verify")
        user = "x" * 124000
        loop_config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"review_agent_max_evidence_rounds": 0},
            }
        )
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            ReviewActionLoop(client, loop_config, _LoopEvidence(), _LoopTrace()).run(
                agent_id="pid-341458",
                system="System instruction",
                user=user,
                stage="autofix.verify",
                allowed_refs=set(),
                validate_final=lambda data, refs: data,
            )
        elapsed = time.monotonic() - started
        assert elapsed < 6.0, f"autofix.verify hung for {elapsed:.2f}s"
        assert elapsed >= 1.5
        assert caught.value.reason == "timeout"
        assert caught.value.resumable is True
        assert int(count_path.read_text(encoding="utf-8")) == 3
        time.sleep(0.2)
        assert _spawn_child_count() == 0, "isolated attempt child survived SIGKILL"
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)


def test_action_loop_verify_tls_header_stall_large_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Match PID 339170: ReviewActionLoop review.verify, cached json_object,
    ~190k-token body, TLS server reads the full request then never writes
    headers. Main must return at the attempt deadline even if llm-request
    is still in SSLSocket.read.
    """
    proc = None
    try:
        proc, port, count_path, cert = _start_stall_server(tmp_path, "silent", tls=True)
        assert cert is not None
        monkeypatch.setenv("SSL_CERT_FILE", str(cert))
        monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(cert))
        monkeypatch.setenv("WENYI_LLM_HARD_DEADLINE_S", "2")
        client = _make_client(
            f"https://127.0.0.1:{port}/v1",
            timeout=300.0,
            max_retries=0,
            max_output_tokens=65536,
        )
        _seed_cached_json_object(client)
        user = "x" * 124000
        messages = [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": user},
        ]
        estimate = len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 256 + 65536
        assert 180_000 <= estimate <= 210_000, estimate
        loop_config = Config.from_dict(
            {
                "language": {"source": "en", "target": "zh"},
                "llm": {"preset": "fake"},
                "pipeline": {"review_agent_max_evidence_rounds": 0},
            }
        )
        started = time.monotonic()
        with pytest.raises(ProviderRequestError) as caught:
            ReviewActionLoop(client, loop_config, _LoopEvidence(), _LoopTrace()).run(
                agent_id="pid-339170",
                system="System instruction",
                user=user,
                stage="review.verify",
                allowed_refs=set(),
                validate_final=lambda data, refs: data,
            )
        elapsed = time.monotonic() - started
        worker_alive = any(
            thread.name == "llm-request" and thread.is_alive() for thread in threading.enumerate()
        )
        assert elapsed < 6.0, (
            f"action-loop verify hung for {elapsed:.2f}s worker_alive={worker_alive}"
        )
        assert elapsed >= 1.5
        assert caught.value.reason == "timeout"
        assert caught.value.resumable is True
        accepted = int((count_path.read_text(encoding="utf-8") or "0").strip() or "0")
        assert accepted <= 1
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
            # Isolated attempts use spawn + a Pipe; a few leftover tracker fds are OK.
            assert fds_after <= fds_before + 8
    finally:
        if proc is not None:
            proc.kill()
            proc.wait(timeout=5)
