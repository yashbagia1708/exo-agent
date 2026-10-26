"""The NASA downloader retries slow or failed requests. No network needed: `pytest -q`."""

import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "download_data", pathlib.Path(__file__).parent.parent / "data" / "download_data.py"
)
download_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_data)


class _Resp:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return b"pl_name,hostname\nKepler-22 b,Kepler-22\n"


def test_read_timeout_is_retried(monkeypatch):
    # what happened on GitHub: the first request timed out mid-read
    calls = []

    def urlopen(req, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise TimeoutError("The read operation timed out")
        return _Resp()

    monkeypatch.setattr(download_data.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download_data.time, "sleep", lambda s: None)

    assert download_data.fetch_csv().startswith("pl_name")
    assert len(calls) == 2
