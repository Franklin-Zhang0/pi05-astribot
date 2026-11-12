"""Failure reporting must precede a process pool waiting for active encoders."""

from concurrent.futures import Future
import io
import json
import sys

import pytest

from examples.engram import preprocess_multi as preprocessing


class FlushedStderr(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushed_text = []

    def flush(self):
        self.flushed_text.append(self.getvalue())
        super().flush()


@pytest.mark.parametrize("exception_type", [ValueError, KeyboardInterrupt])
def test_main_reports_original_failure_and_cancels_pending_before_pool_exit(monkeypatch, tmp_path, exception_type):
    stderr = FlushedStderr()
    failure = exception_type("original decode failure")
    failure.__cause__ = RuntimeError("original worker traceback")
    success, failed, pending, active, later_pending = [Future() for _ in range(5)]
    success.set_result({"destination": "finished_640.mp4", "frames": 10})
    failed.set_exception(failure)
    assert active.set_running_or_notify_cancel()
    futures = [success, failed, pending, active, later_pending]
    receipt = tmp_path / "receipt.jsonl"
    exit_events = []

    class Pool:
        def __init__(self, **kwargs):
            self.submitted = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            result = futures[self.submitted]
            self.submitted += 1
            return result

        def __exit__(self, exc_type, exc_value, traceback):
            # This is where the real executor waits for running jobs. Both the
            # original traceback and all pending cancellations must exist now.
            assert exc_value is failure
            assert "Traceback (most recent call last):" in stderr.getvalue()
            assert "original worker traceback" in stderr.getvalue()
            assert f"{exception_type.__name__}: original decode failure" in stderr.getvalue()
            assert stderr.flushed_text[-1] == stderr.getvalue()
            assert pending.cancelled()
            assert later_pending.cancelled()
            assert active.running()
            assert not active.cancelled()
            assert json.loads(receipt.read_text())["destination"] == "finished_640.mp4"
            exit_events.append("pool waited after immediate error report and cancellations")
            return False

    monkeypatch.setattr(preprocessing, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(preprocessing, "jobs_for_manifest", lambda *args, **kwargs: [{"role": "head"}] * 5)
    monkeypatch.setattr(preprocessing, "as_completed", lambda values: iter([success, failed]))
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "argv", ["preprocess_multi.py", "manifest.json", "--adjacent", "--receipt", str(receipt)])
    with pytest.raises(exception_type, match="original decode failure") as caught:
        preprocessing.main()
    assert caught.value is failure
    assert len(exit_events) == 1
    assert len(receipt.read_text().splitlines()) == 1


def test_main_cancels_already_submitted_jobs_when_submission_fails(monkeypatch):
    stderr = FlushedStderr()
    pending = Future()
    failure = RuntimeError("submission failed")

    class Pool:
        def __init__(self, **kwargs):
            self.submitted = 0

        def __enter__(self):
            return self

        def submit(self, *args, **kwargs):
            self.submitted += 1
            if self.submitted == 2:
                raise failure
            return pending

        def __exit__(self, exc_type, exc_value, traceback):
            assert exc_value is failure
            assert "RuntimeError: submission failed" in stderr.flushed_text[-1]
            assert pending.cancelled()
            return False

    monkeypatch.setattr(preprocessing, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(preprocessing, "jobs_for_manifest", lambda *args, **kwargs: [{"role": "head"}] * 2)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "argv", ["preprocess_multi.py", "manifest.json", "--adjacent"])
    with pytest.raises(RuntimeError, match="submission failed") as caught:
        preprocessing.main()
    assert caught.value is failure
