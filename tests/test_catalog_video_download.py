import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from download_catalog_video import download_entry, find_download_entry  # noqa: E402


class FakeResponse(io.BytesIO):
    def __init__(self, content: bytes, status: int = 200) -> None:
        super().__init__(content)
        self.status = status

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class CatalogVideoDownloadTest(unittest.TestCase):
    def test_find_download_entry_rejects_unknown_id(self) -> None:
        with self.assertRaises(KeyError):
            find_download_entry({"open_video_candidates": []}, "missing")

    def test_download_is_validated_and_atomically_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "id": "sample",
                "direct_url": "https://example.test/sample.webm",
                "local_path": "clips/sample.webm",
                "split": "holdout_candidate",
            }
            result = download_entry(
                entry,
                root=root,
                opener=lambda request, timeout: FakeResponse(b"video"),
                video_probe=lambda path: {"width": 640, "height": 360},
            )
            destination = root / "clips" / "sample.webm"
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual(destination.read_bytes(), b"video")
            self.assertFalse(destination.with_suffix(".webm.part").exists())
            self.assertTrue(destination.with_suffix(".webm.download.json").is_file())

    def test_partial_download_is_resumed_when_server_supports_range(self) -> None:
        seen_range: list[str | None] = []

        def opener(request: object, timeout: float) -> FakeResponse:
            seen_range.append(request.get_header("Range"))
            return FakeResponse(b"-new", status=206)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "clips" / "sample.webm"
            destination.parent.mkdir(parents=True)
            destination.with_suffix(".webm.part").write_bytes(b"old")
            entry = {
                "id": "sample",
                "direct_url": "https://example.test/sample.webm",
                "local_path": "clips/sample.webm",
            }
            download_entry(
                entry,
                root=root,
                opener=opener,
                video_probe=lambda path: {"width": 640, "height": 360},
            )
            self.assertEqual(destination.read_bytes(), b"old-new")
            self.assertEqual(seen_range, ["bytes=3-"])

    def test_invalid_video_keeps_partial_and_does_not_create_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "id": "sample",
                "direct_url": "https://example.test/sample.webm",
                "local_path": "clips/sample.webm",
            }
            with self.assertRaisesRegex(RuntimeError, "not a decodable video"):
                download_entry(
                    entry,
                    root=root,
                    opener=lambda request, timeout: FakeResponse(b"broken"),
                    video_probe=lambda path: None,
                )
            destination = root / "clips" / "sample.webm"
            self.assertFalse(destination.exists())
            self.assertEqual(destination.with_suffix(".webm.part").read_bytes(), b"broken")

    def test_auto_transport_falls_back_to_resumable_curl(self) -> None:
        seen_command: list[str] = []

        def failing_opener(request: object, timeout: float) -> FakeResponse:
            raise OSError("TLS handshake failed")

        def curl_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            seen_command.extend(command)
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"video")
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {
                "id": "sample",
                "direct_url": "https://example.test/sample.webm",
                "local_path": "clips/sample.webm",
            }
            result = download_entry(
                entry,
                root=root,
                opener=failing_opener,
                curl_runner=curl_runner,
                video_probe=lambda path: {"width": 640, "height": 360},
            )
        self.assertEqual(result["transport"], "curl")
        self.assertEqual(result["python_https_error"], "TLS handshake failed")
        self.assertIn("--retry-all-errors", seen_command)
        self.assertIn("-4", seen_command)


if __name__ == "__main__":
    unittest.main()
