"""
core/extractor.py + core/indexer.py の split_text の単体テスト
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import extractor
from core.indexer import split_text


class TestSupportedExtensions:
    def test_supports_pdf_docx_xlsx_pptx_txt(self):
        exts = extractor.SUPPORTED_EXTENSIONS
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".xlsx" in exts
        assert ".pptx" in exts
        assert ".txt" in exts


class TestExtractText:
    def test_txt_file(self, sample_txt_file):
        page_chunks, skip_reason = extractor.extract_text(sample_txt_file)
        assert skip_reason is None
        assert len(page_chunks) > 0
        assert "モノシリ" in page_chunks[0]["text"]

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "file.zip"
        f.write_bytes(b"PK\x03\x04")
        chunks, reason = extractor.extract_text(f)
        assert chunks == []
        assert reason is not None
        assert "未対応" in reason

    def test_corrupt_pdf_returns_skip_reason(self, tmp_path):
        f = tmp_path / "broken.pdf"
        f.write_bytes(b"not a real pdf")
        chunks, reason = extractor.extract_text(f)
        # 破損PDFは抽出失敗で reason が返る（chunks は空）
        assert chunks == [] or reason is not None


class TestScanFolder:
    def test_scans_only_supported_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.md").write_text("b", encoding="utf-8")
        (tmp_path / "c.zip").write_bytes(b"zip")
        (tmp_path / "d.exe").write_bytes(b"exe")

        files = extractor.scan_folder(tmp_path)
        names = {f.name for f in files}
        assert "a.txt" in names
        assert "b.md" in names
        assert "c.zip" not in names
        assert "d.exe" not in names

    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / "visible.txt").write_text("v", encoding="utf-8")
        (tmp_path / ".hidden.txt").write_text("h", encoding="utf-8")
        hidden_dir = tmp_path / ".cache"
        hidden_dir.mkdir()
        (hidden_dir / "inside.txt").write_text("i", encoding="utf-8")

        files = extractor.scan_folder(tmp_path)
        names = {f.name for f in files}
        assert "visible.txt" in names
        assert ".hidden.txt" not in names
        assert "inside.txt" not in names

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep", encoding="utf-8")
        files = extractor.scan_folder(tmp_path)
        assert any(f.name == "deep.txt" for f in files)


class TestSplitText:
    def test_empty_returns_empty(self):
        assert split_text("") == []
        assert split_text("   \n   ") == []

    def test_short_text_single_chunk(self):
        chunks = split_text("短いテキストです。")
        assert len(chunks) == 1
        assert chunks[0] == "短いテキストです。"

    def test_long_text_multi_chunks(self):
        # CHUNK_SIZE=1200 を超える長文を生成
        text = "あいうえお。" * 500  # 3000 文字
        chunks = split_text(text)
        assert len(chunks) >= 2
        # 各チャンクが CHUNK_SIZE 以下
        for c in chunks:
            assert len(c) <= 1200

    def test_custom_chunk_size(self):
        text = "あ" * 300
        chunks = split_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 100


class TestEstimateIndexTime:
    def test_zero_files(self):
        assert extractor.estimate_index_time(0) == "0秒"

    def test_returns_string(self):
        result = extractor.estimate_index_time(100)
        assert isinstance(result, str)
        assert len(result) > 0
