"""
core/hash_manager.py の単体テスト
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core import hash_manager as hm


class TestComputeHash:
    def test_sha256_of_known_content(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert hm.compute_hash(f) == expected

    def test_missing_file_returns_none(self, tmp_path):
        missing = tmp_path / "nothere.txt"
        assert hm.compute_hash(missing) is None

    def test_large_file_uses_streaming(self, tmp_path):
        f = tmp_path / "big.bin"
        data = b"A" * (200 * 1024)  # 200KB（chunk 65536 を 4 回）
        f.write_bytes(data)
        assert hm.compute_hash(f) == hashlib.sha256(data).hexdigest()


class TestLightweightHash:
    def test_same_file_same_hash(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("content", encoding="utf-8")
        h1 = hm.compute_lightweight_hash(f)
        h2 = hm.compute_lightweight_hash(f)
        assert h1 == h2
        assert h1 is not None

    def test_missing_returns_none(self, tmp_path):
        assert hm.compute_lightweight_hash(tmp_path / "missing") is None


class TestMakeFolderId:
    def test_deterministic(self, tmp_path):
        assert hm.make_folder_id(tmp_path) == hm.make_folder_id(str(tmp_path))

    def test_length_16(self, tmp_path):
        assert len(hm.make_folder_id(tmp_path)) == 16


class TestManifest:
    def test_load_empty_manifest(self, isolate_data_paths):
        assert hm._load_manifest() == {}

    def test_save_and_load_hashes_roundtrip(self, isolate_data_paths):
        fid = "abcd1234"
        data = {"/tmp/a.pdf": "h1", "/tmp/b.pdf": "h2"}
        hm.save_hashes(fid, data)
        loaded = hm.load_hashes(fid)
        assert loaded == data

    def test_save_replaces_existing_folder_entries(self, isolate_data_paths):
        fid = "folder1"
        hm.save_hashes(fid, {"/a": "1", "/b": "2"})
        hm.save_hashes(fid, {"/c": "3"})
        assert hm.load_hashes(fid) == {"/c": "3"}

    def test_update_file_hash(self, isolate_data_paths, tmp_path):
        fid = "folder1"
        fpath = tmp_path / "a.pdf"
        key = str(fpath)
        hm.save_hashes(fid, {key: "old"})
        hm.update_file_hash(fid, fpath, "new")
        assert hm.load_hashes(fid)[key] == "new"

    def test_delete_folder_hashes(self, isolate_data_paths):
        hm.save_hashes("f1", {"/a": "1"})
        hm.save_hashes("f2", {"/b": "2"})
        hm.delete_folder_hashes("f1")
        assert hm.load_hashes("f1") == {}
        assert hm.load_hashes("f2") == {"/b": "2"}

    def test_clear_all_hashes(self, isolate_data_paths):
        hm.save_hashes("f1", {"/a": "1"})
        hm.save_hashes("f2", {"/b": "2"})
        hm.clear_all_hashes()
        assert hm.load_hashes("f1") == {}
        assert hm.load_hashes("f2") == {}


class TestGetDiff:
    def _setup_folder(self, tmp_path):
        folder = tmp_path / "docs"
        folder.mkdir()
        a = folder / "a.txt"
        b = folder / "b.txt"
        a.write_text("A", encoding="utf-8")
        b.write_text("B", encoding="utf-8")
        return folder, [a, b]

    def test_first_run_all_new(self, isolate_data_paths, tmp_path):
        folder, files = self._setup_folder(tmp_path)
        fid = hm.make_folder_id(folder)
        new, unchanged, deleted = hm.get_diff(fid, files, use_full_hash=True)
        assert len(new) == 2
        assert unchanged == []
        assert deleted == []

    def test_after_save_all_unchanged(self, isolate_data_paths, tmp_path):
        folder, files = self._setup_folder(tmp_path)
        fid = hm.make_folder_id(folder)
        hashes = {str(f): hm.compute_hash(f) for f in files}
        hm.save_hashes(fid, hashes)

        new, unchanged, deleted = hm.get_diff(fid, files, use_full_hash=True)
        assert new == []
        assert len(unchanged) == 2
        assert deleted == []

    def test_modified_file_detected(self, isolate_data_paths, tmp_path):
        folder, files = self._setup_folder(tmp_path)
        fid = hm.make_folder_id(folder)
        hashes = {str(f): hm.compute_hash(f) for f in files}
        hm.save_hashes(fid, hashes)

        # a.txt を書き換え
        files[0].write_text("CHANGED", encoding="utf-8")
        new, unchanged, deleted = hm.get_diff(fid, files, use_full_hash=True)
        assert len(new) == 1
        assert new[0] == files[0]
        assert len(unchanged) == 1
        assert deleted == []

    def test_deleted_file_detected(self, isolate_data_paths, tmp_path):
        folder, files = self._setup_folder(tmp_path)
        fid = hm.make_folder_id(folder)
        hashes = {str(f): hm.compute_hash(f) for f in files}
        hm.save_hashes(fid, hashes)

        # 2 番目を削除
        new, unchanged, deleted = hm.get_diff(fid, [files[0]], use_full_hash=True)
        assert str(files[1]) in deleted
