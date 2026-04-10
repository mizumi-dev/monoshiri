"""
モノシリ ファイルテキスト抽出モジュール
各ファイル形式からテキストとページ/スライドのメタデータを抽出する。
外部送信なし・完全ローカル処理。
"""
from __future__ import annotations
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 対応ファイル拡張子
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".doc",
    ".pptx",
    ".xlsx", ".xls",
    ".txt", ".csv", ".md",
    ".dwg", ".dxf",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
}

# テキストを完全抽出できる形式
EXTRACTABLE_EXTENSIONS: set[str] = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".csv", ".md",
}

# ファイル名のみ記録する形式（テキスト抽出不可）
FILENAME_ONLY_EXTENSIONS: set[str] = {
    ".doc", ".xls",
    ".dwg", ".dxf",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp",
}

# 形式の表示名マッピング
EXTENSION_LABELS: dict[str, str] = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".doc": "Word（旧形式）",
    ".pptx": "PowerPoint",
    ".xlsx": "Excel",
    ".xls": "Excel（旧形式）",
    ".txt": "テキスト",
    ".csv": "CSV",
    ".md": "Markdown",
    ".dwg": "CAD（DWG）",
    ".dxf": "CAD（DXF）",
    ".jpg": "画像（JPG）",
    ".jpeg": "画像（JPEG）",
    ".png": "画像（PNG）",
    ".gif": "画像（GIF）",
    ".bmp": "画像（BMP）",
}


def extract_text(file_path: Path) -> tuple[list[dict], str | None]:
    """
    ファイルからテキストとメタデータを抽出する。

    Returns:
        (page_chunks, skip_reason)
        page_chunks: [{"text": str, "page": int|None, "slide": int|None}]
        skip_reason: スキップ理由（成功時はNone）
    """
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        return [], f"未対応形式: {ext}"

    # ファイル名のみ記録する形式
    if ext in FILENAME_ONLY_EXTENSIONS:
        label = EXTENSION_LABELS.get(ext, ext)
        return [{"text": f"[{label}] {file_path.name}", "page": None, "slide": None}], None

    try:
        if ext == ".pdf":
            return _extract_pdf(file_path)
        elif ext == ".docx":
            return _extract_docx(file_path)
        elif ext == ".pptx":
            return _extract_pptx(file_path)
        elif ext == ".xlsx":
            return _extract_xlsx(file_path)
        elif ext in {".txt", ".csv", ".md"}:
            return _extract_text_file(file_path)
    except MemoryError:
        return [], "メモリ不足（ファイルが大きすぎます）"
    except Exception as e:
        logger.error(f"抽出エラー {file_path.name}: {e}")
        return [], f"抽出エラー: {str(e)[:120]}"

    return [], "未対応形式"


def _extract_pdf(file_path: Path) -> tuple[list[dict], None]:
    """PyMuPDFを使いページごとにテキスト抽出

    BUG-027修正: get_text("text") → get_text("blocks") + 座標ソート
    財務PDFの表構造（年度×指標）を保持するため、テキストブロックをY座標→X座標順に
    ソートし直す。これにより列単位での誤抽出を防ぎ、行単位の自然読み順を保持する。
    round(b[1] / 10) でY座標を10px単位に丸め、同一行の左右ブロックをX座標で並べる。
    """
    import fitz  # PyMuPDF

    page_chunks = []
    doc = fitz.open(str(file_path))
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            # blocks = [(x0, y0, x1, y1, text, block_no, block_type), ...]
            blocks = page.get_text("blocks")
            # テキストブロック(block_type==0)のみ、Y→X座標順でソート
            text_blocks = sorted(
                [b for b in blocks if b[6] == 0],
                key=lambda b: (round(b[1] / 10), b[0])  # 行単位(10px)でY→X
            )
            text = "\n".join(b[4].strip() for b in text_blocks if b[4].strip())
            if text:
                page_chunks.append({
                    "text": text,
                    "page": page_num + 1,
                    "slide": None,
                })
    finally:
        doc.close()

    return page_chunks, None


def _extract_docx(file_path: Path) -> tuple[list[dict], None]:
    """python-docxを使い全文テキスト抽出"""
    from docx import Document

    doc = Document(str(file_path))
    texts: list[str] = []

    # 段落テキスト
    for para in doc.paragraphs:
        stripped = para.text.strip()
        if stripped:
            texts.append(stripped)

    # テーブルセル
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                stripped = cell.text.strip()
                if stripped:
                    texts.append(stripped)

    full_text = "\n".join(texts)
    if not full_text:
        return [], None

    return [{"text": full_text, "page": None, "slide": None}], None


def _extract_pptx(file_path: Path) -> tuple[list[dict], None]:
    """python-pptxを使いスライドごとにテキスト抽出"""
    from pptx import Presentation

    prs = Presentation(str(file_path))
    page_chunks = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts: list[str] = []
        for shape in slide.shapes:
            # テキストフレームを持つシェイプ
            if hasattr(shape, "text_frame"):
                for para in shape.text_frame.paragraphs:
                    text = para.text.strip()
                    if text:
                        texts.append(text)
            elif hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
            # テーブル
            if shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        t = cell.text.strip()
                        if t:
                            texts.append(t)

        if texts:
            page_chunks.append({
                "text": "\n".join(texts),
                "page": None,
                "slide": slide_num,
            })

    return page_chunks, None


def _extract_xlsx(file_path: Path) -> tuple[list[dict], None]:
    """openpyxlを使いセルテキストのみ抽出（グラフ・画像はスキップ）"""
    import openpyxl

    wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
    texts: list[str] = []

    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    val = str(cell.value).strip()
                    if val and val.lower() != "none":
                        texts.append(val)

    wb.close()

    if not texts:
        return [], None

    return [{"text": "\n".join(texts), "page": None, "slide": None}], None


def _extract_text_file(file_path: Path) -> tuple[list[dict], str | None]:
    """テキスト・CSV・Markdownファイルを読み込む"""
    # 文字コードを順番に試みる
    for encoding in ("utf-8", "utf-8-sig", "shift_jis", "cp932", "euc_jp", "latin-1"):
        try:
            text = file_path.read_text(encoding=encoding).strip()
            if text:
                return [{"text": text, "page": None, "slide": None}], None
            return [], None  # 空ファイル
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return [], f"読み込みエラー: {str(e)[:80]}"

    return [], "文字コードを判定できませんでした"


def scan_folder(folder_path: Path) -> list[Path]:
    """
    フォルダ内の対象ファイルをすべて列挙する（サブフォルダ含む）
    隠しファイル・隠しフォルダはスキップする。
    """
    files: list[Path] = []
    try:
        for f in folder_path.rg