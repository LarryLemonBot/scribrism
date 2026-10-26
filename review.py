"""Local document extraction and hashing utilities. No model or network access."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import math

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def extract(path):
    """Read every text paragraph/page; retain ordering and disclose exclusions."""
    path = Path(path)
    if path.stat().st_size > 25_000_000:
        raise ValueError('File exceeds the 25 MB ingestion limit.')
    ext = path.suffix.lower()
    notes = []
    if ext in {'.txt', '.md'}:
        text = path.read_text(encoding='utf-8-sig')
    elif ext == '.docx':
        ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        with ZipFile(path) as z:
            info = z.getinfo('word/document.xml')
            if info.file_size > 25_000_000:
                raise ValueError('Expanded document XML exceeds 25 MB.')
            document = ET.fromstring(z.read(info))
            body = document.find(ns + 'body')
            if body is None:
                raise ValueError('DOCX has no document body.')
            paragraphs = []
            for p in body.iter(ns + 'p'):
                # Deleted revisions use delText; only current w:t nodes are read.
                paragraphs.append(''.join(n.text or '' if n.tag == ns+'t' else
                                          '\t' if n.tag == ns+'tab' else '\n'
                                          for n in p.iter() if n.tag in
                                          {ns+'t', ns+'tab', ns+'br', ns+'cr'}))
            text = '\n\n'.join(paragraphs)
            notes.append('Scanned body paragraphs and table cells, including headings, template wording and references. Headers, footers, comments, images and equations are not text-classified.')
            if document.findall('.//' + ns + 'del') or document.findall('.//' + ns + 'ins'):
                notes.append('Tracked changes exist. Inspect extracted text against the intended final Word view.')
    elif ext == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Encrypted PDF requires an unencrypted review copy.')
        pages = [page.extract_text() or '' for page in reader.pages]
        if any(not p.strip() for p in pages):
            raise ValueError('PDF contains a page without extractable text; OCR or source document required.')
        text = '\n\n'.join(pages)
        notes.append('PDF text extraction may reorder columns. Inspect the extracted text; images are not scanned.')
    else:
        raise ValueError('Supported inputs: DOCX, text PDF, TXT and Markdown. This format needs an explicit text conversion.')
    if not text.strip():
        raise ValueError('No extractable text.')
    return text, notes

def window_ranges(count, size, stride):
    if size <= 0 or stride <= 0 or stride > size:
        raise ValueError('Window size/stride leave uncovered tokens.')
    start = 0
    while start < count:
        end = min(count, start + size)
        yield start, end
        if end == count:
            break
        start += stride

def token_scores(count, windows):
    sums, visits = [0.0]*count, [0]*count
    for window in windows:
        a, b, score = window['token_start'], window['token_end'], window['score']
        if not (0 <= a < b <= count and math.isfinite(score) and 0 <= score <= 1):
            raise ValueError('Invalid window result.')
        for i in range(a, b):
            sums[i] += score
            visits[i] += 1
    if not count or any(n == 0 for n in visits):
        raise ValueError('Incomplete token coverage.')
    return [s/n for s, n in zip(sums, visits)]
