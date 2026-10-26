"""Read authored PPTX text and notes directly, without rendering or network I/O.

Extraction does not require Office, presentation rendering or network access.
"""
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, BadZipFile

NS = {
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'rel': 'http://schemas.openxmlformats.org/package/2006/relationships',
}
MAX_FILE = 20_000_000
MAX_PACKAGE = 100_000_000
FURNITURE = {'sldNum', 'dt', 'ftr', 'hdr'}


def _xml(z, name):
    try:
        info = z.getinfo(name)
        if info.file_size > MAX_FILE:
            raise ValueError('PPTX XML exceeds 20 MB: ' + name)
        data = z.read(info)
        if b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
            raise ValueError('PPTX DTD/entity declarations are not supported.')
        return ET.fromstring(data)
    except (KeyError, ET.ParseError) as e:
        raise ValueError('Missing or invalid PPTX XML: ' + name) from e


def _rels(z, part):
    name = posixpath.join(posixpath.dirname(part), '_rels', posixpath.basename(part) + '.rels')
    if name not in z.namelist():
        return {}
    result = {}
    for r in _xml(z, name):
        rid = r.get('Id')
        if not rid or rid in result:
            raise ValueError('Missing or duplicate PPTX relationship ID.')
        result[rid] = dict(r.attrib)
    return result


def _target(part, rel):
    if rel.get('TargetMode') == 'External':
        raise ValueError('External slide or notes content is not read.')
    target = rel.get('Target', '')
    if not target or '\\' in target or '?' in target or '#' in target:
        raise ValueError('Invalid PPTX part target.')
    name = posixpath.normpath(target.lstrip('/') if target.startswith('/') else
                             posixpath.join(posixpath.dirname(part), target))
    if name == '..' or name.startswith('../'):
        raise ValueError('PPTX relationship escapes package.')
    return name


def _paragraphs(root):
    result = []
    for p in root.findall('.//a:p', NS):
        text = ''.join(n.text or '' if n.tag == '{' + NS['a'] + '}t' else '\n'
                       for n in p.iter() if n.tag in {'{' + NS['a'] + '}t', '{' + NS['a'] + '}br'})
        if text.strip():
            result.append(text.strip())
    return result


def _content(root, part, notes=False):
    tree = root.find('p:cSld/p:spTree', NS)
    if tree is None:
        raise ValueError('Missing PPTX shape tree: ' + part)
    output, skipped = [], []
    # Refuse content whose prose cannot be recovered rather than scoring only
    # the easy text and pretending the document is complete.
    for tag in ('pic', 'oleObj', 'contentPart'):
        if tree.find('.//p:' + tag, NS) is not None:
            raise ValueError('PPTX contains images or embedded content needing separate review: ' + part)
    if tree.find('.//a:blip', NS) is not None:
        raise ValueError('PPTX contains image-filled content needing separate review: ' + part)
    if any(n.tag.endswith('}AlternateContent') for n in tree.iter()):
        raise ValueError('PPTX alternate content is not supported: ' + part)

    def visit(container):
        for shape in container:
            if shape.tag == '{' + NS['p'] + '}grpSp':
                visit(shape)
            elif shape.tag == '{' + NS['p'] + '}sp':
                ph = shape.find('p:nvSpPr/p:nvPr/p:ph', NS)
                kind = ph.get('type', '') if ph is not None else ''
                if kind in FURNITURE or (notes and kind == 'sldImg'):
                    skipped.append(kind)
                    continue
                paragraphs = _paragraphs(shape)
                if paragraphs:
                    output.append('\n'.join(paragraphs))
            elif shape.tag == '{' + NS['p'] + '}graphicFrame':
                data = shape.find('a:graphic/a:graphicData', NS)
                table = data.find('a:tbl', NS) if data is not None else None
                if table is None:
                    raise ValueError('PPTX chart/SmartArt/graphic requires separate review: ' + part)
                for row in table.findall('a:tr', NS):
                    output.append('\t'.join('\n'.join(_paragraphs(cell)) for cell in row.findall('a:tc', NS)))
    visit(tree)
    return output, skipped


def read_pptx(path):
    """Return exact text blocks plus source coverage, in presentation order.

    Includes hidden slides and full notes. Excludes identified slide-number,
    date/header/footer placeholders and notes thumbnails; no OCR or master text.
    """
    path = Path(path)
    if path.stat().st_size > MAX_FILE:
        raise ValueError('Input exceeds 20 MB')
    try:
        with ZipFile(path) as z:
            infos = z.infolist()
            if len(infos) > 5000 or sum(i.file_size for i in infos) > MAX_PACKAGE:
                raise ValueError('PPTX package exceeds extraction limits.')
            if len({i.filename for i in infos}) != len(infos):
                raise ValueError('Duplicate PPTX package parts.')
            if any(i.flag_bits & 1 for i in infos):
                raise ValueError('Encrypted PPTX content is not supported.')
            if any('vbaProject' in i.filename for i in infos):
                raise ValueError('Macro-enabled presentation is not supported.')
            part = 'ppt/presentation.xml'
            root = _xml(z, part)
            ids = root.findall('p:sldIdLst/p:sldId', NS)
            if not ids:
                raise ValueError('No presentation slides found; unsupported namespace or empty deck.')
            rels = _rels(z, part)
            blocks, slides, seen = [], [], set()
            for index, sid in enumerate(ids, 1):
                rid = sid.get('{' + NS['r'] + '}id')
                rel = rels.get(rid, {})
                if not rel.get('Type', '').endswith('/slide'):
                    raise ValueError('Missing slide relationship.')
                name = _target(part, rel)
                if name in seen:
                    raise ValueError('Repeated slide part.')
                seen.add(name)
                slide = _xml(z, name)
                visible, skipped = _content(slide, name)
                if visible:
                    blocks.append({'text': '\n'.join(visible), 'role': 'slide_text',
                                   'paragraph_style': 'PPTX slide ' + str(index), 'slide': index, 'part': name})
                note_rels = [r for r in _rels(z, name).values() if r.get('Type', '').endswith('/notesSlide')]
                if len(note_rels) > 1:
                    raise ValueError('Multiple notes parts for one slide.')
                note_name, note_text = None, []
                if note_rels:
                    note_name = _target(name, note_rels[0])
                    note_text, note_skipped = _content(_xml(z, note_name), note_name, notes=True)
                    skipped.extend(note_skipped)
                    for block in note_text:
                        # Preserve whole notes paragraphs and their line breaks.
                        for paragraph in re.split(r'\n\s*\n', block):
                            if paragraph.strip():
                                blocks.append({'text': paragraph.strip(), 'role': 'speaker_notes',
                                               'paragraph_style': 'PPTX slide ' + str(index) + ' notes',
                                               'slide': index, 'part': note_name})
                slides.append({'slide': index, 'part': name, 'hidden': slide.get('show') == '0',
                               'notes_part': note_name, 'slide_characters': len('\n'.join(visible)),
                               'notes_characters': len('\n'.join(note_text)), 'excluded_placeholders': skipped})
            if not any(b['text'].strip() for b in blocks):
                raise ValueError('No authored PPTX text or speaker notes.')
            return {'blocks': blocks, 'slides': slides,
                    'scope': 'Authored slide text (including tables and hidden slides) and speaker notes; '
                             'references included. Shape order, not geometric reading-order inference. '
                             'Identified slide-number/date/header/footer placeholders and notes thumbnails excluded. '
                             'No slide-master artwork, OCR, chart, SmartArt, embedded-object or media interpretation.'}
    except (BadZipFile, RuntimeError) as e:
        raise ValueError('PPTX is invalid, encrypted or unreadable.') from e


def extract_pptx(path):
    data = read_pptx(path)
    text = '\n\n'.join(block['text'] for block in data['blocks'])
    notes = [data['scope'], 'Direct PPTX input: ' + str(len(data['slides'])) + ' slides; ' +
             str(sum(s['notes_part'] is not None for s in data['slides'])) + ' speaker-note parts.']
    return text, notes
