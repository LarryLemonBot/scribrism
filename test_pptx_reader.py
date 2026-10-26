import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from xml.sax.saxutils import escape

import pptx_reader

P = pptx_reader.NS['p']
A = pptx_reader.NS['a']
R = pptx_reader.NS['r']
REL = pptx_reader.NS['rel']


def shape(text, placeholder=''):
    ph = '<p:ph type="' + placeholder + '"/>' if placeholder else ''
    return '<p:sp><p:nvSpPr><p:nvPr>' + ph + '</p:nvPr></p:nvSpPr><p:txBody>' + \
           '<a:p><a:r><a:t>' + escape(text) + '</a:t></a:r></a:p></p:txBody></p:sp>'


def container(content, notes=False):
    tag = 'notes' if notes else 'sld'
    return f'<p:{tag} xmlns:p="{P}" xmlns:a="{A}"><p:cSld><p:spTree>{content}</p:spTree></p:cSld></p:{tag}>'


def fixture(path, *, extra='', external=False):
    # Deliberately nonnumeric presentation order; include an orphan notes part
    # so a glob-based extractor would produce wrong ordering/extra content.
    with ZipFile(path, 'w') as z:
        z.writestr('ppt/presentation.xml', f'<p:presentation xmlns:p="{P}" xmlns:r="{R}"><p:sldIdLst><p:sldId r:id="r2"/><p:sldId r:id="r1"/></p:sldIdLst></p:presentation>')
        mode = ' TargetMode="External"' if external else ''
        z.writestr('ppt/_rels/presentation.xml.rels', f'<Relationships xmlns="{REL}"><Relationship Id="r1" Type="{R}/slide" Target="slides/slide1.xml"/><Relationship Id="r2" Type="{R}/slide" Target="slides/slide2.xml"{mode}/></Relationships>')
        z.writestr('ppt/slides/slide2.xml', container(shape('First in deck') + shape('2', 'sldNum') + extra))
        z.writestr('ppt/slides/slide1.xml', container(shape('References') + shape('Citation stays included')))
        z.writestr('ppt/slides/_rels/slide2.xml.rels', f'<Relationships xmlns="{REL}"><Relationship Id="n" Type="{R}/notesSlide" Target="../notesSlides/notesSlide7.xml"/></Relationships>')
        z.writestr('ppt/notesSlides/notesSlide7.xml', container(shape('Complete speaker notes: café & evidence') + shape('2', 'sldNum'), True))
        z.writestr('ppt/notesSlides/notesSlide99.xml', container(shape('ORPHAN SHOULD NOT APPEAR'), True))


class PPTXReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'deck.pptx'

    def tearDown(self):
        self.tmp.cleanup()

    def test_presentation_relationship_order_notes_and_references(self):
        fixture(self.path)
        text, _ = pptx_reader.extract_pptx(self.path)
        self.assertEqual(text, 'First in deck\n\nComplete speaker notes: café & evidence\n\nReferences\nCitation stays included')
        data = pptx_reader.read_pptx(self.path)
        self.assertEqual(data['slides'][0]['notes_part'], 'ppt/notesSlides/notesSlide7.xml')
        self.assertIsNone(data['slides'][1]['notes_part'])

    def test_group_table_and_manual_breaks_not_joined(self):
        group = '<p:grpSp>' + shape('Grouped text') + '</p:grpSp>'
        table = '<p:graphicFrame><a:graphic><a:graphicData><a:tbl><a:tr>'
        for text in ['Cell A', 'Cell B']:
            table += '<a:tc><a:txBody><a:p><a:r><a:t>' + text + '</a:t></a:r></a:p></a:txBody></a:tc>'
        table += '</a:tr></a:tbl></a:graphicData></a:graphic></p:graphicFrame>'
        line = shape('Before').replace('</a:r></a:p>', '</a:r><a:br/><a:r><a:t>After</a:t></a:r></a:p>')
        fixture(self.path, extra=group + table + line)
        text, _ = pptx_reader.extract_pptx(self.path)
        self.assertIn('Grouped text\nCell A\tCell B\nBefore\nAfter', text)

    def test_unreadable_visual_content_is_not_silently_skipped(self):
        for extra in ['<p:pic/>', '<p:graphicFrame><a:graphic><a:graphicData/></a:graphic></p:graphicFrame>']:
            fixture(self.path, extra=extra)
            with self.assertRaises(ValueError):
                pptx_reader.extract_pptx(self.path)

    def test_external_slide_is_not_fetched(self):
        fixture(self.path, external=True)
        with self.assertRaisesRegex(ValueError, 'External'):
            pptx_reader.extract_pptx(self.path)

    def test_nonzip_is_explicit_failure(self):
        self.path.write_text('not a PPTX')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            pptx_reader.extract_pptx(self.path)

    def test_duplicate_parts_fail_closed(self):
        fixture(self.path)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with ZipFile(self.path, 'a') as z:
                z.writestr('ppt/slides/slide1.xml', container(shape('replacement')))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pptx_reader.extract_pptx(self.path)


if __name__ == '__main__':
    unittest.main()
