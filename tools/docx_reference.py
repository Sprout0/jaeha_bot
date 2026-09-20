# -*- coding: utf-8 -*-
"""pandoc 의 기본 reference.docx 를 한국어 공문 서식으로 고쳐 놓는다.

tools/build_docs_docx.py 가 부른다. 여기서 정하는 것은 글꼴·용지·제목 크기·표 테두리
같은 '모양'뿐이고, 내용은 전부 docs/final 의 markdown 에서 온다.

  python tools/docx_reference.py <pandoc.exe> <출력할 reference.docx 경로>
"""
import re, shutil, subprocess, zipfile, os, sys, tempfile

KO = 'w:ascii="맑은 고딕" w:eastAsia="맑은 고딕" w:hAnsi="맑은 고딕" w:cs="맑은 고딕"'
FT = '<w:rPr><w:rFonts ' + KO + '/><w:sz w:val="18"/><w:color w:val="6B7280"/></w:rPr>'


if len(sys.argv) != 3:
    sys.exit("사용: python tools/docx_reference.py <pandoc> <나올 reference.docx>")
PANDOC, OUT = sys.argv[1], os.path.abspath(sys.argv[2])

# pandoc 이 들고 있는 기본 서식 틀을 꺼내 푼다. 여기에 우리 서식을 덧칠한다.
REF = tempfile.mkdtemp(prefix="docx-ref-")
_seed = os.path.join(REF, "_seed.docx")
open(_seed, "wb").write(subprocess.run(
    [PANDOC, "--print-default-data-file", "reference.docx"],
    capture_output=True, check=True).stdout)
with zipfile.ZipFile(_seed) as _z:
    _z.extractall(REF)
os.remove(_seed)


def read(p):
    return open(os.path.join(REF, p), encoding="utf-8").read()


def write(p, s):
    open(os.path.join(REF, p), "w", encoding="utf-8").write(s)


# ── styles.xml ────────────────────────────────────────────────
s = read("word/styles.xml")

# 문서 기본값: 맑은 고딕 10.5pt, 줄간격 1.4, 문단 뒤 4pt
s = re.sub(
    r"<w:docDefaults>.*?</w:docDefaults>",
    ('<w:docDefaults><w:rPrDefault><w:rPr>'
     '<w:rFonts ' + KO + '/><w:sz w:val="21"/><w:szCs w:val="21"/>'
     '<w:lang w:val="en-US" w:eastAsia="ko-KR" w:bidi="ar-SA"/>'
     '</w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>'
     # 워드는 한글과 숫자 사이에 제멋대로 공백을 넣는다 — '2세'가 '2 세'로 찍힌다
     '<w:autoSpaceDE w:val="0"/><w:autoSpaceDN w:val="0"/>'
     '<w:spacing w:after="80" w:line="336" w:lineRule="auto"/>'
     '</w:pPr></w:pPrDefault></w:docDefaults>'),
    s, flags=re.S)

BOT = '<w:pBdr><w:bottom w:val="single" w:sz="6" w:space="4" w:color="9AA4B2"/></w:pBdr>'


def restyle(style_id, ppr, rpr):
    """한 스타일의 pPr/rPr 를 통째로 갈아끼운다."""
    global s
    m = re.search(r'<w:style [^>]*w:styleId="%s"[ >].*?</w:style>' % style_id, s, re.S)
    if not m:
        sys.exit("스타일 없음: " + style_id)
    blk = m.group(0)
    blk = re.sub(r"<w:pPr>.*?</w:pPr>", "", blk, flags=re.S)
    blk = re.sub(r"<w:rPr>.*?</w:rPr>", "", blk, flags=re.S)
    blk = blk.replace("</w:style>", ppr + rpr + "</w:style>")
    s = s[:m.start()] + blk + s[m.end():]


restyle("Title",
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="140" '
        'w:line="288" w:lineRule="auto"/>' + BOT + '</w:pPr>',
        '<w:rPr><w:rFonts ' + KO + '/><w:b/><w:color w:val="000000"/>'
        '<w:sz w:val="34"/><w:szCs w:val="34"/></w:rPr>')
restyle("Subtitle",
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="280"/></w:pPr>',
        '<w:rPr><w:rFonts ' + KO + '/><w:color w:val="4B5563"/>'
        '<w:sz w:val="21"/><w:szCs w:val="21"/></w:rPr>')

# 제목 단계: 15 / 12.5 / 11 / 10.5 pt, 전부 검정 굵게
for sid, sz, before, after, bdr in [
    ("Heading1", 30, 400, 140, BOT),
    ("Heading2", 25, 300, 100, ""),
    ("Heading3", 22, 240, 80, ""),
    ("Heading4", 21, 200, 60, ""),
    ("Heading5", 21, 180, 60, ""),
    ("Heading6", 21, 160, 60, ""),
]:
    restyle(sid,
            '<w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="%d" w:after="%d" '
            'w:line="300" w:lineRule="auto"/>%s<w:outlineLvl w:val="%d"/></w:pPr>'
            % (before, after, bdr, int(sid[-1]) - 1),
            '<w:rPr><w:rFonts ' + KO + '/><w:b/><w:color w:val="000000"/>'
            '<w:sz w:val="%d"/><w:szCs w:val="%d"/></w:rPr>' % (sz, sz))

# 본문: 양쪽 정렬 (공문 서식)
for sid in ("BodyText", "FirstParagraph"):
    restyle(sid, '<w:pPr><w:jc w:val="both"/></w:pPr>', "")

# 표 안(Compact)은 왼쪽 정렬로 되돌린다. 좁은 칸에서 양쪽 정렬을 하면 낱말 사이가
# 흉하게 벌어진다 — '화자   홀드아웃   재현율' 처럼.
restyle("Compact",
        '<w:pPr><w:jc w:val="left"/><w:spacing w:before="0" w:after="0" '
        'w:line="264" w:lineRule="auto"/></w:pPr>', "")

# 목차 제목: 본문 제목과 같은 모양으로. outlineLvl 9 = 본문 — 이걸 명시하지 않으면
# Heading1 에서 물려받아 '목차'가 목차의 첫 줄로 들어간다.
restyle("TOCHeading",
        '<w:pPr><w:keepNext/><w:spacing w:before="0" w:after="160" '
        'w:line="300" w:lineRule="auto"/>' + BOT + '<w:outlineLvl w:val="9"/></w:pPr>',
        '<w:rPr><w:rFonts ' + KO + '/><w:b/><w:color w:val="000000"/>'
        '<w:sz w:val="30"/><w:szCs w:val="30"/></w:rPr>')

# 목차 항목: 줄간격을 좁힌다. 기본값(줄간격 1.4 + 뒤 4pt)이면 12쪽 문서의 목차가 두
# 쪽으로 넘쳐 한 쪽이 거의 비어 버린다 — 2026-09-20 제출본 점검에서 나온 것이다.
for lvl in (1, 2, 3):
    if 'w:styleId="TOC%d"' % lvl not in s:
        continue          # 참조 docx 에 없으면 워드가 기본 목차 스타일을 쓴다
    restyle("TOC%d" % lvl,
            '<w:pPr><w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9350"/></w:tabs>'
            '<w:spacing w:before="0" w:after="0" w:line="264" w:lineRule="auto"/>'
            '<w:ind w:left="%d"/></w:pPr>' % ((lvl - 1) * 220), "")

# 그림·표 설명: 작게, 가운데, 회색
for sid in ("Caption", "ImageCaption", "TableCaption"):
    restyle(sid,
            '<w:pPr><w:jc w:val="center"/><w:spacing w:before="60" w:after="200"/></w:pPr>',
            '<w:rPr><w:rFonts ' + KO + '/><w:color w:val="4B5563"/>'
            '<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr>')
restyle("Figure",
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="160" w:after="0"/></w:pPr>', "")
restyle("CaptionedFigure",
        '<w:pPr><w:jc w:val="center"/><w:spacing w:before="160" w:after="0"/></w:pPr>', "")

# 인용문: 왼쪽 세로선 + 들여쓰기 (보고서의 '주의' 문단이 여기로 온다)
restyle("BlockText",
        '<w:pPr><w:pBdr><w:left w:val="single" w:sz="18" w:space="10" w:color="C7CDD6"/></w:pBdr>'
        '<w:ind w:left="240"/><w:spacing w:before="120" w:after="120"/></w:pPr>',
        '<w:rPr><w:rFonts ' + KO + '/><w:color w:val="374151"/>'
        '<w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>')

# 링크: 검정 · 밑줄 없음. 제출본은 종이로도 읽고, 상대 경로 링크는 PDF 에서 열리지
# 않는다 — 파란 밑줄만 남아 장식이 된다.
restyle("Hyperlink", "", '<w:rPr><w:color w:val="1F2937"/><w:u w:val="none"/></w:rPr>')

# 고정폭: 코드·파일경로
s = re.sub(r'(<w:style w:type="character" w:styleId="VerbatimChar">.*?)<w:rPr>.*?</w:rPr>',
           r'\1<w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="맑은 고딕" '
           r'w:cs="Consolas"/><w:sz w:val="19"/><w:szCs w:val="19"/>'
           r'<w:color w:val="1F2937"/></w:rPr>', s, flags=re.S)

# 표: 실선 격자 + 머리행 음영·굵게, 셀 여백
s = re.sub(
    r'<w:style w:type="table" w:default="1" w:styleId="Table">.*?</w:style>',
    '<w:style w:type="table" w:default="1" w:styleId="Table"><w:name w:val="Table"/>'
    '<w:basedOn w:val="TableNormal"/><w:qFormat/>'
    '<w:pPr><w:spacing w:before="20" w:after="20" w:line="276" w:lineRule="auto"/></w:pPr>'
    '<w:rPr><w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr>'
    '<w:tblPr><w:tblInd w:w="0" w:type="dxa"/>'
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="6" w:space="0" w:color="8A94A6"/>'
    '<w:left w:val="single" w:sz="6" w:space="0" w:color="8A94A6"/>'
    '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="8A94A6"/>'
    '<w:right w:val="single" w:sz="6" w:space="0" w:color="8A94A6"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="B6BDC9"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="B6BDC9"/>'
    '</w:tblBorders>'
    '<w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="108" w:type="dxa"/>'
    '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="108" w:type="dxa"/></w:tblCellMar>'
    '</w:tblPr>'
    '<w:tblStylePr w:type="firstRow"><w:rPr><w:b/></w:rPr>'
    '<w:tcPr><w:shd w:val="clear" w:color="auto" w:fill="EDF0F4"/>'
    '<w:vAlign w:val="center"/></w:tcPr></w:tblStylePr>'
    '</w:style>', s, flags=re.S)

write("word/styles.xml", s)

# ── 쪽 번호 바닥글 ────────────────────────────────────────────
write("word/footer1.xml",
      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
      '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="0" w:after="0"/></w:pPr>'
      '<w:r>' + FT + '<w:fldChar w:fldCharType="begin"/></w:r>'
      '<w:r>' + FT + '<w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
      '<w:r>' + FT + '<w:fldChar w:fldCharType="separate"/></w:r>'
      '<w:r>' + FT + '<w:t>1</w:t></w:r>'
      '<w:r>' + FT + '<w:fldChar w:fldCharType="end"/></w:r>'
      '</w:p></w:ftr>')

r = read("word/_rels/document.xml.rels")
assert "footer1.xml" not in r, "이미 바닥글이 붙어 있다 — ref 를 다시 풀 것"
r = r.replace("</Relationships>",
              '<Relationship Id="rIdFooterKo" '
              'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" '
              'Target="footer1.xml"/></Relationships>')
write("word/_rels/document.xml.rels", r)

c = read("[Content_Types].xml")
c = c.replace("</Types>",
              '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-'
              'officedocument.wordprocessingml.footer+xml"/></Types>')
write("[Content_Types].xml", c)

# ── 용지: A4, 상하좌우 20mm, 바닥글 연결 ──────────────────────
d = read("word/document.xml")
d = d.replace("<w:sectPr>",
              '<w:sectPr><w:footerReference w:type="default" r:id="rIdFooterKo"/>'
              '<w:pgSz w:w="11906" w:h="16838"/>'
              '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" '
              'w:header="680" w:footer="567" w:gutter="0"/>')
write("word/document.xml", d)

# ── 다시 압축 ────────────────────────────────────────────────
if os.path.exists(OUT):
    os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(REF):
        for f in files:
            full = os.path.join(root, f)
            z.write(full, os.path.relpath(full, REF).replace("\\", "/"))
shutil.rmtree(REF, ignore_errors=True)
print("  서식 틀 %s (%.1f KB)" % (os.path.basename(OUT), os.path.getsize(OUT) / 1024))
