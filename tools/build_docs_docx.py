# -*- coding: utf-8 -*-
"""docs/final 의 제출 문서를 docx 로 만든다.

markdown 이 원본이고 docx 는 산출물이다. 원본을 고친 뒤 이 스크립트를 다시 돌리면
docx 가 따라온다 — 손으로 고친 docx 는 다음 실행에서 지워지니 주의할 것.

필요한 것
  * pandoc   — PATH 에 없으면 `pip install pypandoc_binary` 한 뒤 그 파이썬으로 돌리거나
               --pandoc 으로 경로를 준다
  * Chrome   — 그림(SVG)을 PNG 로 굽는 데 쓴다

사용
  python tools/build_docs_docx.py
  powershell tools/update_docx_fields.ps1     # 목차·쪽번호 채우기 (워드 필요)

서식(글꼴·용지·제목 크기)은 tools/docx_reference.py 가 만든다.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

# pandoc 의 markdown. gfm 은 이미지 폭 지정(link_attributes)을 못 받는다.
#   -smart              따옴표·줄표를 멋대로 바꾸지 않는다
#   -subscript          '3~6세', '21~27%' 의 물결을 아래첨자로 읽지 않게
#   +gfm_auto_identifiers  제목 id 를 깃허브와 같게 — report.md 의 본문 목차 링크가 이걸 쓴다
#
# auto_identifiers 를 끄면 안 된다. gfm_auto_identifiers 는 그 위에서 방식만 바꾸는
# 것이라, 밑을 끄면 제목에 id 가 아예 안 붙고 본문 목차 링크가 전부 죽는다.
READER = "markdown-smart-subscript-superscript+gfm_auto_identifiers"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL = os.path.join(ROOT, "docs", "final")
OUTDIR = os.path.join(FINAL, "docx")

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# SVG 원본 → 내보낼 픽셀 크기. viewBox 와 같게 두고 2배로 굽는다.
FIGURES = {
    "pipeline-block.svg": (1000, 620),
    "wake-two-stage.svg": (1000, 600),
    "patent-system.svg": (1000, 600),     # 발명신고용 도 1·2 (흑백, 수치 없음)
    "patent-dialog.svg": (1000, 700),
}

# (원본 md, 결과 docx, 목차 생성 여부)
# 번호와 이름은 결과보고서 끝의 '별첨' 표와 같아야 한다.
DOCS = [
    ("결과보고서.md", "01_결과보고서.docx", True),
    ("report.md", "02_별첨1_기술상세보고서.docx", True),
    # 별첨 2 = 출원 서류 초안 두 장. 한글 양식에 옮겨 붙일 내용이라 목차는 없다.
    (os.path.join("patent", "발명신고서-초안.md"),
     "03_별첨2_발명신고서_초안.docx", False),
    (os.path.join("patent", "발명의내용설명서-초안.md"),
     "04_별첨2_발명의내용설명서_초안.docx", False),
    ("api-전환안.md", "05_별첨3_전면API구성검토.docx", False),
    # 별첨이 아니라 근거 자료 — 결과보고서 5.1 의 선행조사 과정이 여기에 있다.
    (os.path.join("patent", "wake-two-stage-verification.md"),
     "06_근거_호출어2단계검증_선행기술대비.docx", True),
]


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    sys.exit("크롬을 찾지 못했다. SVG 를 PNG 로 구울 수 없다.")


def render_figures(workdir):
    """SVG 를 2배 해상도 PNG 로 굽는다. 워드는 SVG 를 제대로 못 싣는다."""
    chrome = find_chrome()
    figdir = os.path.join(workdir, "figures")
    os.makedirs(figdir, exist_ok=True)
    for svg, (w, h) in FIGURES.items():
        src = os.path.join(FINAL, "figures", svg)
        dst = os.path.join(figdir, svg[:-4] + ".png")
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
             "--default-background-color=FFFFFFFF", "--force-device-scale-factor=2",
             "--window-size=%d,%d" % (w, h),
             "--screenshot=" + dst.replace("\\", "/"),
             "file:///" + src.replace("\\", "/")],
            capture_output=True, timeout=180)
        if not os.path.exists(dst):
            sys.exit("PNG 를 굽지 못했다: " + svg)
        print("  그림 %s → %dx%d px" % (svg, w * 2, h * 2))
    return figdir


def preprocess(md_path, workdir):
    """pandoc 에 넘기기 전에 markdown 을 손본다. 제목 줄을 메타데이터로 올리고,
    SVG 참조를 구운 PNG 로 바꾸고, 특허 문서에는 도면을 실제로 끼워 넣는다."""
    text = open(md_path, encoding="utf-8").read()

    # 첫 '# 제목' 줄을 떼어 pandoc 의 title 로 넘긴다 → Title 스타일이 붙는다
    m = re.match(r"#\s+(.+?)\n", text)
    title = m.group(1).strip() if m else os.path.basename(md_path)
    if m:
        text = text[m.end():].lstrip("\n")

    # 표지 표와 본문 사이의 가로줄은 뗀다 — 쪽이 갈리면서 새 쪽 머리에 줄만 남는다
    head, sep, body = text.partition("\n## ")
    text = re.sub(r"(?m)^-{3,}\s*$", "", head).rstrip() + "\n" + sep + body

    # 특허 문서 8절은 도면을 파일 경로로만 가리킨다 — docx 에는 그림을 실제로 넣는다
    text = text.replace(
        "파일: `../figures/wake-two-stage.svg` (벡터. 문서에 그대로 삽입 가능)",
        "![](figures/wake-two-stage.svg)\n\n"
        "*【도 1】 호출어 2단계 검증의 시점과 구간*")

    # 그림: 확장자를 바꾸고 본문 폭에 맞춘다
    text = re.sub(r"\]\((?:\.\./)?figures/([a-z-]+)\.svg\)",
                  r"](figures/\1.png){width=100%}", text)

    # 그림 바로 뒤의 한 줄짜리 *설명* 은 그림 설명글로 합친다. 워드에서 가운데 정렬된
    # 캡션이 되고, 본문 문단으로 어정쩡하게 남는 일이 없다. 여러 줄짜리 해설은 건드리지
    # 않는다 — 그건 캡션이 아니라 본문이다.
    def fold_caption(m):
        return "![%s](%s)%s\n" % (m.group(3), m.group(1), m.group(2))

    text = re.sub(
        r"!\[[^\]]*\]\((figures/[a-z-]+\.png)\)(\{[^}]*\})?\n\n\*([^*\n]+)\*\n(?=\n|$)",
        fold_caption, text)

    dst = os.path.join(workdir, os.path.basename(md_path))
    open(dst, "w", encoding="utf-8").write(text)
    return dst, title


PAGEBREAK = ('<w:p><w:pPr><w:spacing w:after="0"/></w:pPr>'
             '<w:r><w:br w:type="page"/></w:r></w:p>')


def move_toc_after_cover(docx):
    """목차를 표지(첫 표) 뒤로 내리고 앞뒤를 쪽 나눔으로 끊는다.

    pandoc 은 목차를 제목 바로 밑에 놓는데, 그러면 표지 표가 목차 뒤로 밀려난다.
    공문은 표지가 1쪽, 목차가 2쪽이다."""
    with zipfile.ZipFile(docx) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")

    m = re.search(r"<w:sdt>(?:(?!</w:sdt>).)*?TOC \\o.*?</w:sdt>", doc, re.S)
    if not m:
        return  # 목차가 없으면 그대로 둔다
    toc = m.group(0)
    rest = doc[:m.start()] + doc[m.end():]

    end = rest.find("</w:tbl>")
    if end == -1:
        return  # 표지 표가 없다 — 옮길 자리가 없으니 원래대로
    end += len("</w:tbl>")
    doc = rest[:end] + PAGEBREAK + toc + PAGEBREAK + rest[end:]

    parts["word/document.xml"] = doc.encode("utf-8")
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in parts.items():
            z.writestr(n, b)


def check_internal_links(docx):
    """문서 안쪽을 가리키는 링크가 실제 책갈피에 닿는지 본다.

    report.md 는 본문 첫머리에 손으로 쓴 목차를 두고 있어서, 제목 id 생성이 조용히
    꺼지면 링크 열두 개가 전부 죽은 채로 나간다. 겉보기로는 멀쩡해 보인다."""
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")
    anchors = set(re.findall(r'<w:hyperlink w:anchor="([^"]+)"', doc))
    marks = set(re.findall(r'<w:bookmarkStart[^>]*w:name="([^"]+)"', doc))
    dead = sorted(a for a in anchors if a not in marks)
    if dead:
        sys.exit("%s: 내부 링크 %d개가 끊겼다 — %s"
                 % (os.path.basename(docx), len(dead), ", ".join(dead[:3])))
    if anchors:
        print("    내부 링크 %d개 확인" % len(anchors))


def build(pandoc, reference):
    if os.path.isdir(OUTDIR):
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR)

    workdir = tempfile.mkdtemp(prefix="docx-build-")
    try:
        render_figures(workdir)
        for src, out, toc in DOCS:
            md, title = preprocess(os.path.join(FINAL, src), workdir)
            dst = os.path.join(OUTDIR, out)
            cmd = [pandoc, os.path.basename(md),
                   "--from", READER,
                   "--to", "docx",
                   "--standalone",
                   # 원본 표의 구분선(|---|)이 전부 같은 길이라, 이 값이 작으면 pandoc 이
                   # 칸 너비를 똑같이 박아 버린다. 크게 줘서 워드가 내용에 맞춰 잡게 한다.
                   "--columns=10000",
                   "--reference-doc", reference,
                   "--metadata", "title=" + title,
                   "--metadata", "lang=ko-KR",
                   "--metadata", "toc-title=목차",
                   # 문서 제목을 Title 로 올렸으니 나머지를 한 단계씩 당긴다
                   "--shift-heading-level-by=-1",
                   "--output", dst]
            if toc:
                cmd[-3:-3] = ["--toc", "--toc-depth=2"]
            r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit("pandoc 실패 (%s)\n%s" % (src, r.stderr))
            if r.stderr.strip():
                print("  [pandoc] " + r.stderr.strip())
            if toc:
                move_toc_after_cover(dst)
            check_internal_links(dst)
            size = os.path.getsize(dst)
            print("  %s → %s (%.0f KB)" % (src, out, size / 1024))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def find_pandoc():
    p = shutil.which("pandoc")
    if p:
        return p
    try:  # pip install pypandoc_binary 로 받아 둔 것
        import pypandoc
        return pypandoc.get_pandoc_path()
    except Exception:
        sys.exit("pandoc 이 없다. `pip install pypandoc_binary` 뒤 다시 돌리거나 "
                 "--pandoc 으로 경로를 줄 것.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pandoc")
    ap.add_argument("--reference", help="직접 만든 서식 틀. 없으면 그때그때 만든다")
    a = ap.parse_args()
    pandoc = a.pandoc or find_pandoc()

    tmp = None
    reference = a.reference
    if not reference:
        tmp = tempfile.mkdtemp(prefix="docx-ref-out-")
        reference = os.path.join(tmp, "reference-ko.docx")
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "docx_reference.py"),
                        pandoc, reference], check=True)

    print("docx 생성 →", OUTDIR)
    try:
        build(pandoc, reference)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print("끝. 워드에서 열어 목차 필드를 갱신할 것(F9) — 또는 tools/update_docx_fields.ps1")
