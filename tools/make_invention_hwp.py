# -*- coding: utf-8 -*-
"""세종대 산학협력단 발명신고 양식(HWP)을 채워 새 파일로 낸다.

양식 원본은 이 저장소 밖에 있고(jaeha_bot_docs/docs/제출 보고서), 원본은 읽기만 한다.
넣는 내용의 출처는 docs/final/patent/ 의 두 초안 문서다.

사용
  python tools/make_invention_hwp.py "<양식 폴더>"
  python tools/make_invention_hwp.py "<양식 폴더>" --hwpml <신고서.xml> <설명서.xml>

도면 두 장은 docs/final/figures 의 SVG 를 크롬으로 구워 넣는다(FIGS).

만드는 방식
  * 체크박스는 HWPML 의 Value 를 고친다. 양식 개체라서 글자가 아니다.
  * 글자는 COM 으로 칸에 넣는다. 칸은 구역(list) 번호로 찾고, 쓰기 전에 그 칸에
    있어야 하는 문구를 확인한다 — 엉뚱한 칸에 쓰는 사고를 막는 장치다.

한글 자동화에서 부딪힌 것 세 가지. 전부 조용히 멈추거나 터진다.

🔴 1. 파일을 읽고 쓰는 동작(열기·저장·그림 넣기)이 멈춘다면 **보안 승인 팝업**이 사람의
   클릭을 기다리는 것이다(2026-09-19 확인: 창 제목 '한글'인 대화상자가 떠 있었다).
   한글 2024(13.0.0.2151)에서 `RegisterModule(...)` 이 늘 False 이고, pyhwpx 에 딸린 DLL 은
   32비트 프로세스에서 잘 올라오며 레지스트리 이름도 맞는데도 거부된다 — 모듈이 먹지
   않으니 한글 프로세스를 새로 띄울 때마다 팝업이 뜬다. 보안 모듈 DLL 은 레지스트리
   HKCU\\Software\\HNC\\HwpAutomation\\Modules 로 등록하는 방식이며 regsvr32 대상이 아니다.
   각 단계에 시간 제한을 두어 무한정 기다리지 않게 했고, --hwpml 로 양식 열기를 건너뛸 수
   있다(미리 받아 둔 HWPML 을 문자열로 넘기면 파일을 읽지 않는다).

🔴 2. SetTextFile 로 올린 문서를 고친 뒤 save_as 를 부르면 영구히 멈춘다. 고치지 않은
   상태의 save_as 는 즉시 끝난다. 그래서 '고치기'와 '저장'을 나눈다.

🔴 3. 그 둘을 한 프로세스에서 인스턴스만 바꿔 하면 '개체가 서버에 연결되지 않았습니다'
   로 터진다. quit 한 프로세스가 죽는 중인데 거기에 새 문서를 붙이기 때문이다.
   그래서 단계마다 별도 파이썬 프로세스로 돈다(--phase).
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))



def load_content():
    spec = importlib.util.spec_from_file_location(
        "desc_content", os.path.join(ROOT, "tools", "invention_desc_content.py"))
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    return C


TITLE_KO = load_content().TITLE_KO

SRC_SINKO = "[양식] 01.발명신고서 등_2023 (1).hwp"
SRC_DESC = "[양식] 2. 발명의내용설명서 (3).hwp"
OUT_SINKO = "발명신고서_재하봇_초안.hwp"
OUT_DESC = "발명의내용설명서_재하봇_초안.hwp"

# 도면: (SVG, 원본 크기 px, 넣을 크기 mm). 발명신고용으로 흑백·수치 없이 따로 그린 것이다
# (보고서용 그림은 수치와 주석이 많아 칸 폭으로 줄이면 읽히지 않았다). 칸 안쪽 폭 130mm 남짓.
FIGS = [
    ("patent-system.svg", (1000, 600), (124, 74)),
    ("patent-wake.svg", (1000, 620), (124, 77)),
]

# 양식 본문의 글꼴. 양식 안내글(3.1~3.4 제목 포함)이 한컴바탕 10pt 이고, 그냥 쓰면
# 칸마다 다른 기본 글꼴이 딸려 온다(명칭 칸은 고정폭 글꼴로 나왔다).
BODY_FONT = ("한컴바탕", 10)


# 체크할 양식 개체. 이름은 양식이 들고 있는 것이고, 캡션으로 한 번 더 확인한다.
# 동의 항목(CheckBox18~25)은 본인이 직접 체크할 사항이라 건드리지 않는다.
CHECKS = [
    ("CheckBox30", "", "기술성숙도 — 연구실환경 테스트 (TRL 9칸 중 4번째)"),
    ("CheckBox4", "원천특허 확보", "출원목적"),
    ("CheckBox1", "대한민국", "출원희망국"),
    ("CheckBox11", "미공개", "발명의 공개여부"),
]

# 이 칸이 길어지면 신고서가 2쪽으로 넘어간다(양식은 1쪽 구성이다). 짧게 유지할 것.
ETC = [
    "정식 선행기술조사 요청. 2026. 9. 30. 교내 결과보고서 제출 예정으로, 해당 제출이 "
    "공개에 해당하는 경우 그 전에 출원이 필요합니다.",
]

# 🔴 구역번호가 큰 칸부터 쓴다. 칸을 고치면 그 뒤 구역번호가 밀린다 —
# 기타(B46)를 고치면 거기 붙어 있던 메모가 사라져 양도증 칸의 번호가 하나 당겨졌다.
# 내림차순으로 쓰면 아직 안 쓴 칸의 번호가 흔들리지 않는다.
PLAN_SINKO = [
    (168, "B2", "", [TITLE_KO]),          # 양도증의 발명의 명칭
    (160, "B46", "특허출원이 시급하거나", ETC),
    (11, "B2", "", [TITLE_KO]),           # 발명신고서의 발명의 명칭
]


def plan_desc():
    C = load_content()
    return [   # 위와 같은 이유로 내림차순
        (13, "B8", "반드시 작성하지 않아도", C.CHUNGGU),
        (11, "B7", "3.4 발명의 구성 및 작용", C.SANGSE_4),
        (10, "B6", "3.3 발명이 이루고자 하는", C.SANGSE_3),
        (9, "B5", "3.2 그 분야 종래기술", C.SANGSE_2),
        (8, "B4", "3.1 발명이 속하는 기술분야", C.SANGSE_1),
        (6, "B3", "도면이 있을 경우", C.DOMYEON),
        (4, "B2", "", C.MYEONGCHING),
    ]


# (칸 계획, 글꼴). 신고서 칸은 양식 글꼴이 그대로 따라와서 건드리지 않는다.
PLANS = {"sinko": (lambda: PLAN_SINKO, None), "desc": (plan_desc, BODY_FONT)}


def render_figs(work):
    """도면 SVG 를 2배 해상도 PNG 로 굽는다. 경로 목록을 돌려준다."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from build_docs_docx import find_chrome
    chrome, out = find_chrome(), []
    for svg, (w, h), _ in FIGS:
        src = os.path.join(ROOT, "docs", "final", "figures", svg)
        png = os.path.join(work, svg[:-4] + ".png")
        subprocess.run(
            [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
             "--default-background-color=FFFFFFFF", "--force-device-scale-factor=2",
             "--window-size=%d,%d" % (w, h), "--screenshot=" + png.replace("\\", "/"),
             "file:///" + src.replace("\\", "/")], capture_output=True, timeout=180)
        if not os.path.exists(png):
            sys.exit("도면을 굽지 못했다: " + svg)
        out.append(png)
    return out


def new_hwp():
    from pyhwpx import Hwp
    return Hwp(new=True, visible=False)


def check_boxes(xml):
    for name, caption, what in CHECKS:
        pat = re.compile(
            r'(<FORMOBJECT[^>]*Name="%s"[^>]*>.*?<BUTTONSET[^>]*Caption="%s"[^>]*Value=")0(")'
            % (re.escape(name), re.escape(caption)), re.S)
        xml, n = pat.subn(r"\g<1>1\g<2>", xml)
        if n != 1:
            sys.exit("%s(%s) 를 %d개 찾았다 — 1개여야 한다" % (name, what, n))
        print("  체크: %s" % what)
    return xml


def fill_cells(hwp, plan, font=None):
    for lid, addr, expect, lines in plan:
        if not hwp.set_pos(lid, 0, 0):
            sys.exit("구역 %d 로 이동 실패" % lid)
        got = hwp.get_cell_addr()
        hwp.MoveListBegin()
        hwp.MoveSelListEnd()
        cur = (hwp.get_selected_text() or "").replace("\r\n", "\n")
        hwp.Cancel()
        if got != addr:
            sys.exit("구역 %d 의 칸이 %s 여야 하는데 %s 다" % (lid, addr, got))
        if expect and expect not in cur:
            sys.exit("%s 칸에 %r 가 없다: %r" % (addr, expect, cur[:100]))

        hwp.MoveListBegin()
        hwp.MoveSelListEnd()
        hwp.Delete()
        for i, line in enumerate(lines):
            if i:
                hwp.BreakPara()
            if line:
                hwp.insert_text(line)
        # 양식의 안내글이 빨간색이라, 그 자리에 쓰면 글자색이 그대로 딸려온다.
        # 안내글은 '제출 시 삭제'하라는 것이므로 본문이 빨간 채로 나가면 안 된다.
        hwp.MoveListBegin()
        hwp.MoveSelListEnd()
        hwp.CharShapeTextColorBlack()
        if font:
            hwp.set_font(FaceName=font[0], Height=font[1], Bold=False)
        hwp.Cancel()
        print("  %s — %d줄" % (addr, len(lines)))


# ── 단계별 실행 (각각 별도 프로세스) ─────────────────────────────


def phase_dump(src, out_xml):
    hwp = new_hwp()
    if not hwp.open(src):
        sys.exit("양식을 열지 못했다: %s" % src)
    open(out_xml, "w", encoding="utf-8").write(hwp.GetTextFile("HWPML2X", ""))


def phase_fill(in_xml, out_xml, plan_name, figs, fig_list):
    hwp = new_hwp()
    if hwp.SetTextFile(open(in_xml, encoding="utf-8").read(), "HWPML2X", "") != 1:
        sys.exit("HWPML 을 문서로 올리지 못했다")
    plan, font = PLANS[plan_name]
    fill_cells(hwp, plan(), font)
    if figs:
        hwp.set_pos(fig_list, 0, 0)
        hwp.MoveListEnd()
        for png, (*_, (w, h)) in zip(figs, FIGS):
            hwp.BreakPara()
            # sizeoption=1 이 지정 크기다(0 은 원래 크기라 width 를 무시한다).
            hwp.insert_picture(png, embedded=True, sizeoption=1, width=w, height=h)
        print("  도면 %d장 삽입" % len(figs))
    open(out_xml, "w", encoding="utf-8").write(hwp.GetTextFile("HWPML2X", ""))


def phase_save(in_xml, out_hwp):
    hwp = new_hwp()
    if hwp.SetTextFile(open(in_xml, encoding="utf-8").read(), "HWPML2X", "") != 1:
        sys.exit("완성본을 올리지 못했다")
    hwp.save_as(out_hwp)      # 고치지 않은 상태라 멈추지 않는다


def hwp_pids():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Hwp.exe", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return {int(line.split('","')[1]) for line in out.splitlines() if line.startswith('"Hwp')}


def run(*args, timeout=300):
    # 🔴 단계는 os._exit 로 끝나서 한글을 닫지 않는다. 그 단계가 띄운 한글만 골라 끈다 —
    # 안 끄면 돌릴 때마다 숨은 한글이 쌓이고, 다음 열기가 멈춘다. 사용자가 원래 열어 둔
    # 한글은 before 에 들어 있으므로 건드리지 않는다.
    before = hwp_pids()
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--phase"] + list(args),
                           timeout=timeout)
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
        print("  %d초 안에 끝나지 않았다 — 한글 보안 팝업이 응답을 기다리는 중일 수 있다" % timeout)
    for pid in hwp_pids() - before:
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    if not ok:
        sys.exit("단계 실패: %s" % (args,))


def build(xml, plan_name, out, figs=(), fig_list=0):
    work = tempfile.mkdtemp(prefix="hwpfill-")
    a = os.path.join(work, "a.xml")
    b = os.path.join(work, "b.xml")
    open(a, "w", encoding="utf-8").write(xml)
    run("fill", a, b, plan_name, ";".join(figs), str(fig_list))
    run("save", b, out)
    shutil.rmtree(work, ignore_errors=True)
    print("  → %s" % os.path.basename(out))


def main(folder, xml_sinko, xml_desc):
    def hwpml(given, template):
        if given:
            return open(given, encoding="utf-8").read()
        work = tempfile.mkdtemp(prefix="hwpdump-")
        p = os.path.join(work, "t.xml")
        run("dump", os.path.join(folder, template), p)
        xml = open(p, encoding="utf-8").read()
        shutil.rmtree(work, ignore_errors=True)
        return xml

    print("발명신고서")
    build(check_boxes(hwpml(xml_sinko, SRC_SINKO)), "sinko",
          os.path.join(folder, OUT_SINKO))

    print("발명의 내용 설명서")
    figs = render_figs(tempfile.mkdtemp(prefix="hwpfig-"))
    build(hwpml(xml_desc, SRC_DESC), "desc",
          os.path.join(folder, OUT_DESC), figs=figs, fig_list=6)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--phase":
        kind = args[1]
        if kind == "dump":
            phase_dump(args[2], args[3])
        elif kind == "fill":
            phase_fill(args[2], args[3], args[4], [p for p in args[5].split(";") if p],
                       int(args[6]))
        elif kind == "save":
            phase_save(args[2], args[3])
        else:
            sys.exit("모르는 단계: %s" % kind)
        # 🔴 그냥 끝내면 프로세스가 안 죽는다 — COM 정리가 한글을 기다리며 멈춘다.
        # 할 일은 이미 파일로 다 나갔으므로 정리 없이 끊는다.
        sys.stdout.flush()
        os._exit(0)

    xs = xd = None
    if "--hwpml" in args:
        i = args.index("--hwpml")
        xs, xd = args[i + 1], args[i + 2]
        args = args[:i]
    if not args:
        sys.exit(__doc__)
    main(args[0], xs, xd)
