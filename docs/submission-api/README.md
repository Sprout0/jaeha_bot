# 제출본 — 전면 API 구성 (2026-09-22)

제출할 파일은 `제출본/` 폴더에 모두 있다. 이 폴더 밖의 옛 문서(`docs/final/`)는 로컬 구성 시점의
판이므로 제출에 쓰지 않는다.

| 제출본 파일 | 내용 | 원본 |
|---|---|---|
| `01_결과보고서.docx` / `.pdf` | 결과보고서 | `결과보고서.md` |
| `02_별첨1_기술상세보고서.docx` / `.pdf` | 별첨 1. 기술 상세 보고서 | `기술상세보고서.md` |
| `03_별첨2_발명신고서_전면API.hwp` | 별첨 2. 발명신고서 (세종대 산학협력단 양식) | `tools/make_invention_hwp.py --api` |
| `04_별첨2_발명의내용설명서_전면API.hwp` | 별첨 2. 발명의 내용 설명서 (청구항 10, 도면 3) | `tools/invention_desc_content_api.py` |

발명신고서에서 성명·주민등록번호·주소·지분·동의 체크는 비워 두었다. 본인이 직접 채운다.

## 다시 만들기

```bash
python tools/build_docs_docx.py --bundle api
powershell tools/update_docx_fields.ps1 -Dir docs/submission-api/제출본 -Pdf
```

한글 파일은 한글이 설치된 PC에서 `tools/make_invention_hwp.py "<폴더>" --api`로 만든다
(`발명신고서_재하봇_전면API.hwp`, `발명의내용설명서_재하봇_전면API.hwp`로 나오므로 번호를 붙여 넣는다).
