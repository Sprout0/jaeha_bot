# docs/final/docx 의 문서를 워드로 한 번 열어 목차·쪽번호 필드를 채운다.
#
# pandoc 이 넣은 목차는 '필드'라서, 워드가 한 번 계산해 주기 전에는 비어 보인다.
# 받는 사람이 F9 를 눌러야 하는 문서를 넘길 수는 없으므로 여기서 미리 채워 둔다.
# -Pdf 를 주면 확인용 PDF 도 같이 뽑는다.
param(
    [string]$Dir = "$PSScriptRoot\..\docs\final\docx",
    [switch]$Pdf
)

$Dir = (Resolve-Path $Dir).Path
$files = Get-ChildItem -Path $Dir -Filter *.docx | Where-Object { $_.Name -notlike '~$*' }
if (-not $files) { Write-Error "docx 가 없다: $Dir"; exit 1 }

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    foreach ($f in $files) {
        $doc = $word.Documents.Open($f.FullName, $false, $false)
        # 본문뿐 아니라 머리글·바닥글의 필드까지 훑어야 쪽번호가 제대로 박힌다
        foreach ($story in $doc.StoryRanges) { $null = $story.Fields.Update() }
        foreach ($toc in $doc.TablesOfContents) {
            $toc.Update()
            # 목차 줄간격은 참조 서식으로 못 잡는다(워드가 자기 TOC 스타일을 쓴다).
            # 기본값이면 12쪽 문서의 목차가 두 쪽으로 넘쳐 한 쪽이 거의 빈다.
            foreach ($p in $toc.Range.Paragraphs) {
                $p.Format.SpaceBefore = 0
                $p.Format.SpaceAfter = 2
                $p.Format.LineSpacingRule = 0   # 0 = wdLineSpaceSingle
            }
        }
        # pandoc 은 모든 칸을 같은 너비로 박아 놓는다. 워드에게 내용에 맞춰 다시 잡게 한 뒤
        # (1=내용에 맞춤) 본문 폭까지 늘린다(2=창에 맞춤). 비율은 내용이 정한다.
        foreach ($t in $doc.Tables) {
            $t.AutoFitBehavior(1); $t.AutoFitBehavior(2)
            # 칸 하나가 쪽 경계에서 반으로 갈리지 않게.
            $t.Rows.AllowBreakAcrossPages = $false
            # 표를 안내 문장과 붙여 둔다. 작은 표(6줄 이하)는 통째로, 큰 표는 머리행과
            # 앞 두 줄만 — 머리행 하나만 쪽 끝에 남는 꼴을 막는다.
            $n = $t.Rows.Count
            $keep = if ($n -le 6) { $n - 1 } else { 2 }
            for ($i = 1; $i -le $keep; $i++) {
                $t.Rows($i).Range.ParagraphFormat.KeepWithNext = $true
            }
            $prev = $t.Range.Previous(4, 1)      # 4 = wdParagraph
            if ($prev) { $prev.ParagraphFormat.KeepWithNext = $true }
        }
        # 표·목록 바로 뒤 문단은 붙어 보인다. 그 문단에만 위 여백을 준다.
        $prevWasBlock = $false
        foreach ($p in $doc.Paragraphs) {
            $isList = ($p.Range.ListFormat.ListType -ne 0)
            $inTable = $p.Range.Information(12)  # 12 = wdWithInTable
            if ($prevWasBlock -and -not $isList -and -not $inTable) {
                $p.Format.SpaceBefore = 8
            }
            $prevWasBlock = ($isList -or $inTable)
        }
        $doc.Repaginate()
        $pages = $doc.ComputeStatistics(2)   # wdStatisticPages
        $doc.Save()
        if ($Pdf) {
            $out = [IO.Path]::ChangeExtension($f.FullName, '.pdf')
            $doc.ExportAsFixedFormat($out, 17)   # wdExportFormatPDF
        }
        $doc.Close($false)
        Write-Output ("{0} — {1}쪽" -f $f.Name, $pages)
    }
}
finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
