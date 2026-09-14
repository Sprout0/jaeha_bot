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
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        # pandoc 은 모든 칸을 같은 너비로 박아 놓는다. 워드에게 내용에 맞춰 다시 잡게 한 뒤
        # (1=내용에 맞춤) 본문 폭까지 늘린다(2=창에 맞춤). 비율은 내용이 정한다.
        foreach ($t in $doc.Tables) { $t.AutoFitBehavior(1); $t.AutoFitBehavior(2) }
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
