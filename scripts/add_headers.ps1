# add_headers.ps1
# Add standardized file headers to source files (.py, .sh, .c, .h)
# Usage: Run from repository root in PowerShell

$Author = 'Denis Kurka'
$Year = '2025'
$License = 'CC0'
$SkipMarker = "Author: $Author"
$Exts = @('*.py','*.sh','*.c','*.h')

function Get-Brief($path, $text, $ext) {
    # Skip shebang and encoding
    $lines = $text -split("\r?\n")
    $idx = 0
    if ($lines[0] -match '^#!') { $idx = 1 }
    if ($idx -lt $lines.Length -and ($lines[$idx] -match '#.*coding[:=]')) { $idx++ }

    if ($ext -eq '.py') {
        # try module docstring without complex regex
        $endIdx = [Math]::Min($idx+9,$lines.Length-1)
        $joined = ($lines[$idx..$endIdx]) -join "`n"
        $doc = $null
        if ($joined.Contains('"""')) {
            $parts = $joined -split '"""',3
            if ($parts.Length -ge 3) { $doc = $parts[1] }
        } elseif ($joined.Contains("'''")) {
            $parts = $joined -split "'''",3
            if ($parts.Length -ge 3) { $doc = $parts[1] }
        }
        if ($doc) {
            $first = ($doc -split "`n")[0]
            $firstTrim = $first.Trim()
            return $firstTrim.Substring(0,[Math]::Min(77,$firstTrim.Length))
        }
    }
    # look for first comment line
    $endIdx2 = [Math]::Min($idx+9,$lines.Length-1)
    for ($i=$idx; $i -le $endIdx2; $i++) {
        $ln = $lines[$i].Trim()
        if ($ln -match '^(#|//)\s*(.+)') { return $matches[2] }
        if ($ln -match '^/\*\s*(.+)') { return $matches[1] }
    }
    return $path.Name
}

function Make-Header($ext, $brief) {
    if ($ext -in @('.py','.sh')) {
        return "# File: $brief`n# Author: $Author`n# Year: $Year`n# License: $License`n`n"
    } else {
        return "/*`n * File: $brief`n * Author: $Author`n * Year: $Year`n * License: $License`n */`n`n"
    }
}

$Root = Get-Location
$files = Get-ChildItem -Recurse -Include $Exts -File
$updated = 0

foreach ($f in $files) {
    $ext = $f.Extension.ToLower()
    $text = Get-Content -Raw -LiteralPath $f.FullName -ErrorAction SilentlyContinue
    if (-not $text) { continue }
    if ($text -match [regex]::Escape($SkipMarker)) { continue }
    $brief = Get-Brief -path $f -text $text -ext $ext
    $header = Make-Header -ext $ext -brief $brief
    $lines = $text -split("\r?\n")
    $insertAt = 0
    if ($lines[0] -match '^#!') { $insertAt = 1 }
    if ($insertAt -lt $lines.Length -and ($lines[$insertAt] -match '#.*coding[:=]')) { $insertAt++ }
    $before = $lines[0..($insertAt-1)] -join "`n"
    if ($before.Length -gt 0) { $before = $before + "`n" }
    $after = $lines[$insertAt..($lines.Length-1)] -join "`n"
    $newText = $before + $header + $after
    Set-Content -LiteralPath $f.FullName -Value $newText -Encoding UTF8
    Write-Host "Updated: $($f.FullName.Substring($Root.Path.Length+1))"
    $updated++
}

Write-Host "Headers added to $updated files." -ForegroundColor Green
