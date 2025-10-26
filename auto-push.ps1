$interval = 60

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "自動Gitプッシュスクリプト起動" -ForegroundColor Cyan
Write-Host "チェック間隔: $interval 秒" -ForegroundColor Cyan
Write-Host "Ctrl+C で終了できます" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

while ($true) {
    $timestamp = Get-Date -Format "yyyy/MM/dd HH:mm:ss"
    Write-Host "[$timestamp] 変更をチェック中..." -ForegroundColor Yellow
    
    $status = git status --porcelain
    
    if ($status) {
        Write-Host "[$timestamp] 変更が検出されました！" -ForegroundColor Green
        Write-Host "変更内容:" -ForegroundColor Gray
        Write-Host $status -ForegroundColor Gray
        
        Write-Host "  -> ファイルをステージング中..." -ForegroundColor Cyan
        git add -A
        
        $commitMessage = "Auto commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Host "  -> コミット中: $commitMessage" -ForegroundColor Cyan
        git commit -m $commitMessage
        
        Write-Host "  -> プッシュ中..." -ForegroundColor Cyan
        git push origin main
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "   プッシュ完了！" -ForegroundColor Green
        } else {
            Write-Host "   プッシュに失敗しました" -ForegroundColor Red
        }
        
        Write-Host ""
    } else {
        Write-Host "[$timestamp] 変更なし" -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds $interval
}
