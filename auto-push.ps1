# 自動Git プッシュスクリプト
# 1分ごとに変更をチェックして、変更があれば自動的にコミット＆プッシュします

# 設定
$interval = 60  # チェック間隔（秒） - 1分

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "自動Gitプッシュスクリプト起動" -ForegroundColor Cyan
Write-Host "チェック間隔: $interval 秒" -ForegroundColor Cyan
Write-Host "Ctrl+C で終了できます" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 無限ループ
while ($true) {
    $timestamp = Get-Date -Format "yyyy/MM/dd HH:mm:ss"
    
    Write-Host "[$timestamp] 変更をチェック中..." -ForegroundColor Yellow
    
    # Gitステータスを取得
    $status = git status --porcelain
    
    if ($status) {
        Write-Host "[$timestamp] 変更が検出されました！" -ForegroundColor Green
        Write-Host "変更内容:" -ForegroundColor Gray
        Write-Host $status -ForegroundColor Gray
        
        try {
            # すべての変更をステージング
            Write-Host "  → ファイルをステージング中..." -ForegroundColor Cyan
            git add -A
            
            # コミット
            $commitMessage = "Auto commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            Write-Host "  → コミット中: $commitMessage" -ForegroundColor Cyan
            git commit -m $commitMessage
            
            # プッシュ
            Write-Host "  → プッシュ中..." -ForegroundColor Cyan
            git push origin main
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  ✓ プッシュ完了！" -ForegroundColor Green
            } else {
                Write-Host "  ✗ プッシュに失敗しました" -ForegroundColor Red
                Write-Host "  エラーコード: $LASTEXITCODE" -ForegroundColor Red
            }
        }
        catch {
            Write-Host "  ✗ エラーが発生しました: $_" -ForegroundColor Red
        }
        
        Write-Host ""
    } else {
        Write-Host "[$timestamp] 変更なし" -ForegroundColor Gray
    }
    
    # 指定された間隔だけ待機
    Start-Sleep -Seconds $interval
}

