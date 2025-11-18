$batchSize = 5
$total = 13
for ($start = 1; $start -le $total; $start += $batchSize) {
    $end = [math]::Min($start + $batchSize - 1, $total)
    Write-Host "Running batch $start to $end"
    $processes = @()
    for ($i = $start; $i -le $end; $i++) {
        $videoPath = "./deer/$i.mp4"
        if (Test-Path $videoPath) {
            Write-Host "Starting process for $videoPath"
            $proc = Start-Process -NoNewWindow -FilePath "python" -ArgumentList "deer.py --use_pytorch --pytorch_model ./deer.pt --unsafe_dist 100 --danger_dist 100 --video $videoPath" -PassThru
            $processes += $proc
        } else {
            Write-Host "Video $videoPath not found, skipping"
        }
    }
    foreach ($p in $processes) {
        $p.WaitForExit()
    }
}