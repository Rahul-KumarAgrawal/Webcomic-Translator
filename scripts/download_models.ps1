 = @(
    @{
        Url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar";
        DestDir = "D:\Translate\Translator\model\paddle_cache\whl\det\ml";
        FileName = "Multilingual_PP-OCRv3_det_infer.tar"
    },
    @{
        Url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/japan_PP-OCRv4_rec_infer.tar";
        DestDir = "D:\Translate\Translator\model\paddle_cache\whl\rec\japan";
        FileName = "japan_PP-OCRv4_rec_infer.tar"
    },
    @{
        Url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/multilingual/korean_PP-OCRv4_rec_infer.tar";
        DestDir = "D:\Translate\Translator\model\paddle_cache\whl\rec\korean";
        FileName = "korean_PP-OCRv4_rec_infer.tar"
    }
)

foreach ( in ) {
    if (-not (Test-Path -Path .DestDir)) {
        New-Item -ItemType Directory -Force -Path .DestDir | Out-Null
    }
    
     = Join-Path -Path .DestDir -ChildPath .FileName
    Write-Host "
Downloading ..."
    Invoke-WebRequest -Uri .Url -OutFile 
    
    Write-Host "Extracting ..."
    tar -xf  -C .DestDir
    
    Write-Host "Cleaning up ..."
    Remove-Item  -Force
}
Write-Host "All models downloaded and extracted!"
