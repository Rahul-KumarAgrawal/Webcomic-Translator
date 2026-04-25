 = @(
    @{
        Url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar";
        DestDir = "D:\Translate\Translator\model\paddle_cache\whl\det\en";
        FileName = "en_PP-OCRv3_det_infer.tar"
    },
    @{
        Url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar";
        DestDir = "D:\Translate\Translator\model\paddle_cache\whl\rec\en";
        FileName = "en_PP-OCRv4_rec_infer.tar"
    }
)

foreach ( in ) {
    if (-not (Test-Path -Path .DestDir)) {
        New-Item -ItemType Directory -Force -Path .DestDir | Out-Null
    }
    
     = Join-Path -Path .DestDir -ChildPath .FileName
    Invoke-WebRequest -Uri .Url -OutFile 
    
    tar -xf  -C .DestDir
    Remove-Item  -Force
}
