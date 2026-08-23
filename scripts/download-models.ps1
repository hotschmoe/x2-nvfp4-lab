param(
    [string]$ModelRoot = (Join-Path $PSScriptRoot '..\models')
)

$ErrorActionPreference = 'Stop'

$models = @(
    @{
        Name = 'Qwen3.8-27B-NVFP4-Unsloth'
        Repo = 'unsloth/Qwen3.8-27B-NVFP4'
        Files = @(
            'config.json', 'generation_config.json', 'model.safetensors.index.json',
            'model.safetensors', 'model_mtp.safetensors',
            'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'chat_template.jinja',
            'hf_quant_config.json', 'preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json'
        )
    },
    @{
        Name = 'Ornith-1.5-35B-A3B-NVFP4'
        Repo = 'ornith-ai/Ornith-1.5-35B-A3B-NVFP4'
        Files = @(
            'config.json', 'generation_config.json', 'model.safetensors.index.json',
            'tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json', 'chat_template.jinja',
            'hf_quant_config.json', 'preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json',
            'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors'
        )
    }
)

foreach ($model in $models) {
    $destination = Join-Path $ModelRoot $model.Name
    New-Item -ItemType Directory -Force -Path $destination | Out-Null
    foreach ($file in $model.Files) {
        $url = "https://huggingface.co/$($model.Repo)/resolve/main/$file?download=true"
        $out = Join-Path $destination $file
        # -C - is deliberate: each invocation safely resumes interrupted large LFS downloads.
        & curl.exe -L --fail --retry 8 --retry-all-errors -C - --output $out $url
        if ($LASTEXITCODE) { throw "Download failed: $url" }
    }
}
