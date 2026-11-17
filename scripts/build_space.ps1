# Assemble the Hugging Face Space folder in .\space_build (git-ignored). Nothing is uploaded by this script.
#   powershell -ExecutionPolicy Bypass -File scripts\build_space.ps1 [-ModelDir runs\e2_long]
# Upload afterwards (only after `hf auth login`, done by the owner):
#   hf repo create medskel-ai-demo --repo-type space --space-sdk gradio --private
#   hf upload <user>/medskel-ai-demo space_build . --repo-type space
param([string]$ModelDir = "runs\e2_long")
$out = "space_build"
New-Item -ItemType Directory -Force $out, "$out\model", "$out\src\medskel_ai" | Out-Null
Copy-Item app.py, ABOUT.md, space\README.md, space\requirements.txt $out -Force
Copy-Item src\medskel_ai\__init__.py, src\medskel_ai\data.py, src\medskel_ai\model.py, src\medskel_ai\predict.py "$out\src\medskel_ai" -Force
Copy-Item "$ModelDir\best.pt" "$out\model" -Force
Copy-Item "$ModelDir\threshold*.json" "$out\model" -Force
# Example photographs are copied ONLY if the owner has put approved files into .\space_examples
if (Test-Path space_examples) { New-Item -ItemType Directory -Force "$out\examples" | Out-Null; Copy-Item space_examples\*.png "$out\examples" -Force }
Get-ChildItem $out -Recurse -File | ForEach-Object { "{0,10}  {1}" -f $_.Length, $_.FullName.Substring((Resolve-Path $out).Path.Length + 1) }
