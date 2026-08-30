# Ghost bypass for WDDM dll lock (PID 7700 pattern)
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Id 7700 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
# Move-Item bypasses delete lock (proven)
if (Test-Path "I:\llama.cpp\build\bin\ggml-base.dll") {
  try { Move-Item "I:\llama.cpp\build\bin\ggml-base.dll" "C:\Users\james\AppData\Local\Temp\ggml-base-locked.dll" -Force; Write-Host "  moved locked dll" } catch {}
}
# Fallback: use build2 fresh dir when build\bin is ghost-locked
if (Test-Path "I:\llama.cpp\build\bin\ggml-base.dll") {
  Write-Host "  build still locked — using build2"
  cmake -B "I:\llama.cpp\build2" -S "I:\llama.cpp" -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_BUILD_TYPE=Release -G Ninja -DCMAKE_MAKE_PROGRAM="C:/Den/den-py314/Scripts/ninja.exe"
  cmake --build "I:\llama.cpp\build2" --config Release --target llama-server llama-bench
} else {
  cmake --build "I:\llama.cpp\build" --config Release --target llama-server llama-bench
}
