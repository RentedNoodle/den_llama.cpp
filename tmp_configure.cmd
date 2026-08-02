call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" > NUL 2>&1
cd /d D:\den_llama.cpp
rmdir /s /q build_win 2>NUL
"C:\Program Files\CMake\bin\cmake.exe" -B build_win -G "Visual Studio 17 2022" -A x64 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a
echo EXIT CODE: %ERRORLEVEL%
