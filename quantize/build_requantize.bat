@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cl /nologo /std:c++17 /O2 /EHsc /I E:\workspace\s2.cpp\ggml\include E:\workspace\s2.cpp\quantize\requantize_q8.cpp E:\workspace\s2.cpp\build-cuda\ggml\src\ggml-base.lib /Fe:E:\workspace\s2.cpp\quantize\requantize_q8.exe
