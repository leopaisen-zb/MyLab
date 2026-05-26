@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "h:\BaiduNetdiskDownload\mylab(1)\mylab(1)"
"C:\ProgramData\miniconda3\Scripts\conda.exe" run -n matgen python "h:\BaiduNetdiskDownload\mylab(1)\mylab(1)\matgen_app\benchmark_batch.py"
