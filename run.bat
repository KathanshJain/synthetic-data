@echo off
rem Start both servers (API :8000 + web :5173)
cd /d "%~dp0"
venv\Scripts\python.exe scripts\dev.py
