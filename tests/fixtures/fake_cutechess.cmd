@echo off
REM Cross-platform launcher shim for the fake cutechess-cli test fixture.
REM On Windows a .cmd file is needed so subprocess can execute it directly.
python "%~dp0fake_cutechess.py" %*
