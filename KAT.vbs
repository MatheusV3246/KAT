Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd.exe /c python app.py", 0, False

MsgBox "KAT iniciado com sucesso!", vbInformation, "KAT"

