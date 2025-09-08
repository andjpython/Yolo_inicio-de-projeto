# Yolo_inicio-de-projeto

Projeto inicial com YOLOv11 (CPU) no Windows.

## Pré-requisitos
- Python 3.10+ (recomendado 3.10 ou 3.11)
- PowerShell ou CMD

## Instalação rápida
```powershell
# dentro da pasta do projeto
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```

## Arquivos úteis
- `man.py`: script principal para inferência (imagem, vídeo, pasta, webcam, RTSP)
- `yolo11n.pt`: peso leve do YOLOv11 (baixado automaticamente se não existir)
- `dependecias.txt`: exemplos de comandos de uso
- `runs/`: saídas das execuções

## Exemplos de uso
```powershell
# imagem
python man.py --source .\bus.jpg

# vídeo
python man.py --source .\teste.mp4 --show --vid-stride 2

# webcam (índice 0)
python man.py --source 0 --show

# pasta (curinga)
python man.py --source .\imagens\*.jpg

# RTSP
python man.py --source "rtsp://usuario:senha@192.168.0.10:554/Streaming/Channels/101"
```

As saídas são salvas em `runs/<name>/` (por padrão `runs/run_cpu`).

## Observações
- Este projeto está configurado para CPU (`device=cpu`, `half=False`).
- Para vídeos, ter FFmpeg no sistema ajuda o OpenCV na decodificação.