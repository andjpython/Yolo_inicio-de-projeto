# -*- coding: utf-8 -*-
r"""
YOLOv11 - CPU Only (Windows)
Projeto mínimo para imagem, vídeo, pasta e webcam
Estrutura pensada para: D:\AREA DO PROGRAMADOR\PROJETO YOLO

Como usar (PowerShell ou CMD):
  # imagem
  python man.py --source "D:\AREA DO PROGRAMADOR\PROJETO YOLO\bus.jpg"

  # vídeo
  python man.py --source "D:\AREA DO PROGRAMADOR\PROJETO YOLO\teste.mp4"

  # webcam (índice 0)
  python man.py --source 0 --show

  # pasta (curinga) - todas JPG da pasta imagens
  python man.py --source "D:\AREA DO PROGRAMADOR\PROJETO YOLO\imagens\*.jpg"

  # stream RTSP
  python man.py --source "rtsp://usuario:senha@192.168.0.10:554/Streaming/Channels/101"

Verificação rápida no PowerShell:
  Test-Path ".\bus.jpg"
  Se der False, então o bus.jpg não está na pasta atual (talvez foi movido).
  Você pode localizar com: Get-ChildItem -Recurse -Filter bus.jpg
  Ou passar o caminho absoluto entre aspas: "D:\AREA DO PROGRAMADOR\PROJETO YOLO\bus.jpg"

Saída dos resultados:
  D:\AREA DO PROGRAMADOR\PROJETO YOLO\runs\<name>\*

"""

import argparse
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# --- Configs padrão do projeto ---
PROJECT_DIR = Path(r"D:\AREA DO PROGRAMADOR\PROJETO YOLO").resolve()
RUNS_DIR    = PROJECT_DIR / "runs"
MODEL_PATH  = PROJECT_DIR / "yolo11n.pt"   # troque para yolo11s.pt se quiser melhor precisão

# Hiperparâmetros padrão
DEFAULT_IMGSZ = 640
DEFAULT_CONF  = 0.6
DEFAULT_IOU   = 0.45

USE_DEVICE = "cpu"
USE_HALF   = False   # half = False em CPU


def ensure_exists(path: Union[str, Path]):
    p = Path(str(path))
    if str(path).isdigit():  # webcam index (0/1/2...)
        return
    # Para curinga (*.jpg) não dá para validar totalmente, então checamos o pai
    if "*" in str(p):
        parent = p.parent
        if not parent.exists():
            raise FileNotFoundError(f"Pasta não encontrada: {parent}")
        return
    if not p.exists():
        raise FileNotFoundError(f"Fonte não encontrada: {p}")


def load_model():
    from ultralytics import YOLO
    # baixa automaticamente se não existir localmente
    model = YOLO("yolo11n.pt")
    return model




def predict_generic(model, source, imgsz, conf, iou, project, name, show, vid_stride):
    """
    Usa API de alto nível do Ultralytics.
    Funciona para: arquivo de imagem, arquivo de vídeo, curinga de pasta, stream RTSP.
    """
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=USE_DEVICE,
        half=USE_HALF,
        vid_stride=vid_stride,
        show=show,        # abre janelas (evite em notebook)
        save=True,
        project=str(project),
        name=name,
        exist_ok=True,
        verbose=False
    )
    # results é uma lista de Results; todos compartilham o mesmo save_dir
    print("✅ Saídas salvas em:", results[0].save_dir)
    return results


def webcam_loop(model, cam_index: int, imgsz, conf, iou):
    """
    Loop OpenCV com overlay de FPS. Pressione 'q' ou 'ESC' para sair.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a webcam de índice {cam_index}")

    frames = 0
    import time
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            r = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                device=USE_DEVICE,
                half=USE_HALF,
                vid_stride=1,
                verbose=False
            )
            annotated = r[0].plot()
            frames += 1

            if frames % 10 == 0:
                fps = frames / (time.time() - start)
                cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("YOLOv11 CPU - Webcam (q/esc para sair)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv11 CPU Only - Detector genérico (imagem, vídeo, pasta, webcam, RTSP)."
    )
    parser.add_argument("--source", required=True,
                        help="Caminho de imagem/vídeo/pasta (curinga) ou índice da webcam (ex.: 0) ou URL RTSP.")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Tamanho do lado menor (default: 640).")
    parser.add_argument("--conf",  type=float, default=DEFAULT_CONF,  help="Confiança mínima (default: 0.6).")
    parser.add_argument("--iou",   type=float, default=DEFAULT_IOU,   help="IoU para NMS (default: 0.45).")
    parser.add_argument("--name",  default="run_cpu", help="Nome da subpasta dentro de runs/.")
    parser.add_argument("--project", default=str(RUNS_DIR), help="Pasta raiz dos resultados (default: runs/).")
    parser.add_argument("--show", action="store_true", help="Mostrar janelas (OpenCV).")
    parser.add_argument("--vid-stride", type=int, default=2, help="Pular frames em vídeo/stream (default: 2).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Normaliza webcam index se for número puro
    source = args.source
    if source.isdigit():
        source = int(source)

    # Valida existência quando aplicável
    try:
        ensure_exists(source)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Carrega o modelo
    try:
        model = load_model()
        print(f"✔ Modelo carregado: {MODEL_PATH.name}")
        print(f"✔ Dispositivo: {USE_DEVICE} | half={USE_HALF} | imgsz={args.imgsz}")
    except Exception as e:
        print(f"❌ Erro ao carregar o modelo: {e}")
        sys.exit(1)

    # Webcam: usamos loop customizado
    if isinstance(source, int):
        webcam_loop(model, source, args.imgsz, args.conf, args.iou)
        return

    # Demais fontes: API de alto nível
    try:
        predict_generic(
            model=model,
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            project=Path(args.project),
            name=args.name,
            show=args.show,
            vid_stride=args.vid_stride
        )
    except Exception as e:
        print(f"❌ Falha na inferência: {e}")
        print("Dicas:")
        print("  - Verifique se o arquivo não está vazio (0 KB) e abre no VLC/Media Player.")
        print("  - Para .mp4/.avi, ter o FFmpeg no sistema ajuda o OpenCV a decodificar.")
        print("  - Ajuste --vid-stride para 1 se quiser processar todos os frames.")
        sys.exit(1)


if __name__ == "__main__":
    main()

