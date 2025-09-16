#!/usr/bin/env python3
import argparse, time, sys, os, logging
import cv2
from ultralytics import YOLO
import ultralytics
import time

# Ultralytics loglarını sustur (isteğe bağlı)
ultralytics.utils.LOGGER.setLevel(logging.CRITICAL)



def parse_args():
    ap = argparse.ArgumentParser("YOLOv8 Tracking Bench (CPU/MPS/CoreML)")
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", choices=["cpu","mps","coreml"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--model_pt", default="yolov8n.pt")
    ap.add_argument("--model_ml", default="yolov8n.mlpackage")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--print_every", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--warmup", type=int, default=5, help="İlk N kareyi ortalamadan hariç tut")
    return ap.parse_args()

def draw_text(img, text, org=(12, 32)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

def main():
    args = parse_args()

    # Cihaz & model
    device = "cpu"
    if args.mode == "mps":
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            device = "mps"
    if args.mode == "coreml":
        model = YOLO(args.model_ml, task="detect")
        engine = "coreml"
    else:
        model = YOLO(args.model_pt)
        engine = device

    print(f"Engine: {args.mode} | Device: {engine} | imgsz={args.imgsz}")

    # Video meta
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        sys.exit(f"❌ Video açılamadı: {args.source}")
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps0 = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps0, (w0, h0))

    # Track jeneratörü
    kw = dict(
        source=args.source, tracker=args.tracker, conf=args.conf, iou=args.iou,
        max_det=args.max_det, imgsz=args.imgsz, stream=True, persist=True, verbose=False
    )
    if args.mode in ("cpu","mps"):
        kw["device"] = engine
    gen = model.track(**kw)

    start_time = time.time()

    # FPS ölçümü (doğru yöntem)
    # - t_prev_end: bir önceki iterasyonun BİTİŞ zamanı
    # - İki iterasyon arası geçen süre = inference+tracking+decode vb. toplam süre (pipeline)
    t_start = time.perf_counter()
    t_prev_end = t_start
    frames = 0
    ema_pipeline = None
    ema_infer = None
    a = args.alpha
    warm = args.warmup  # ilk N kare hariç

    try:
        for res in gen:
            t_loop_start = time.perf_counter()
            # Pipeline dt: önceki loop bitişinden bu loop başlangıcına
            if frames > 0:
                dt_pipe = t_loop_start - t_prev_end
                if dt_pipe > 0:
                    fps_pipe = 1.0 / dt_pipe
                    ema_pipeline = fps_pipe if ema_pipeline is None else a*ema_pipeline + (1-a)*fps_pipe

            # Inference-only (ms) varsa al
            infer_ms = None
            if hasattr(res, "speed") and isinstance(res.speed, dict):
                infer_ms = res.speed.get("inference", None)
                if infer_ms:
                    fps_inf = 1000.0 / infer_ms
                    ema_infer = fps_inf if ema_infer is None else a*ema_infer + (1-a)*fps_inf

            frame = res.plot()  # çizim (isteğe bağlı, ölçüme dâhil etmek istersen t_prev_end'i buradan sonra güncellemek doğru)
            if writer: writer.write(frame)
            if args.show:
                # Overlay: pipeline ve inference fps (varsa)
                line = f"FPS(pipeline): {ema_pipeline:5.1f}" if ema_pipeline else ""
                if ema_infer:
                    line = (line + "  |  " if line else "") + f"FPS(infer): {ema_infer:5.1f}"
                if line:
                    draw_text(frame, line)
                cv2.imshow("YOLO Track Bench", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            t_prev_end = time.perf_counter()
            frames += 1
            if args.print_every and frames % args.print_every == 0:
                if ema_pipeline:
                    msg = f"[{frames:05d}] FPS(pipeline, EMA): {ema_pipeline:5.1f}"
                    if ema_infer:
                        msg += f" | FPS(infer, EMA): {ema_infer:5.1f}"
                    print(msg)
    finally:
        if writer: writer.release()
        if args.show: cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    eff_frames = max(0, frames - warm)
    avg_pipeline = (eff_frames / elapsed) if elapsed > 0 else 0.0  # uçtan uca ortalama
    # Not: Warmup’u tamamen dışlamak istersen, warmup boyunca zaman toplamını ayrı tutup düşebilirsin.

    print("\nDone.",
          f"Frames: {frames}",
          f"| Avg FPS (pipeline): {avg_pipeline:5.1f}",
          f"| EMA pipeline: {ema_pipeline:5.1f}" if ema_pipeline else "",
          f"| EMA infer: {ema_infer:5.1f}" if ema_infer else "",
          sep=" ")
    
    end_time = time.time()
    print(frames / (end_time - start_time))
    


if __name__ == "__main__":
    main()
