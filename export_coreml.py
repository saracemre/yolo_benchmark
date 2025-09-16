from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.export(
        format="coreml",
        imgsz=640,     # hız için 320/416 da deneyebilirsiniz
        half=True,     # FP16
        int8=False,    # önce kapalı tutun; INT8 istiyorsanız aşağıya bakın
        nms=True       # NMS'i göm
    )
    print("✅ Export bitti. yolov8n.mlpackage oluşturuldu.")

if __name__ == "__main__":
    main()
