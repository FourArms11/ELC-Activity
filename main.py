import cv2
from deepface import DeepFace
from ultralytics import YOLO

# Load YOLOv8 model (downloads automatically first time)
model = YOLO("yolov8s.pt")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Emotion-Aware Object Detection... Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Object Detection using YOLOv8
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    # Step 2: Emotion Detection using DeepFace
    try:
        analysis = DeepFace.analyze(
        frame,
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='opencv',
        align=True
)

        for face in analysis:
            emotion = face['dominant_emotion']
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w']
            h = face['region']['h']

            # Draw emotion box and label
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"Emotion: {emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    except Exception as e:
        pass

    # Show the output window
    cv2.imshow("Emotion-Aware Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()