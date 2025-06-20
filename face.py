from ultralytics import YOLO
import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1240)  # Width
cap.set(4, 680)  # Height

# Load YOLO model
model = YOLO("../Yolo-Wight/yolo11n.pt")  # Correct the path if needed

while True:
    success, img = cap.read()
    if not success:
        break

    # Run model on the image (stream=False is fine here)
    results = model(img)

    # Process detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Optional: show confidence and class
            if box.conf is not None and box.cls is not None:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show image
    cv2.imshow("YOLO Webcam", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
