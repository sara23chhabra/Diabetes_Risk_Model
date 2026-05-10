from ultralytics import YOLO

# =========================
# LOAD MODEL
# =========================
model = YOLO("weights/best.pt")

# =========================
# MAIN PREDICTION FUNCTION
# =========================
def detect_food(image_path, confidence_threshold=0.15):

    results = model(image_path)

    detected_foods = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Skip weak detections
            if confidence < confidence_threshold:
                continue

            food_name = model.names[class_id]

            detected_foods.append({
                "food": food_name,
                "confidence": round(confidence, 2)
            })

    return detected_foods