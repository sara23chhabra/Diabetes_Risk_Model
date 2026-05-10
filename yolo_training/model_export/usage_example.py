from food_detector import detect_food

image_path = "../sample_images/sample1.jpg"

results = detect_food(image_path)

print(results)