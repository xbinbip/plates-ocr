import yolov5
import cv2
import pytesseract
import gradio as gr
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import easyocr

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def plot_img(img,title = 'Input Image',subplot = 111):
    plt.figure()
    plt.subplot(subplot)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()

# load model
model = yolov5.load('keremberke/yolov5m-license-plate')

# set model parameters
model.conf = 0.35  # NMS confidence threshold
model.iou = 0.55  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 10  # maximum number of detections per image

def image_preprocess(image):

    if image is None:
        raise ValueError("Image cannot be null")

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Apply thresholding to segment text from background
    thresh_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply morphological operations to remove noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # Return preprocessed image
    return dilated_image

def preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise using morphological operations
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(opening, (5, 5), 0)

    # Apply thresholding again to get a binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh

def detect_plate_demo(image):
    results = model(image, size=800)
    results = model(image, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    threshold = 0.5
    best_guess = 0
    best_box = 0
    for i, (box, score, category) in enumerate(zip(boxes, scores, categories)):
        if score > best_guess:
            best_guess = score
            best_box = i
        if score > threshold:
            print(f"Prediction {i+1}:")
            print(f"  Box: {box}")
            print(f"  Score: {score:.2f}")
            print(f"  Category: {category}")
        

# show detection bounding boxes on image
    results.show()
    print(f'Best guess: {best_guess}, best box: {best_box}')
    print(results.xyxy[0][best_box])

    return results.ims[0]

def detect_plate(image):
    results = model(image, size=800)
    results = model(image, augment=True)

    # parse results
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]

    threshold = 0.5
    best_guess = 0
    best_box = 0
    for i, (box, score, category) in enumerate(zip(boxes, scores, categories)):
        if score > best_guess:
            best_guess = score
            best_box = i
        if score > threshold:
            print(f"Prediction {i+1}:")
            print(f"  Box: {box}")
            print(f"  Score: {score:.2f}")
            print(f"  Category: {category}")
        

    return results.xyxy[0][best_box].numpy()

def test(image,psm = 7):
    assert image is not None, "Image is null"
    assert isinstance(image, np.ndarray), "Image is not a numpy array"

    try:
        bounding_box = detect_plate(image)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if bounding_box is None:
        print("No bounding box found")
        return None

    plot_img(image, 'Input Image', 111)

    x1, y1, x2, y2, _, _ = bounding_box
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

    cropped = image[y1:y2, x1:x2]

    plot_img(cropped, 'Cropped Image', 222)

    result = preprocess(cropped)

    #text = pytesseract.image_to_string(result, lang='eng', config='--psm ' + str(psm))
    # text = reader.readtext(result)
    # print(text)

    # Extract text using easyocr
    text_results = reader.readtext(result)
    print(text_results)

    # Concatenate text results into a single string
    text = ' '.join([result[1] for result in text_results])

    # Write the text on top of the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cropped = cv2.putText(cropped, text, (55, 120), font, 0.5, (0, 123, 255), 2)

    return [result, cropped, text]

def test2(image, psm):
    cropped = test(image, psm)
    return cropped



# Create a Gradio interface
# demo = gr.Interface(
#     fn=test,
#     inputs=gr.Image(type="numpy",sources=["upload"]),
#     outputs=gr.Image(type="numpy",),
#     title="License Plate Detector",
#     description="Upload an image to detect the license plate"
# )

demo = gr.Interface(
    fn=test,
    inputs=[gr.Image(type="numpy",sources=["upload"]),gr.Slider(0,13,step=1,label="Tesseract PSMs")],
    outputs=[gr.Image(type="numpy",),gr.Image(type="numpy",),"text"],
    title="License Plate Detector",
    description="Upload an image to detect the license plate"
)

# Launch the demo
demo.launch()