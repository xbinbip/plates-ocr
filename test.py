from PIL import Image
import matplotlib.pyplot as plt
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import os
import gradio as gr
import numpy as np
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from PIL import ImageDraw

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')


def plot_img(img,title = 'Input Image',subplot = 111):
    plt.figure()
    plt.subplot(111)
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def get_preprocessed_image(image_tensor):
    image_array = image_tensor.squeeze().numpy()
    mean = np.array(OPENAI_CLIP_MEAN)[:, None, None]
    std = np.array(OPENAI_CLIP_STD)[:, None, None]
    image_array = (image_array * std) + mean
    image_array = (image_array * 255).astype(np.uint8)
    image_array = np.moveaxis(image_array, 0, -1)
    return Image.fromarray(image_array)

def process_image(image):
    texts = [["a licence plate"]]

    inputs = processor(text=texts, images=image, return_tensors="pt")
    for k,v in inputs.items():
        print(k,v.shape)

    with torch.no_grad():
        outputs = model(**inputs)

    unnormalized_image = get_preprocessed_image(inputs.pixel_values)

    target_sizes = torch.Tensor([unnormalized_image.size[::-1]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.4)
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    zipped = zip(boxes, scores, labels)
    for box, score, label in zipped:
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")


    visualized_image = unnormalized_image.copy()

    draw = ImageDraw.Draw(visualized_image)

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.4:
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = tuple(box)
            draw.rectangle(xy=((x1, y1), (x2, y2)), outline="red")
            draw.text(xy=(x1, y1), text=text[label], font_size=20, fill="blue")
        else:
            continue

    return draw._image



if __name__ == "__main__":
    demo = gr.Interface(
        process_image,
        inputs=[gr.Image(type="pil", label="Input Image")],
        outputs=[gr.Image(type="pil", label="Output Images")],
        title="Object Detection Demo",
        description="Upload image to detect objects",
        examples=[["image1.jpg"]]
    )


    demo.launch()