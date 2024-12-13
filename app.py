from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os

app = Flask(__name__)

# setup the paths for file upload and annotated images
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'uploads/annotated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# cats vs dogs model
cats_vs_dogs_model = YOLO('pets/runs/detect/train5/weights/best.pt')

# stanford dog breeds model
dog_breeds_model = YOLO('pets/runs/detect/train11/weights/best.pt')

# route to serve the home page
@app.route('/')
def index():
    return render_template('index.html')

# route to handle the file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error="No file part")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file")

    try:
        # save the uploaded file
        uploaded_file_path = Path(UPLOAD_FOLDER) / file.filename
        file.save(uploaded_file_path)

        # load the image using PIL
        img = Image.open(uploaded_file_path)

        # store, annotated models to a list
        annotated_images = []

        models = [cats_vs_dogs_model, dog_breeds_model]

        for model in models:
            results = model(img)  # perform prediction using YOLOv8 model
            
            result = results[0]  # get the first result in the list
            annotated_image_np = result.plot()  # this returns a NumPy array

            # reverse the color channels for PIL
            annotated_image_np_rgb = annotated_image_np[..., ::-1]  # convert BGR to RGB

            # convert back to numpy array to PIL image
            annotated_image_pil = Image.fromarray(annotated_image_np_rgb, 'RGB')

            # store the annotated image
            annotated_images.append(annotated_image_pil)

        # lazy, but I want to just send 1 image back to the user, so combining both predictions to one image
        extra_width = 20  # extra width between images
        total_width = sum(image.width for image in annotated_images) + extra_width
        max_height = max(image.height for image in annotated_images)

        # white background
        combined_annotated_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))  

        current_x = 0
        for image in annotated_images:
            combined_annotated_image.paste(image, (current_x, 0))
            current_x += image.width + extra_width 

        # resize the final combined image to fit within a max width while maintaining aspect ratio
        max_width = 1000
        width_percent = (max_width / float(combined_annotated_image.size[0]))
        new_height = int((float(combined_annotated_image.size[1]) * float(width_percent)))
        resized_combined_image_pil = combined_annotated_image.resize((max_width, new_height))

        # save the resized annotated image
        annotated_image_path = Path(ANNOTATED_FOLDER) / file.filename
        resized_combined_image_pil.save(annotated_image_path)

        # return the path of the annotated image
        annotated_image_url = f"/uploads/annotated/{file.filename}"

        return render_template('result.html', annotated_image=annotated_image_url)

    except Exception as e:
        return render_template('index.html', error=f"Error during prediction: {str(e)}")

# route back to serve the annotated images
@app.route('/uploads/annotated/<filename>')
def send_annotated_image(filename):
    return send_from_directory(ANNOTATED_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)