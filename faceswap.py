from insightface.app import FaceAnalysis
import cv2
import os
import insightface
from PIL import Image

providers = ['CUDAExecutionProvider']

# Initialize the FaceAnalysis module for face detection and recognition
app = FaceAnalysis(
    name="buffalo_l",  # Use the buffalo_l model (for detection and recognition)
    root='./insightface',  # Path to store InsightFace model files
    allowed_modules=['detection'],  # Enable detection features only
    providers=providers  # Use the CPU for inference, we can also use GPU
)
app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))  # Set detection thresholds and image size

# Load the face swapping model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=['CUDAExecutionProvider'])

# Load the target image (dummy face) for face swapping
img2_path = 'dummy_face.png'
img2 = cv2.imread(img2_path)

# Perform face detection on the target image
faces2 = app.get(img2)
if len(faces2) == 0:
    raise ValueError("No face detected in dummy_face.png")  # Exit if no face is detected in the target image

# Select the first detected face as the reference face for swapping
face2 = faces2[0]

# Load LFW dataset
lfw_dir = './LFW_dataset'  
output_dir = './swapped_img'  
os.makedirs(output_dir, exist_ok=True)

# Iterate through all images in the LFW dataset and perform face swapping
for root, dirs, files in os.walk(lfw_dir):
    for file in files:
        # Process only image files with supported extensions
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img1_path = os.path.join(root, file)
            img1 = cv2.imread(img1_path)
            
            # Perform face detection on the source image
            faces1 = app.get(img1)
            if len(faces1) == 0:
                print(f"No face detected in {img1_path}, skipping.")  # Skip images with no detected faces
                continue

            # Perform face swapping
            result = swapper.get(img1, faces1[0], face2, paste_back=True)

            # Save the swapped image
            temp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for PIL compatibility
            rimg = Image.fromarray(temp)
            output_path = os.path.join(output_dir, file)
            rimg.save(output_path)  # Save the swapped image

            print(f"Saved swapped image to {output_path}")
