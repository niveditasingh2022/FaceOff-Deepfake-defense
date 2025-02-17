"""
Main Functionality:
This script processes face-swapped images by calculating their cosine similarity 
with all face embeddings in the celebrity database. It identifies whether the 
faces in the swapped images match any known faces in the database.

Key Steps:
1. Creates a celebrity database from the LFW dataset by extracting face embeddings.
2. Processes face-swapped images to compute face embeddings.
3. Calculates cosine similarity between the swapped image embeddings and the celebrity database.
4. Annotates the swapped images with the best match and similarity score.
5. Saves the results, including processed images and similarity scores, in a CSV file.
"""

import os
import csv
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from insightface.app import FaceAnalysis

# Path and configuration variables
LFW_DIR = './LFW_dataset'  # Directory containing the LFW dataset
SWAPPED_IMG_DIR = './swapped_img'  # Directory containing face-swapped images
SWAPPED_RECOGNITION_DIR = './swapped_recognition'  # Output directory for processed images
RECOGNITION_CSV_PATH = './recognition_results.csv'  # Path for the results CSV file
EMBEDDINGS_PATH = 'all_embeddings.npy'  # Path to save/load celebrity face embeddings
LABELS_PATH = 'all_labels.npy'  # Path to save/load celebrity labels
THRESHOLD = 0.4  # Similarity threshold for face recognition

# Create output directory if it doesn't exist
os.makedirs(SWAPPED_RECOGNITION_DIR, exist_ok=True)

# Initialize InsightFace FaceAnalysis for face detection and recognition
app = FaceAnalysis(
    name="buffalo_l",
    root='./insightface_2',
    allowed_modules=['detection', 'recognition'],
    providers=['CPUExecutionProvider']
)
app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))

# 1. Create a celebrity database from the LFW dataset by extracting embeddings
def create_celebrity_database(lfw_dir):
    """
    Process the LFW dataset to create face embeddings and labels for the celebrity database.
    """
    embeddings = []
    labels = []
    for root, _, files in os.walk(lfw_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for image files
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                faces = app.get(img)
                if len(faces) == 0:
                    print(f"No face detected in {img_path}, skipping.")
                    continue
                # Extract the label from the file name (e.g., "person_name_1.jpg")
                label = '_'.join(os.path.splitext(file)[0].split('_')[:-1])
                embeddings.append(faces[0].normed_embedding)
                labels.append(label)
    return np.array(embeddings), np.array(labels)

# Load or generate face embeddings and labels
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(LABELS_PATH):
    all_embeddings = np.load(EMBEDDINGS_PATH)
    all_labels = np.load(LABELS_PATH)
else:
    all_embeddings, all_labels = create_celebrity_database(LFW_DIR)
    np.save(EMBEDDINGS_PATH, all_embeddings)
    np.save(LABELS_PATH, all_labels)
    print("Embeddings and labels saved.")

# Create a dictionary-based celebrity database
celebrity_database = defaultdict(list)
for label, embedding in zip(all_labels, all_embeddings):
    celebrity_database[label].append(embedding)

# 2. Function to compute the face embedding of an input image
def get_face_embedding(img):
    """
    Extract the face embedding from the given image.
    """
    faces = app.get(img)
    return faces[0].normed_embedding if len(faces) > 0 else None

# 3. Draw recognition results on the image
def draw_recognition_results(img, faces, result_label, similarity):
    """
    Annotate the image with bounding boxes, labels, and similarity scores.
    """
    img_copy = img.copy()
    for face in faces:
        bbox = face.bbox.astype(int)  # Get bounding box coordinates
        color = (0, 255, 0) if similarity > THRESHOLD else (0, 0, 255)  # Green if recognized, red otherwise
        cv2.rectangle(img_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label_text = f"{result_label}: {similarity:.2f}"
        cv2.putText(img_copy, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img_copy

# 4. Process face-swapped images
def process_swapped_images(swapped_img_dir, output_dir, csv_path):
    """
    Process face-swapped images by calculating cosine similarity with the celebrity database.
    Save annotated images and similarity results.
    """
    # Initialize the results CSV file
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Face Embedding', 'Best Match'])

    for root, _, files in os.walk(swapped_img_dir):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for image files
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                face_embedding = get_face_embedding(img)

                if face_embedding is None:
                    print(f"No face detected in {img_path}, skipping.")
                    continue

                max_similarity = -1
                best_match = "Unidentified"

                # Compare the embedding with the celebrity database
                for label, embeddings in celebrity_database.items():
                    for embedding in embeddings:
                        similarity = cosine_similarity([face_embedding], [embedding])[0][0]
                        if similarity > max_similarity:
                            max_similarity = similarity
                            if max_similarity > THRESHOLD:
                                best_match = label

                print(f"Processed: {file}, Best match: {best_match}, Similarity: {max_similarity:.2f}")
                faces = app.get(img)
                result_img = draw_recognition_results(img, faces, best_match, max_similarity)

                # Save the annotated image
                result_img_path = os.path.join(output_dir, file)
                Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)).save(result_img_path)

                # Append results to the CSV file
                with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([img_path, face_embedding.tolist(), best_match])

# Execute the processing of swapped images
process_swapped_images(SWAPPED_IMG_DIR, SWAPPED_RECOGNITION_DIR, RECOGNITION_CSV_PATH)
print("Swapped image processing completed.")
