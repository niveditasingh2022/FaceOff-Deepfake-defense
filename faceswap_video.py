from insightface.app import FaceAnalysis
import cv2
import os
import insightface
from PIL import Image
import subprocess

# Initialize the FaceAnalysis module for face detection and recognition
app = FaceAnalysis(
    name="buffalo_l",  # Use the buffalo_l model (for detection and recognition)
    root='./insightface',  # Path to store InsightFace model files
    allowed_modules=['detection', 'recognition'],  # Enable detection features only
    providers=['CUDAExecutionProvider']  # Use GPU if available
)
app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))  # Set detection thresholds and image size

# Load the face swapping model
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])

# Load the target image (dummy face) for face swapping
img2_path = 'dummy_face.png'
img2 = cv2.imread(img2_path)

# Perform face detection on the target image (dummy face)
faces2 = app.get(img2)
if len(faces2) == 0:
    raise ValueError("No face detected in dummy_face.png")  # Exit if no face is detected in the dummy face image

# Select the first detected face as the reference face for swapping
face2 = faces2[0]

# Load the video to process
# video_path = r"D:\CelebV-HQ\__lRwnjxeCg_2.mp4"
video_path = "__lRwnjxeCg_2.mp4"
cap = cv2.VideoCapture(video_path)

# Extract audio from the original video
audio_path = "original_audio.aac"
subprocess.run(f'ffmpeg -i "{video_path}" -vn -acodec copy "{audio_path}"', shell=True)


# Get the video properties (frame width, height, FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare the output video writer
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' or others
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame of the video
while True:
    ret, frame = cap.read()
    # print(f"ret : {ret}")
    # print(type(frame))
    # print(frame.shape)
    if not ret:
        break  # End of video

    # Perform face detection on the current frame
    
    faces1 = app.get(frame)
    if len(faces1) == 0:
        print("No face detected in this frame, skipping.")
        out.write(frame)  # Write the frame as is if no face detected
        continue
    else:
        print(f"Detected {len(faces1)} faces in this frame.")

    # # Debugging: print the detected face's properties
    # print(f"Detected face properties: {faces1[0].__dict__}")
    
    
    # Perform face swapping with the first detected face in the current frame
    result = swapper.get(frame, faces1[0], face2, paste_back=True)

    # Convert result to RGB for compatibility with PIL and save it to the output video
    temp = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    rimg = Image.fromarray(temp)

    # Write the swapped frame to the output video
    out.write(result)

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Swapped video saved to {output_video_path}")


# Add the original audio back to the swapped video
final_output_video = "final_video.mp4"
subprocess.run(
    f'ffmpeg -i "{output_video_path}" -i "{audio_path}" -c:v copy -c:a aac -strict experimental "{final_output_video}"',
    shell=True
)

print(f"Final video with original audio saved as {final_output_video}")