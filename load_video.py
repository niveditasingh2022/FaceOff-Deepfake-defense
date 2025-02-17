import cv2
import os

# 비디오 경로
video_path = r"D:\CelebV-HQ\__lRwnjxeCg_2.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오가 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 저장할 디렉토리 생성 (이미지 파일을 저장할 경로)
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

frame_count = 0

# 비디오의 프레임을 하나씩 읽어들여 저장
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break  # 비디오 끝까지 읽었으면 종료

    # 각 프레임을 'frame_번호.png' 형태로 저장
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, frame)

    print(f"Saved {frame_filename}")
    frame_count += 1

# 리소스 해제
cap.release()

print(f"All frames have been saved to {output_dir}")
