from insightface.app import FaceAnalysis
import cv2
import insightface
import subprocess
import numpy as np

# FaceAnalysis 설정
app = FaceAnalysis(
    name="buffalo_l",
    root='./insightface',
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider']
)
app.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))

# Face Swapper 모델 로드
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=['CPUExecutionProvider'])

# 타겟 얼굴 로드
img2 = cv2.imread('dummy_face.png')
faces2 = app.get(img2)
if len(faces2) == 0:
    raise ValueError("No face detected in dummy_face.png")
face2 = faces2[0]

# 원본 비디오 로드
video_path = r"D:\CelebV-HQ\__lRwnjxeCg_2.mp4"
cap = cv2.VideoCapture(video_path)

# 비디오 속성 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# FFmpeg 파이프 실행 (출력 비디오 + 원본 오디오 결합)
final_output_video = "final_video.mp4"
ffmpeg_cmd = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    '-s', f'{frame_width}x{frame_height}', '-pix_fmt', 'bgr24',
    '-r', str(fps), '-i', '-',
    '-i', video_path, '-map', '0:v', '-map', '1:a',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
    '-c:a', 'aac', '-b:a', '192k',
    '-shortest', final_output_video
]
process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# 프레임 처리 & FFmpeg로 직접 전달
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ⚠ **프레임이 None이 아닌지 체크**
    if frame is None:
        print("[Warning] 빈 프레임이 감지됨. FFmpeg로 전달하지 않음.")
        continue

    # **메모리 정렬 최적화**
    frame = np.ascontiguousarray(frame.copy())

    # 얼굴 탐지
    faces1 = app.get(frame)
    if len(faces1) > 0:
        result = swapper.get(frame, faces1[0], face2, paste_back=True)
    else:
        result = frame  # 얼굴이 없으면 원본 유지

    # **FFmpeg에 프레임 전달**
    try:
        process.stdin.write(result.tobytes())  
    except BrokenPipeError:
        print("[Error] FFmpeg 프로세스가 예상보다 빨리 종료됨.")
        break
    except OSError as e:
        print(f"[Error] FFmpeg 입력 중 문제 발생: {e}")
        break

# 리소스 정리
cap.release()
process.stdin.close()
process.wait()

print(f"Final video saved as {final_output_video}")
