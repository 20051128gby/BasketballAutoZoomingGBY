import cv2

# 视频文件路径
video_path = r'E:\vscodeP\data\Video\3月25日.mp4'

# 尝试打开视频文件
capture = cv2.VideoCapture(video_path)

# 检查视频是否打开成功
if not capture.isOpened():
    print(f"Error: Could not open video file {video_path}")
else:
    # 尝试读取视频的第一帧
    ret, img = capture.read()
    if ret:
        # 如果成功读取帧，打印图像尺寸
        frame_height, frame_width, channels = img.shape
        print(f"Frame shape: Height={frame_height}, Width={frame_width}, Channels={channels}")
    else:
        print("Error: Failed to read the first frame from the video.")