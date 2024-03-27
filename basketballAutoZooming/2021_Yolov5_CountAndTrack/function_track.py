import json
import cv2
from utils.plots import plot_one_box
from detector import Get_Pred
capture = cv2.VideoCapture(r'E:\vscodeP\data\Video\3月25日.mp4')
ret, img = capture.read()
frame_height, frame_width, channels = img.shape
            # 定义视频文件的名称和编码格式
video_filename = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20
video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
def detect_video(path):
    capture = cv2.VideoCapture(path)
    temp = 0
    while True:
        ret, img = capture.read()
        temp+=1
        if ret is not True:
            break
        detect = Get_Pred(model_path)
        value_json = detect.process(img)
        value = json.loads(value_json)
        pred_xyxy = value['xyxy']
        for xyxy in pred_xyxy:
            plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=1)
        cv2.imshow('Video', img)       
        #video.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break


if __name__ == '__main__':
    model_path = r'E:\vscodeP\basketballzooming\basketballAutoZooming\2021_Yolov5_CountAndTrack\weights\yolov5s.pt'
    pred_path = r'E:\vscodeP\data\Video\3月25日.mp4'
    detect_video(pred_path)
    video.release()
    cv2.destroyAllWindows()
q