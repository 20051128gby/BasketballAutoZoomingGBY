import json
import cv2
import numpy as np
from utils.plots import plot_one_box
from detector import Get_Pred
from PIL import Image

import numpy as np
import myUtils

final_list=[]
pred_path = r'E:\vscodeP\data\Video\3月25日.mp4'
model_path = r'E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\weights\yolov5s.pt'


capture = cv2.VideoCapture(pred_path)
ret, img = capture.read()
frame_height, frame_width, channels = img.shape          
video_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 20
video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

def detect_video():
    global capture
    temp = 0
    while temp<1:
        ret, img = capture.read()
        temp+=1
        if ret is not True:
            break
        detect = Get_Pred(model_path)
        value_json = detect.process(img)
        value = json.loads(value_json)
        
        pred_xyxy = value['xyxy']
        cropList=myUtils.crop(img,pred_xyxy)

        #展示裁剪后的人类图片
        #display_cropped_images(cropList)


        for xyxy in pred_xyxy:
            #plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=1)
            print((xyxy))
        

        # cv2.imwrite("whole.jpg", img)  
        
        global final_list 
        final_list= myUtils.resize_with_white_background(cropList)

        
        myUtils.print_confidence_score(final_list, temp)
        #a = int(input("执行下一帧"))
        # cv2.imshow('Video', img)      
             

        #video.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break




if __name__ == '__main__':
   
    
    detect_video(pred_path)
    video.release()
    capture.release()
    cv2.destroyAllWindows()
