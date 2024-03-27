import json
import cv2
from utils.plots import plot_one_box
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from detector import Get_Pred
capture = cv2.VideoCapture(r'E:\vscodeP\data\Video\3月25日.mp4')
ret, img = capture.read()
frame_height, frame_width, channels = img.shape
            # 定义视频文件的名称和编码格式
video_filename = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
fps = 20
video = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))


def crop(image, boxes_list):
    # 初始化一个空列表来存储裁剪后的图像
    cropped_images = []
    # 遍历所有的边界框
    i=0
    for box in boxes_list:
        # 解包每个边界框的坐标
        if(calRatio(box)>0.8):
            continue
        x1, y1, x2, y2 = map((int),box)

        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]

        # 将裁剪后的图像添加到列表中
        cropped_images.append(cropped_image)
        cv2.imwrite(str(i)+".jpg",cropped_image)
        i+=1

    return cropped_images

def calRatio(x):
    x1,y1,x2,y2=x
    #print(x2-x1/y2-y1)
    return (x2-x1)/(y2-y1)


def display_cropped_images(cropped_images):
   
    # 定义每行显示的图像数量
    IMAGES_PER_ROW = 4

    num_rows = (len(cropped_images) + IMAGES_PER_ROW - 1) // IMAGES_PER_ROW
    fig, axes = plt.subplots(num_rows, IMAGES_PER_ROW, figsize=(IMAGES_PER_ROW * 5, num_rows * 5))
    for i, img in enumerate(cropped_images):
        # 计算当前图像的行和列
        row = i // IMAGES_PER_ROW
        col = i % IMAGES_PER_ROW

        axes[row, col].imshow(img)
        axes[row, col].axis('off')  

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 显示整个图像网格
    plt.show()

    # 创建翻页功能
    def next_page():
        clear_output(wait=True)
        display(fig)
    def prev_page():
        clear_output(wait=True)
        display(fig)
    plt.gcf().canvas.mpl_connect('key_press_event', lambda event: on_key(event, next_page, prev_page))

def on_key(event, next_func, prev_func):
    if event.key == 'right':
        next_func()  # 右箭头键翻到下一页
    elif event.key == 'left':
        prev_func()  # 左箭头键翻到上一页



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
        cropList=crop(img,pred_xyxy)
        #display_cropped_images(cropList)
        for xyxy in pred_xyxy:
            plot_one_box(xyxy, img, label=None, color=(255, 255, 0), line_thickness=1)
            print((xyxy))
        cv2.imwrite("whole.jpg", img)  
        a = int(input("执行下一帧"))
        cv2.imshow('Video', img)      
             
 
        #video.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            break




if __name__ == '__main__':
    model_path = r'E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\weights\yolov5s.pt'
    pred_path = r'E:\vscodeP\data\Video\3月25日.mp4'
    detect_video(pred_path)
    video.release()
    cv2.destroyAllWindows()
