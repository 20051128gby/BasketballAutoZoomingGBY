import dlib
import cv2
from matplotlib import pyplot as plt

# 初始化dlib的人脸检测器和特征点预测器
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(r"E:\vscodeP\AI\basketballAutoZooming\base_on_2021_Yolov5\shape_predictor_68_face_landmarks.dat")



def plot_face_points(image, shape, color=(0, 255, 0)):
    # 在图像上绘制特征点
    for n in range(68):
        point = shape.part(n)
        cv2.circle(image, (point.x, point.y), 1, color, -1)
    return image




import cv2
import numpy as np

# def edge_extraction(image_path):

#     # 读取图像
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    
    
#     # 创建Sobel算子，分别提取水平和垂直方向的边缘
#     sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 4
#     sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 4
    
#     # 对图像应用Sobel算子
#     sobel_x_image = cv2.filter2D(image, -1, sobel_x)
#     sobel_y_image = cv2.filter2D(image, -1, sobel_y)
    
#     # 计算最终的边缘强度
#     edges = np.sqrt(sobel_x_image**2 + sobel_y_image**2)
#     edges = np.uint8(edges / np.max(edges))
    
#     # 返回边缘提取后的图像
#     return edges,image

# # 显示边缘提取结果



def edge_extraction(image_path):
# 转换为灰度图像
    image = cv2.imread(image_path) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 应用Sobel算子
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度幅值
    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    
    # 阈值处理
    _, thresh = cv2.threshold(gradient_magnitude, 20, 255, cv2.THRESH_BINARY)
    return thresh


edge = edge_extraction(r'E:\vscodeP\AI\11_2520.jpg') 
image = cv2.imread(r'E:\vscodeP\AI\11_2520.jpg')


# 将图片从BGR转换为灰度图，用于人脸检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces = detector(gray, 1)

# 对于每一个检测到的人脸
for k, d in enumerate(faces):
    # 获取特征点
    shape = sp(image, d)

    # 绘制特征点
    image = plot_face_points(image, shape)

# 显示带有特征点的图像
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()