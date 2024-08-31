import cv2
import numpy as np
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# img_path = './face-recognize/images/1.jpg'
# img_path = './face-recognize/images/2.png'
# img_path = './face-recognize/images/demo.jpg'
# img_path = './face-recognize/images/3.png'
# img_path = './face-recognize/images/4.jpg'
img_path = './face-recognize/images/5.jpg'
img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"无法读取图像文件，请检查路径：{img_path}")

faces = app.get(img)

rimg = app.draw_on(img, faces)

output_path = './face-recognize/images/output.jpg'
cv2.imwrite(output_path, rimg)

# 使用 matplotlib 显示图像
plt.imshow(cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

print(f"已保存结果图像到：{output_path}")