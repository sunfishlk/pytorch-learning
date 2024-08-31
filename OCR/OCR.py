import numpy as np
import cv2                          # opencv    读取图像默认为BGR
import matplotlib.pyplot as plt     # matplotlib显示图像默认为RGB
"""######################################################################
# 计算齐次变换矩阵：cv2.getPerspectiveTransform(rect, dst)
# 输入参数		rect输入图像的四个点（四个角）
# 				dst输出图像的四个点（方方正正的图像对应的四个角）
######################################################################
# 仿射变换：cv2.warpPerspective(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
# 透视变换：cv2.warpAffine(src, M, dsize, dst=None, flags=None, borderMode=None, borderValue=None)
# 				src：输入图像     dst：输出图像
# 				M：2×3的变换矩阵
# 				dsize：变换后输出图像尺寸
# 				flag：插值方法
# 				borderMode：边界像素外扩方式
# 				borderValue：边界像素插值，默认用0填充
#
# （Affine Transformation）可实现旋转，平移，缩放，变换后的平行线依旧平行。
# （Perspective Transformation）即以不同视角的同一物体，在像素坐标系中的变换，可保持直线不变形，但是平行线可能不再平行。
#
# 备注：cv2.warpAffine需要与cv2.getPerspectiveTransform搭配使用。
######################################################################"""


def order_points(pts):
	rect = np.zeros((4, 2), dtype="float32")			# 一共4个坐标点
	# 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
	# 计算左上，右下
	s = pts.sum(axis=1)
	rect[0] = pts[np.argmin(s)]			# np.argmin()	求最小值对应的索引
	rect[2] = pts[np.argmax(s)]			# np.argmax()	求最大值对应的索引
	# 计算右上和左下
	diff = np.diff(pts, axis=1)			# np.diff 	求（同一行）列与列之间的差值
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect


def four_point_transform(image, pts):
	rect = order_points(pts)		# 获取输入坐标点
	(tl, tr, br, bl) = rect			# 获取四边形的四个点，每个点有两个值，对应（x, y）坐标
	# 计算输入的w和h值
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))			# 取四边形上下两边中，最大的宽度

	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))			# 取四边形左右两边中，最大的高度

	# 变换后对应坐标位置
	dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
	"""###############################################################################
	# 计算齐次变换矩阵：cv2.getPerspectiveTransform(rect, dst)
	###############################################################################"""
	M = cv2.getPerspectiveTransform(rect, dst)
	
	"""###############################################################################
	# 透视变换（将输入矩形乘以（齐次变换矩阵），得到输出矩阵）
	###############################################################################"""
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
	dim = None
	(h, w) = image.shape[:2]
	if width is None and height is None:
		return image
	if width is None:
		r = height / float(h)
		dim = (int(w * r), height)
	else:
		r = width / float(w)
		dim = (width, int(h * r))
	resized = cv2.resize(image, dim, interpolation=inter)
	return resized


##############################################
image = cv2.imread(r'./OCR/images/OCR.png')
ratio = image.shape[0] / 500.0						# resize之后坐标也会相同变化，故记录图像的比率
orig = image.copy()
image = resize(orig, height=500)
##############################################
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)		# 转换为灰度图
gray = cv2.GaussianBlur(gray, (5, 5), 0)			# 高斯滤波操作
edged = cv2.Canny(gray, 75, 200)					# Canny算法（边缘检测）
##############################################
print("STEP 1: 边缘检测")
cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
##############################################
# 轮廓检测
cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)		# 轮廓检测
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]			# 选定所有轮廓中前五个轮廓，并进行排序
for c in cnts:
	peri = cv2.arcLength(c, True)									# 计算轮廓近似
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)					# 找出轮廓的多边形拟合曲线
	if len(approx) == 4:				# 如果当前轮廓是四个点（矩形），表示当前轮廓是所需求目标
		screenCnt = approx
		break
##############################################
print("STEP 2: 获取轮廓")
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)			# 在原图上画出检测得到的轮廓
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
##############################################
# 透视变换
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)		# 得到的轮廓要乘以图像的缩放尺寸
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)							# 转换为灰度图
ref = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY)[1]					# 二值化处理
ref = resize(ref, height=500)
##############################################
print("STEP 3: 齐次变换")
cv2.imshow("Scanned", ref)
cv2.waitKey(0)
cv2.destroyAllWindows()

##############################################
# 轮廓点绘制的颜色通道是BGR;  但是Matplotlib是RGB;  故在绘图时，(0, 0, 255)会由BGR转换为RGB（红 - 蓝）
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)			# BGR转换为RGB格式
edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)

plt.subplot(2, 2, 1),    plt.imshow(orig),      plt.title('orig')
plt.subplot(2, 2, 2),    plt.imshow(edged),     plt.title('edged')
plt.subplot(2, 2, 3),    plt.imshow(image),     plt.title('contour')
plt.subplot(2, 2, 4),    plt.imshow(ref),       plt.title('rectangle')
plt.show()

"""######################################################################
# 计算轮廓的长度：retval = cv2.arcLength(curve, closed)
# 输入参数：      curve              轮廓（曲线）。
#                closed             若为true,表示轮廓是封闭的；若为false，则表示打开的。（布尔类型）
# 输出参数：      retval             轮廓的长度（周长）。
######################################################################
# 找出轮廓的多边形拟合曲线：approxCurve = approxPolyDP(contourMat, 10, true)
# 输入参数：     contourMat：              轮廓点矩阵（集合）
#               epsilon：                 (double类型)指定的精度, 即原始曲线与近似曲线之间的最大距离。
#               closed：                  (bool类型)若为true, 则说明近似曲线是闭合的; 反之, 若为false, 则断开。
# 输出参数：     approxCurve：             轮廓点矩阵（集合）；当前点集是能最小包容指定点集的。画出来即是一个多边形；
######################################################################"""

