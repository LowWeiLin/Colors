import numpy as np
import cv2
import operator

img = cv2.imread('flower.jpg')

cv2.imshow("img", img)

# resize image such that the longest edge is about 100px
ratio = 100.0/float(max(img.shape[0], img.shape[1]))
img = cv2.resize(img, (0,0), fx=ratio, fy=ratio)


def quantizeColors(img, K=3):
	# perform k-means clustering
	Z = img.reshape((-1,3))
	Z = np.float32(Z)
	# define criteria, number of clusters(K) and apply kmeans()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret,label,center = cv2.kmeans(Z, K, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))
	return res2


def mapColorFrequencies(img):
	colorFreqMap = dict({})
	for y in range(0,img.shape[0]):
		for x in range(0,img.shape[1]):
			pixel = tuple(img[y][x])
			freq = colorFreqMap.get(pixel)
			if freq == None:
				freq = 0
			freq += 1
			colorFreqMap[pixel] = freq

	#print colorFreqMap
	#print "Map size: ", len(colorFreqMap)
	return colorFreqMap

quantized_img = quantizeColors(img)
#cv2.imshow("quantized img", quantized_img)

# get hsv
quantized_img_hsv = cv2.cvtColor(quantized_img, cv2.COLOR_BGR2HSV)

# get frequencies
colorFreqMap = mapColorFrequencies(quantized_img)

# sort dict
sortedList = sorted(colorFreqMap.items(), key=operator.itemgetter(1))
sortedList.reverse()

# convert to percentage
totalPixels = img.shape[0] * img.shape[1]
for i in range(0, len(sortedList)):
	sortedList[i] = (sortedList[i][0], float(sortedList[i][1])/float(totalPixels))

#print sortedList

# culumulate percentages
currPercentage = 0.0
for i in range(0, len(sortedList)):
	currPercentage += sortedList[i][1]
	sortedList[i] = (sortedList[i][0], currPercentage)

print sortedList

# show graphical representation
barHeight = 50
barWidth = 400
barImg = np.zeros((barHeight, barWidth, 3), dtype=np.uint8)
colorIndex = 0

for x in range(0, barWidth):
	percentage = float(x)/float(barWidth)
	for i in range(0, len(sortedList)):
		if sortedList[i][1] > percentage:
			break
		else:
			colorIndex = i+1
	color = list(sortedList[colorIndex][0])
	
	for y in range(0, barHeight):
		barImg[y][x] = color

cv2.imshow("bar", barImg)

cv2.waitKey(0)
cv2.destroyAllWindows()