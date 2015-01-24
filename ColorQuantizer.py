import numpy as np
import cv2
import operator

from ColorIdentifier import ColorIdentifier
import ColorConversion


class ColorQuantizer:
	original_color = None
	original_small_color = None

	quantized_img = None
	quantized_img_hsv = None

	colorFreqMap = None
	percentageList = None

	K = None
	compactness = 0
	mean = 0

	def quantize(self, image, K=3):
		self.K = K
		self.original_color = image
		self.original_small_color = self.resize(image, 100)
		
		self.quantized_img = self.quantizeColors(self.original_small_color, K)
		self.quantized_img_hsv = cv2.cvtColor(self.quantized_img, cv2.COLOR_BGR2HSV)

		self.colorFreqMap = self.mapColorFrequencies(self.quantized_img)
		self.percentageList = self.getPercentageList()

	def optimalK(self, im, sampleK=8):
		quantizer = {}

		for k in range(1, sampleK+1):
			quantizer[k] = ColorQuantizer()
			quantizer[k].quantize(im, k)


		b0 = np.array((2, quantizer[2].compactness))
		b1 = np.array((sampleK, quantizer[sampleK].compactness))
		b = b1 - b0
		b = b/np.linalg.norm(b)

		dist = {}
		maxDist = 0
		maxDistK = 3

		for k in range(3, sampleK):
			pt = np.array((k, quantizer[k].compactness))
			p = pt - b0
			dist[k] = np.linalg.norm(p - b*(np.dot(p, b)))
			# print "K =", k, "d =", dist[k]

			if k == 3:
				maxDist = dist[k]
				maxDistK = 3
			elif dist[k] > maxDist:
				maxDist = dist[k]
				maxDistK = k

		# print "Max dist =", maxDist
		print "Optimal K =", maxDistK

		quantizer[maxDistK].showFrequencyBar()
		quantizer[maxDistK].printColorNames()

	def resize(self, image, longestEdgeLength):
		# resize image such that the longest edge is longestEdgeLength px
		ratio = 100.0/float(max(image.shape[0], image.shape[1]))
		return cv2.resize(image, (0,0), fx=ratio, fy=ratio)


	def quantizeColors(self, img, K=3):
		# perform k-means clustering
		Z = img.reshape((-1,3))
		Z = np.float32(Z)
		# define criteria, number of clusters(K) and apply kmeans()
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		ret,label,center = cv2.kmeans(Z, K, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
		self.compactness = ret

		# Now convert back into uint8, and make original image
		center = np.uint8(center)
		res = center[label.flatten()]
		res2 = res.reshape((img.shape))
		return res2


	def mapColorFrequencies(self, img):
		colorFreqMap = dict({})
		for y in range(0,img.shape[0]):
			for x in range(0,img.shape[1]):
				pixel = tuple(img[y][x])
				freq = colorFreqMap.get(pixel)
				if freq == None:
					freq = 0
				freq += 1
				colorFreqMap[pixel] = freq

		return colorFreqMap


	def getPercentageList(self):
		# sort dict
		sortedList = sorted(self.colorFreqMap.items(), key=operator.itemgetter(1))
		sortedList.reverse()
		# convert to percentage
		totalPixels = self.original_small_color.shape[0] * self.original_small_color.shape[1]
		for i in range(0, len(sortedList)):
			sortedList[i] = (sortedList[i][0], float(sortedList[i][1])/float(totalPixels))

		return sortedList


	def showFrequencyBar(self, percentageList=None):
		if percentageList == None:
			percentageList = list(self.percentageList)

		# culumulate percentages
		currPercentage = 0.0
		for i in range(0, len(percentageList)):
			currPercentage += percentageList[i][1]
			percentageList[i] = (percentageList[i][0], currPercentage)

		# show graphical representation
		barHeight = 50
		barWidth = 400
		barImg = np.zeros((barHeight, barWidth, 3), dtype=np.uint8)
		colorIndex = 0

		for x in range(0, barWidth):
			percentage = float(x)/float(barWidth)
			for i in range(0, len(percentageList)):
				if percentageList[i][1] > percentage:
					break
				else:
					colorIndex = i+1
			color = list(percentageList[colorIndex][0])
			
			for y in range(0, barHeight):
				barImg[y][x] = color

		cv2.imshow("Color Frequency Bar", barImg)


	def printColorNames(self):
		for i in range(0, len(self.percentageList)):
			(B, G, R) = self.percentageList[i][0]
			rgbValue = (R, G, B)
			frequency = self.percentageList[i][1]
			colorName0 = ColorIdentifier.identify(rgbValue, 0)
			colorName1 = ColorIdentifier.identify(rgbValue, 1)
			colorName2 = ColorIdentifier.identify(rgbValue, 2)
			# Remove digits before displaying
			#colorName2 = ''.join(i for i in colorName2 if not i.isdigit())
			print "[", frequency*100, "% ]", colorName0, ",", colorName1, ",", colorName2, " - ", rgbValue

#
#   Main Entry Point
#
if __name__ == '__main__':
	im = cv2.imread("flower.jpg")
	cv2.imshow("Original Image", im)

	quantizer = ColorQuantizer()
	quantizer.optimalK(im, 8)


	cv2.waitKey()