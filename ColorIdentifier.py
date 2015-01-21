from ColorNames import ColorNames
import ColorConversion

# Modified from https://gist.github.com/jdiscar/9144764
# Uses cielab color space for difference calculation

class ColorIdentifier:
	# Returns the name of a specified color
	@staticmethod
	def identify(rgbValue=(0,0,0), colorMap=0):
		return ColorIdentifier.findNearestColorName(rgbValue, ColorNames.colorMaps[colorMap])


	@staticmethod
	def rgbFromStr(s):  
		# s starts with a #.
		r, g, b = int(s[1:3],16), int(s[3:5], 16),int(s[5:7], 16)  
		return r, g, b  


	@staticmethod
	def findNearestBasicColorName((R,G,B)):
		return ColorIdentifier.findNearestColorName((R,G,B), ColorNames.BasicColorMap)


	@staticmethod
	def findNearestWebColorName((R,G,B)):
		return ColorIdentifier.findNearestColorName((R,G,B), ColorNames.WebColorMap)


	@staticmethod
	def findNearestColorName((R,G,B),Map):  
		# Convert to cielab color space
		R, G, B = ColorConversion.rgb_to_cielab(R,G,B)

		mindiff = None
		for d in Map:
			r, g, b = ColorIdentifier.rgbFromStr(Map[d])
			# Convert to cielab color space
			r, g, b = r, g, b = ColorConversion.rgb_to_cielab(r,g,b)

			diff = abs(R -r)*256 + abs(G-g)* 256 + abs(B- b)* 256
			if mindiff is None or diff < mindiff:
				mindiff = diff
				mincolorname = d
		return mincolorname



#
#   Main Entry Point
#
if __name__ == '__main__':
	color = (100, 150, 50)
	print ColorIdentifier.findNearestBasicColorName(color)
	print ColorIdentifier.findNearestWebColorName(color)
	print ColorIdentifier.identify(color)
