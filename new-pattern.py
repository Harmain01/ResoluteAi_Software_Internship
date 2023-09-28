import cv2
import numpy as np

# Load the image
image = cv2.imread('new_pattern data\Image_20230906114131613.bmp')

# Convert the image from BGR to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for yellow color in HSV
lower_yellow = np.array([0, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Threshold the image to get the yellow regions
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# Bitwise-AND the original image and the mask to get the yellow regions
yellow_highlighted = cv2.bitwise_and(image, image, mask=yellow_mask)

# Display the original image and the highlighted yellow regions
# cv2.imshow('Original Image', image)
cv2.imwrite('Highlighted Yellow.jpg', yellow_highlighted)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
