import cv2
import numpy as np

house_priorities = {"blue": 2, "red": 1}

# Load the image
input_image = cv2.imread(r"C:\Users\USER\Downloads\New folder\image_6.jpg")  # Replace with your image path

# Define the color to be segregated (brown) and the replacement color (yellow)
target_color_1 = np.array([0, 25, 45])  # Brown color in BGR
replacement_color_1 = np.array([0, 255, 255])  # Yellow color in BGR

target_color_2 = np.array([2, 41, 9]) # Green from image
replacement_color_2 = np.array([255, 255, 255]) # White color in BGR

# Define a threshold for color distance
color_distance_threshold = 35

# Create a copy of the original image
output_image = input_image.copy()


# Iterate through each pixel and apply the replacement color if the color is similar to the target color
for y in range(output_image.shape[0]): # yellow
    for x in range(output_image.shape[1]):
        pixel_color = output_image[y, x]
        color_distance = np.linalg.norm(pixel_color - target_color_1)
        if color_distance < color_distance_threshold :
            output_image[y, x] = replacement_color_1


for y in range(output_image.shape[0]): #blue
    for x in range(output_image.shape[1]):
        pixel_color = output_image[y, x]
        color_distance = np.linalg.norm(pixel_color - target_color_2)
        if color_distance - 10 < color_distance_threshold + 40:
            output_image[y, x] = replacement_color_2



img = cv2.imread(r"C:\Users\irsha\Downloads\New folder\image_6.jpg")
blur = cv2.GaussianBlur(img,(5,5),0)
gray_image = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

red_triangles = 0
blue_triangles = 0

lower_red = np.array([0,0,150])
upper_red = np.array([80,80,255])
mask_red = cv2.inRange(img, lower_red, upper_red)

lower_blue = np.array([150,0,0])
upper_blue = np.array([255,80,80])
mask_blue = cv2.inRange(img, lower_blue, upper_blue)

mask = mask_red + mask_blue
contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #######

for contour in contours:

    approx = cv2.approxPolyDP(
        contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 0), 5)

    # putting shape name at center of each shape
    if len(approx) == 3:

        color_ranges = {
            'red': ([0, 0, 150], [80, 80, 255]),
            'blue': ([150, 0, 0], [255, 70, 40]),
        }
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect the color of the object based on the color ranges
        detected_color = None
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_img, np.array(lower), np.array(upper))
            if np.any(mask):
                detected_color = color
                break

        if detected_color == 'red':
            red_triangles += 1
        else:
            blue_triangles += 1



print("Number of red triangles: ", red_triangles)
print("Number of blue triangles: ", blue_triangles)

cv2.imshow("triangles", img)

# Display the modified image
cv2.imshow('Modified Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
