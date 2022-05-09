import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
img = cv2.imread('watermelon.jpg') 
# Display original image


# scaler=60
scale_percent = 10 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)

#resizing the image (shrinking the magnified the image)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# Converting those pixels with values 1-127 to 0 and others to 1
img = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]

#bitwise coversion
img = cv2.bitwise_not(img)

cv2.imshow('Original', img)
cv2.waitKey(0)

num_labels, rows = cv2.connectedComponents(img)
beansCoord = dict()
for i in range(len(rows)):
    for j in range(len(rows[i])):
        if rows[i][j] != 0:
            if rows[i][j] not in beansCoord:
                beansCoord[rows[i][j]] = list()
            beansCoord[rows[i][j]].append((i, j))

# print(beansCoord[13])

# Sorts based on width
beansCoord[13].sort(key=lambda value: value[0])
minXCoord = beansCoord[13][0]
maxXCoord = beansCoord[13][-1]

# Sorts based on height
beansCoord[13].sort(key=lambda value: value[1])
minYCoord = beansCoord[13][0]
maxYCoord = beansCoord[13][-1]

#now creating border
print("Min X: ", minXCoord[0], "\tMin Y: ", minYCoord[1])
print("Max X: ", maxXCoord[0], "\tMax Y: ", maxYCoord[1])

#getting mid
midX=(minXCoord[0]+maxXCoord[0])/2
midY=(minYCoord[1]+maxYCoord[1])/2
print(midX,midY)

# now will use Pythagoras Theoram


# borderX=50+midX
# borderY=50+midy


    # filelab = open("label"+str(i)+".txt", "w")
    # filelab.write(str(labels[i]))
    # filelab.close()
    # print("label",i ,"\t" , labels)
    # Map component labels to hue val, 0-179 is the hue range in OpenCV

#responsible for merging the image
label_hue = np.uint8(179*labels/np.max(labels))
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# Converting cvt to BGR
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# set bg label to black
labeled_img[label_hue==0] = 0


# Showing Original Image
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Orginal Image")
plt.show()

#Showing Image after Component Labeling
plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()

'''# # Convert to graycsale

img_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 

# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=400) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

cv2.destroyAllWindows()

'''
