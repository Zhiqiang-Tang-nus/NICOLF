from scipy import ndimage
import cv2

def image_processing(color_image):
    img=color_image[40:310, 360:540]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres,img_bw=cv2.threshold(img,100,255,cv2.THRESH_BINARY)

    down_sample_ratio=0.3
    down_sample_img_bw=cv2.resize(img_bw, None, fx = down_sample_ratio, fy = down_sample_ratio)
    binary_image=down_sample_img_bw/255.0
    binary_image[binary_image > 0] = 1.0

    struct = ndimage.generate_binary_structure(2, 1)
    filled_image=ndimage.binary_dilation(binary_image, structure=struct,iterations=1).astype(binary_image.dtype)
    filled_image=ndimage.binary_fill_holes(filled_image)
    filled_image=filled_image.astype(int)

    return filled_image
