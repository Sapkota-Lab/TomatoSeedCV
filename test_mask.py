import cv2
from src.train_model import mask_bisected_seed_body, overlay_mask

img = cv2.imread("Images/IMG_1474.JPG")
assert img is not None

bodymask = mask_bisected_seed_body(img)
overlay = overlay_mask(img, bodymask)

cv2.imshow("Body Mask Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()