import sys
import os
import numpy as np
import cv2
from scipy.ndimage import maximum_filter


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def find_templ(img, img_tpl):
    # Template shape
    h, w = img_tpl.shape

    # Match map building
    match_map = cv2.matchTemplate(img, img_tpl, cv2.TM_CCOEFF_NORMED)

    max_match_map = np.max(match_map)  # The value of the map for the region closest to the template
    print(max_match_map)
    if (max_match_map < 0.71):  # No matches found
        return []

    a = 0.7  # Coefficient of "similarity", 0 - all, 1 - exact match

    # Cut the map on the threshold
    match_map = (match_map >= max_match_map * a) * match_map

    # Select local max on the map
    match_map_max = maximum_filter(match_map, size=min(w, h))
    # Areas closest to the pattern
    match_map = np.where((match_map == match_map_max), match_map, 0)

    # Coordinates of local max
    ii = np.nonzero(match_map)
    rr = tuple(zip(*ii))

    res = [[c[1], c[0], w, h] for c in rr]

    return res


# Draw a frames of matches found
def draw_frames(img, coord):
    res = img.copy()
    for c in coord:
        top_left = (c[0], c[1])
        bottom_right = (c[0] + c[2], c[1] + c[3])
        cv2.rectangle(res, top_left, bottom_right, color=(0, 0, 255), thickness=5)
    return res


# Crop enter image into shelfs
def crop_image(img_path, n):
    crop_image_folder = "/Users/savchuk/Documents/template-matcher/data/shelf_image/"
    # image from device
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    new_img_h = h // n
    start_h = 0
    for x in range(1, n+1):
        crop_img = img[start_h:start_h+new_img_h, 0:w]
        tn = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite("/Users/savchuk/Documents/template-matcher/data/shelf_image/s_{}_{}.jpg".format(tn, x), crop_img)
        start_h += new_img_h

    return [os.path.join(crop_image_folder, b)
            for b in os.listdir(crop_image_folder)
            if os.path.isfile(os.path.join(crop_image_folder, b))]



def main():
    enter_image_path = "/Users/savchuk/Documents/template-matcher/data/image/0000.jpg"
    template_image_folder = "/Users/savchuk/Documents/template-matcher/data/template/"

    # template image
    templ = [os.path.join(template_image_folder, b) for b in os.listdir(template_image_folder) if os.path.isfile(os.path.join(template_image_folder, b))]



    # shelf count
    shelf_count = 2
    crop_image_list = crop_image(enter_image_path, shelf_count)

    img_tpl = cv2.imread(templ[0], cv2.IMREAD_GRAYSCALE)

    res_list = []
    for img in crop_image_list:
        shelf = int(img[len(img)-5:len(img)-4])
        img_gray = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        coord = find_templ(img_gray, img_tpl)

        # Match count on the shelf
        match_count = len(coord)
        img_res = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_res = draw_frames(img_res, coord)
        tn = os.path.splitext(os.path.basename(img))[0]
        cv2.imwrite("/Users/savchuk/Documents/template-matcher/data/result/res_{}_{}.jpg".format(tn, match_count), img_res)
        for c in coord:
            print(c)

        res_list.append(("res_{}_{}.jpg".format(tn, match_count), shelf, (len(coord))))
        print("- - - - - - - - - - - - - - -")

    print(res_list)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == "__main__":
    print("OpenCV ", cv2.__version__)
    sys.exit(main())