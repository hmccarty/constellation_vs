from os import walk, path, mkdir
import cv2

surf = cv2.xfeatures2d.SURF_create()
sift = cv2.xfeatures2d.SIFT_create()

for (root, dirs, files) in walk('raw_images'):
    save_path = root.split("/")

    # SIFT path setup
    sift_path = save_path
    sift_path[0] = "sift_images"
    sift_path = "/".join(sift_path)
    if not path.isdir(sift_path):
        mkdir(sift_path)
    
    # SURF path setup
    surf_path = save_path
    surf_path[0] = "surf_images"
    surf_path = "/".join(surf_path)
    if not path.isdir(surf_path):
        mkdir(surf_path)

    for f in files:
        img = cv2.imread("{}/{}".format(root, f), 0)

        # SIFT feature detection
        sift_file_path = "{}/{}".format(sift_path, f)
        kp, des = sift.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        cv2.imwrite(sift_file_path, img)

        # SURF feature detection
        surf_file_path = "{}/{}".format(surf_path, f)
        kp, des = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        cv2.imwrite(surf_file_path, img)