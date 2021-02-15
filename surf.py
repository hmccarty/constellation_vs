from os import walk, path, mkdir
import cv2

surf = cv2.xfeatures2d.SURF_create()

for (root, dirs, files) in walk('raw_images'):
    save_path = root.split("/")
    
    surf_path = save_path
    surf_path[0] = "surf_images"
    surf_path = "/".join(surf_path)
    if not path.isdir(surf_path):
        mkdir(surf_path)

    for f in files:
        img = cv2.imread("{}/{}".format(root, f), 0)

        # SURF
        surf_path = "{}/{}".format(surf_path, f)
        kp, des = surf.detectAndCompute(img, None)
        img = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)

        cv2.imwrite(surf_path, img)