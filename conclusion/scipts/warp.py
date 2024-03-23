import cv2
import numpy as np
from random_edge_homography import HOMOGRAPHY_RETURN, RandomEdgeHomography

if __name__ == "__main__":
    img = np.load("data/frame_0.npy")
    print(img.shape)

    seed = 42 * 42 * 69
    np.random.seed(seed)

    crp_shape = (240, 320)
    rho = 64
    reh = RandomEdgeHomography(
        rho=rho, crp_shape=crp_shape, p0=0, homography_return=HOMOGRAPHY_RETURN.VISUAL
    )

    dic = reh(img)

    ones = np.ones_like(dic["img"], dtype=np.uint8) * 255
    H = dic["H"]
    wrp_ones = cv2.warpPerspective(
        ones, np.linalg.inv(H), (ones.shape[1], ones.shape[0])
    )
    wrp_ones = ones - wrp_ones

    cv2.imshow("img", dic["img"])
    cv2.imshow("wrp", dic["wrp"])
    cv2.imshow("wrp_ones", wrp_ones)
    cv2.waitKey()

    # safe all images, follow naming at https://github.com/advimman/lama?tab=readme-ov-file
    cv2.imwrite("data/warped/raw/img.png", dic["img"])
    cv2.imwrite("data/warped/raw/img_crp.png", dic["img_crp"])
    cv2.imwrite("data/warped/raw/wrp_crp.png", dic["wrp_crp"])
    # for inpainting
    cv2.imwrite("data/warped/raw/image1.png", dic["wrp"])
    cv2.imwrite("data/warped/raw/image1_mask001.png", wrp_ones)

    dic = reh.visualize(dic)
    cv2.imwrite("data/warped/viz/img.png", dic["img"])
    cv2.imwrite("data/warped/viz/wrp.png", dic["wrp"])
    cv2.imwrite("data/warped/viz/img_crp.png", dic["img_crp"])
    cv2.imwrite("data/warped/viz/wrp_crp.png", dic["wrp_crp"])
