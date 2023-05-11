import cv2
import numpy as np
import matplotlib.pyplot as plt
from classic_inpainting import PDE_Inpainter, harmonic_inpainting
from utils import create_image_and_mask, mean_squared_error, psnr, ssim
import pandas as pd


if __name__ == "__main__":
    # image_path = './images/turtle.png'
    # mask_path = './images/turtle_mask.png'
    image_path = './images/portrait.png'
    mask_path = './images/portrait_mask.png'
    image = cv2.imread(image_path)
    _, mask = create_image_and_mask(
        image_path, mask_path, add_noise=False)

    # display the original image and corrupted imahge
    cv2.imshow('Original Image', image)
    cv2.imshow('Mask', mask)

    # parameters
    fidelity = 10
    tol = 1e-5
    maxiter = 400
    dt = 0.1

    # perform harmonic inpainting
    inpainted_image = harmonic_inpainting(
        image, mask, fidelity, tol, maxiter, dt)

    # display the inpainted image
    cv2.imshow('Harmonic Inpainted Image', inpainted_image)

    # calculate mse for the image and corrupted image
    mse1 = mean_squared_error(image, inpainted_image)
    psnr1 = psnr(image, inpainted_image)
    ssim1 = ssim(image, inpainted_image)

    mask = 1-mask
    pde = PDE_Inpainter(alpha=1.5, n_iter=100, delta=0.2)
    
    # perform inpainting for each channel
    inpainted_image_pde = np.zeros_like(image)
    for i in range(3):
        inpainted_image_pde[:, :, i] = pde.inpaint(image[:, :, i], mask[:, :, i])

    # display the inpainted image
    cv2.imshow('PDE Inpainted Image', inpainted_image_pde)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # calculate mse for the image and corrupted image
    mse2 = mean_squared_error(image, inpainted_image_pde)
    psnr2 = psnr(image, inpainted_image_pde)
    ssim2 = ssim(image, inpainted_image_pde)

    # make a pandas dataframe
    df = pd.DataFrame({'MSE': [mse1, mse2], 'PSNR': [psnr1, psnr2], 'SSIM': [ssim1, ssim2]},
                        index=['Harmonic', 'PDE Inpainted'])
    print(df)
