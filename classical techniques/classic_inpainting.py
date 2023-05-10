import cv2
import numpy as np

def harmonic_inpainting(input_image, mask, fidelity, tolerance, max_iterations, time_step):
    """
    Applies harmonic inpainting to the input image.

    :param input_image: a numpy array representing the input image
    :param mask: a numpy array representing the mask to be used in inpainting
    :param fidelity: a float representing the fidelity term
    :param tolerance: a float representing the tolerance value for stopping iterations
    :param max_iterations: an integer representing the maximum number of iterations to be performed
    :param time_step: a float representing the time step
    :return: inpainted image
    """

    # Determine the dimensions of the input image
    if input_image.ndim == 3:
        height, width, channels = input_image.shape
    else:
        height, width = input_image.shape
        channels = 1

    # Create a copy of the input image
    inpainted_image = input_image.copy()

    # Perform harmonic inpainting for each color channel
    for i in range(max_iterations):

        # Compute the Laplacian of the inpainted image
        laplacian = cv2.Laplacian(inpainted_image, cv2.CV_64F)

        # Compute the new solution
        new_solution = inpainted_image + time_step * (
                laplacian + fidelity * mask * (input_image - inpainted_image))

        # Compute the difference between the new and old solutions
        diff = np.linalg.norm(new_solution.reshape(height * width * channels, 1) -
                              inpainted_image.reshape(height * width * channels, 1), 2) / \
               np.linalg.norm(new_solution.reshape(height * width * channels, 1), 2)

        # Update the inpainted image
        inpainted_image = new_solution

        # Test the exit condition
        if diff < tolerance:
            break

    return np.uint8(inpainted_image)

class PDE_Inpainter():
    def __init__(self, k=20, alpha = 1, delta=0.25, n_iter=150):
        self.sigma=3
        self.rho=3
        self.k = k
        self.alpha = alpha
        self.delta = delta
        self.n_iter = n_iter

    def find_eigen_vectors(self, J):
        w = np.array([
            2*J[1, 0],
            J[1, 1]-J[0, 0]+np.sqrt((J[1, 1]-J[0, 0])**2+4*J[1, 0])
        ]).T
        w_perp = np.array([
            J[1, 1]-J[0, 0]+np.sqrt((J[1, 1]-J[0, 0])**2+4*J[1, 0]),
            -2*J[1, 0]
        ]).T
        return w/np.linalg.norm(w), w_perp/np.linalg.norm(w_perp)

    def find_corner_intensity(self, img):
        img_copy = np.zeros_like(img)
        img = np.float32(img)
        gray_8bit = cv2.convertScaleAbs(img)
        edges = cv2.Canny(gray_8bit, 50, 200)

        # Find contours of the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original image
        img_with_contours = cv2.drawContours(img_copy, contours, -1, 255, 1)
        return img_with_contours

    def g(self, x):
        temp = 1/(1+(x/self.k)**2)
        return temp

    def find_second_order_derivatives(self, img):
        # Compute first order derivatives using numpy gradient function
        Ix, Iy = np.gradient(img)
        
        # Compute second order derivatives using element-wise operations
        Ixx = np.zeros_like(img)
        Iyy = np.zeros_like(img)
        Ixy = np.zeros_like(img)
        Ixx[1:-1, :] = img[:-2, :] - 2*img[1:-1, :] + img[2:, :]
        Iyy[:, 1:-1] = img[:, :-2] - 2*img[:, 1:-1] + img[:, 2:]
        Ixy[:-1, :-1] = (img[1:, 1:] - img[:-1, 1:]) - (img[1:, :-1] - img[:-1, :-1])

        # Compute SOD
        a = (Ix**2)*Ixx+(Iy**2)*Iyy
        b = 2*Ix*Iy*Ixy
        c = (Ix**2)+(Iy**2)
        c[c==0]=1
        SOD_gradient_dir = (a+b)/c
        SOD_tangential_dir = (a-b)/c
        return SOD_gradient_dir, SOD_tangential_dir

        
        
    def inpaint(self, image, mask):
        current_image = image.copy()
        for i in range(self.n_iter):

            corner_intensity = self.find_corner_intensity(current_image)
            corner_protection_operator = self.g(corner_intensity)
            # Compute the x and y gradients using Sobel operator
            dx = cv2.Sobel(current_image, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(current_image, cv2.CV_64F, 0, 1, ksize=3)

            # Compute the gradient magnitude
            mag = np.sqrt(dx**2 + dy**2)
            edge_stopping_function = self.g(mag)

            SOD_gradient_dir, SOD_tangential_dir = self.find_second_order_derivatives(current_image)

            current_image = current_image + (self.delta*corner_protection_operator)*(edge_stopping_function*SOD_gradient_dir+self.alpha*SOD_tangential_dir)

        return mask*current_image+(1-mask)*image