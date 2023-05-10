import cv2
import numpy as np
from skimage.metrics import structural_similarity

def create_image_and_mask(image_path, mask_path, add_noise=True):
    """
    Creates a corrupted input image and a corresponding mask.

    :param image: a numpy array representing the clean input image
    :param mask: a numpy array representing the mask of the inpainting domain
    :return: a tuple containing the corrupted input image and the corresponding mask
    """
    # read image
    image = cv2.imread(image_path)

    # read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Create a copy of the input image
    input_image = image.copy()

    # Determine the dimensions and number of channels of the input image
    height, width = input_image.shape[:2]
    channels = input_image.shape[2] if len(input_image.shape) == 3 else 1

    # Determine the dimensions and number of channels of the mask
    mask = mask / 255.0
    mask_channels = mask.shape[2] if len(mask.shape) == 3 else 1

    # If the mask has only one channel, add a new third axis to the mask
    if mask_channels == 1:
        mask = np.expand_dims(mask, axis=2)

    # If the input image has a different number of channels than the mask, repeat the mask along the channel axis
    if channels > mask.shape[2]:
        mask = np.repeat(mask, channels, axis=2)

    # If the input image has only one channel, expand the dimensions of both the input image and the mask to 3 channels
    if channels == 1:
        input_image = np.expand_dims(input_image, axis=2)
        mask = np.expand_dims(mask, axis=2)

    # Create the corrupted input image by blending the intact part of the input image with random noise in the missing domain
    if(add_noise):
        noise = np.random.rand(height, width, channels)
    else:
        noise = np.zeros((height, width, channels))
    corrupted_image = mask * input_image + (1 - mask) * noise*255
    corrupted_image = np.uint8(corrupted_image)

    return corrupted_image, mask

def draw_mask(image, single_channel=True):
    '''
        Function to draw a custom mask for the source image.
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Create a window to display the image
    cv2.namedWindow('Draw mask')
    drawing = False
    ix, iy = -1, -1
    curve_pts = []

    # Define a function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        global drawing, ix, iy, curve_pts
        # If the left mouse button is pressed, start drawing a mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            curve_pts = [(ix, iy)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                    cv2.line(image, curve_pts[-1], (x, y), (0, 0, 255), 2)
                    curve_pts.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.line(image, curve_pts[-1], (x, y), (0, 0, 255), 2)
            curve_pts.append((x, y))
            pts = np.array(curve_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (0, 0, 0))

    # Initialize the mask as a black image with the same size as the input image
    mask = 255*np.ones(image.shape[:2], dtype=np.uint8)

    # Register the mouse callback function
    cv2.setMouseCallback('Draw mask', mouse_callback)

    # Display the image and wait for the user to draw the mask
    while True:
        cv2.imshow('Draw mask', image)
        key = cv2.waitKey(1)

        # If the user presses the 'c' key, clear the mask
        if key == ord('c'):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            image = image.copy()

        # If the user presses the 's' key, save the mask and exit the loop
        elif key == ord('s'):
            cv2.destroyWindow('Draw mask')
            if single_channel:
                return mask
            else:
                return cv2.merge([mask, mask, mask])

        # If the user presses the 'q' key or closes the window, exit the loop
        elif key == ord('q') or key == 27 or cv2.getWindowProperty('Draw mask', cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow('Draw mask')
            return None
        
# calculate MSE between two images
def mean_squared_error(imageA, imageB):
    return np.square(np.subtract(imageA, imageB)).mean()

# calculate PSNR between two images
def psnr(imageA, imageB):
    mse = mean_squared_error(imageA, imageB)
    return 10 * np.log10(np.max(imageA) ** 2 / mse)

# calculate SSIM between two images
def ssim(imageA, imageB):
    score, _ = structural_similarity(imageA, imageB, channel_axis=2, full=True)
    return score