from image_converter import get_image_array, convert_grayscale
from sobel_operator import compute_gradient_magnitude, compute_horizontal_gradient_magnitude, \
    compute_vertical_gradient_magnitude, compute_gradient_angle


def detect():
    image = get_image_array('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive\\crop_000010b.bmp')
    gx_gradient = compute_horizontal_gradient_magnitude(image)
    print('gx_gradient: ' + str(gx_gradient[28][28]))
    gy_gradient = compute_vertical_gradient_magnitude(image)
    print('gy_gradient: ' + str(gy_gradient[28][28]))
    gradient = compute_gradient_magnitude(gx_gradient, gy_gradient)
    print('gradient: ' + str(gradient[28][28]))
    theta = compute_gradient_angle(gradient, gx_gradient, gy_gradient)
    print('angle: ' + str(theta[28][28]))


if __name__ == '__main__':
    detect()
