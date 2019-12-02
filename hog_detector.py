from image_converter import get_image_array, convert_grayscale
from sobel_operator import compute_gradient_magnitude


def detect():
    image = get_image_array('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\training_images_positive\\crop_000010b.bmp')
    gradient = compute_gradient_magnitude(image)
    print(gradient)


if __name__ == '__main__':
    detect()
