import cv2
import numpy


def get_image_array(path):
    image = cv2.imread(path)
    image_array = numpy.array(image)
    # print(image_array)
    grayscale_image_array = convert_grayscale(image_array)
    create_image(grayscale_image_array)
    return grayscale_image_array


def create_image(array):
    cv2.imwrite('C:\\Users\\Brandon\\PycharmProjects\\hog\\resources\\outputs\\test.bmp', array)


def convert_grayscale(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            r, g, b = array[i][j]
            array[i][j] = round(0.299*r + 0.587*g + 0.114*b)
    return array
