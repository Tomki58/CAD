from typing import Union

import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


THRESHOLD = 10


def load_image(path: str) -> np.array:
    """ Загружает изображение и конвертирует его в GrayScale """
    img = Image.open(path)
    new_img = img.convert("L")
    ar = np.array(new_img)

    return ar


def dct_image(ar: Union[list, Image.Image]):
    """ Выполняет ДКП всего изображения """
    if isinstance(ar, Image.Image):
        ar = np.array(ar)
    return dct(dct(ar.T, norm="ortho").T, norm="ortho")


def idct_image(ar: Union[list, Image.Image]):
    """ Выполняет обратное ДКП всего изображения """
    if isinstance(ar, Image.Image):
        ar = np.array(ar)
    return idct(idct(ar.T, norm="ortho").T, norm="ortho")


def generate_watermark(size: int):
    """ Генерирует последовательность значений для встраивания """
    return np.random.normal(0.0, 1, size)

def zigzag_scan_n_extract(ar, watermark):
    """ Сканирует изображение в зигзагообразном порядке и модифицирует коэффициенты"""
    # rows, cols = ar.shape[0] - 1, 0
    alpha = 10
    start_row = ar.shape[0] - 1
    start_col = 0

    row_inc = True
    for elem in watermark:
        if row_inc:
            start_row -= 1
        else:
            start_col += 1
        row_inc = not row_inc
        ar[start_row][start_col] += alpha * abs(ar[start_row][start_col]) * elem

    # return ar

def detect_watermark(corrupted_image, watermark):
    """ Производит обнаружение голограммы """

    corrupted_image = dct_image(corrupted_image)

    start_row = corrupted_image.shape[0] - 1
    start_col = 0
    row_inc = True
    sum_ = float()


    for elem in watermark:
        if row_inc:
            start_row -= 1
        else:
            start_col += 1
        row_inc = not row_inc
        sum_ += corrupted_image[start_row][start_col] * elem

    return True if abs(sum_ / len(watermark)) > THRESHOLD else False


if __name__ == "__main__":

    img = load_image("img.jpg")
    dct_img = dct_image(img)
    watermark = generate_watermark(30)

    # Алгоритм Барни
    zigzag_scan_n_extract(dct_img, watermark)

    # Обратное преобразование
    img = idct_image(dct_img)

    print(detect_watermark(img, watermark))

    # Вывод изображения
    Image.fromarray(img).show()