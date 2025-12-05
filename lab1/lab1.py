import numpy as np
import math as mt
import random
from graphics import *

'''
Лабораторна робота №1. Завдання ІІ рівня. Варіант 26.
Виконала: Мельник Анна, студентка групи ІМ-32.

Завдання:
1. Геометрична фігура: Піраміда з трикутною основою.
2. Динаміка: графічна фігура з’являється та гасне (масштабування), 
   змінює колір контуру та заливки.
3. Використовувати матричні операції.
'''

# -------------------------- Налаштування вікна --------------------------
XW = 800
YW = 600
WIN_TITLE = "Lab 1: V26 - Triangular Pyramid Animation"


# -------------------------- Матричні функції ----------------------------

def scale_matrix(s):
    """Матриця масштабування"""
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, s, 0],
        [0, 0, 0, 1]
    ])


def translation_matrix(tx, ty, tz):
    """Матриця переміщення (зсуву)"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [tx, ty, tz, 1]
    ])


def rotation_x_matrix(degrees):
    """Матриця обертання навколо осі X"""
    rad = mt.radians(degrees)
    c = mt.cos(rad)
    s = mt.sin(rad)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1]
    ])


def rotation_y_matrix(degrees):
    """Матриця обертання навколо осі Y"""
    rad = mt.radians(degrees)
    c = mt.cos(rad)
    s = mt.sin(rad)
    return np.array([
        [c, 0, -s, 0],
        [0, 1, 0, 0],
        [s, 0, c, 0],
        [0, 0, 0, 1]
    ])


def project_xy_matrix():
    """Матриця ортогональної проекції на площину XY (відкидаємо Z)"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1]
    ])


# -------------------------- Основна логіка ------------------------------

def create_pyramid():
    """
    Створення координат вершин піраміди з трикутною основою.
    Центр фігури (приблизно) у точці (0,0,0) для коректного обертання.
    """
    # Вершини: (x, y, z, 1)
    # 0: Вершина піраміди (Top)
    # 1, 2, 3: Вершини основи (Base)
    h = 150  # Висота
    w = 100  # Ширина основи

    vertices = np.array([
        [0, -h / 2, 0, 1],  # Top (Вершина)
        [-w, h / 2, -w / 1.5, 1],  # Base 1 (Лівий задній)
        [w, h / 2, -w / 1.5, 1],  # Base 2 (Правий задній)
        [0, h / 2, w, 1]  # Base 3 (Передній)
    ])
    return vertices


def get_random_color():
    """Генерує випадковий колір у форматі hex або назви"""
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'pink']
    return random.choice(colors)


def main():
    win = GraphWin(WIN_TITLE, XW, YW, autoflush=False)
    win.setBackground('white')

    pyramid = create_pyramid()

    current_scale = 0.1
    scale_step = 0.02
    growing = True

    angle_x = 20  # Початковий нахил для кращого огляду (аксонометрія)
    angle_y = 0
    rotation_speed = 2

    # Параметри кольорів
    fill_color = get_random_color()
    outline_color = 'black'

    drawn_objects = []

    print("Анімація запущена. Натисніть на вікно або закрийте його для виходу.")

    while win.isOpen():
        for obj in drawn_objects:
            obj.undraw()
        drawn_objects.clear()

        if growing:
            current_scale += scale_step
            if current_scale >= 1.5:
                growing = False
        else:
            current_scale -= scale_step
            if current_scale <= 0.05:
                growing = True
                fill_color = get_random_color()
                outline_color = get_random_color()

        angle_y += rotation_speed

        M_scale = scale_matrix(current_scale)

        M_rot_x = rotation_x_matrix(angle_x)
        M_rot_y = rotation_y_matrix(angle_y)
        M_rot = M_rot_x.dot(M_rot_y)  # Комбіноване обертання

        M_trans = translation_matrix(XW / 2, YW / 2, 0)

        M_proj = project_xy_matrix()

        M_total = M_scale.dot(M_rot).dot(M_trans).dot(M_proj)

        transformed_vertices = pyramid.dot(M_total)

        faces_indices = [
            [0, 1, 2],  # Side 1
            [0, 2, 3],  # Side 2
            [0, 3, 1],  # Side 3
            [1, 2, 3]  # Base (Bottom)
        ]

        for indices in faces_indices:
            pts = []
            for idx in indices:
                x = transformed_vertices[idx, 0]
                y = transformed_vertices[idx, 1]
                pts.append(Point(x, y))

            poly = Polygon(pts)
            poly.setFill(fill_color)
            poly.setOutline(outline_color)
            poly.setWidth(2)
            poly.draw(win)
            drawn_objects.append(poly)

        win.update()

        time.sleep(0.03)

        if win.checkMouse():
            break

    win.close()


if __name__ == '__main__':
    main()
