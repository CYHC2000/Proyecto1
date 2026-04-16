import cv2
import numpy as np


def imgAdjustment(alpha: float, beta: float, imgResize: np.ndarray) -> np.ndarray:
    """
    Ajusta contraste y brillo y devuelve el canal verde.
    """
    imgAdj = cv2.convertScaleAbs(imgResize, alpha=alpha, beta=beta)
    _, g, _ = cv2.split(imgAdj)

    return g


def dropsDetection(cleanImg: np.ndarray,
                   minDist: int,
                   param1: int,
                   param2: int,
                   minRadius: int,
                   maxRadius: int,
                   imgResize: np.ndarray):
    """
    Detecta círculos y devuelve:
    - número total de objetos
    - diccionario con información por objeto
    """

    imgBlur = cv2.medianBlur(cleanImg, 3)

    circles = cv2.HoughCircles(
        imgBlur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    # Manejo robusto cuando no se detectan círculos
    if circles is None:
        return 0, {}

    circles = np.uint16(np.around(circles))

    objects_data = {}

    for idx, circle in enumerate(circles[0, :], start=1):
        x, y, r = circle

        # Crear máscara para calcular color promedio dentro del círculo
        mask = np.zeros(cleanImg.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        mean_color = cv2.mean(cleanImg, mask=mask)[0]

        objects_data[idx] = {
            "center": (int(x), int(y)),
            "mean_intensity": int(mean_color),
            "diameter": int(r * 2)
        }

    total_objects = len(objects_data)

    return total_objects, objects_data