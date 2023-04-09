import cv2
import dlib
import os
import numpy as np
from typing import List, Tuple
import argparse


def load_image(path: str) -> np.ndarray:
    """
    Загружает изображение из файла.

    :param path: Путь к файлу изображения.
    :return: Массив данных изображения.
    """
    return cv2.imread(path)


def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Обнаруживает лица на изображении.

    :param image: Массив данных изображения.
    :return: Список с координатами обнаруженных лиц (x, y, width, height).
    """
    face_detector = dlib.get_frontal_face_detector()
    detected_faces = face_detector(image, 1)
    return [(face.left(), face.top(), face.width(), face.height()) for face in detected_faces]


def extract_face_features(image, face_rectangle):
    # Создаем объекты детектора лиц и точек лица dlib
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    # Преобразуем кортеж в dlib.rectangle
    dlib_rectangle = dlib.rectangle(face_rectangle[0], face_rectangle[1], face_rectangle[2], face_rectangle[3])

    # Извлекаем лицевые ключевые точки
    shape = predictor(image, dlib_rectangle)

    # Конвертируем BGR (OpenCV) в RGB (dlib)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Извлекаем признаки лица
    face_features = face_recognition_model.compute_face_descriptor(rgb_image, shape)

    return face_features




import numpy as np

def compare_faces(face_features1, face_features2):
    face_features1_np = np.array(face_features1)
    face_features2_np = np.array(face_features2)

    distance = np.linalg.norm(face_features1_np - face_features2_np)

    return distance > 0.6



def search_person(target_image_path: str, images_directory: str) -> List[str]:
    """
    Поиск человека на изображениях в указанном каталоге.

    :param target_image_path: Путь к изображению-образцу человека.
    :param images_directory: Путь к каталогу с изображениями для поиска.
    :return: Список файлов, в которых найден человек.
    """
    target_image = load_image(target_image_path)
    target_faces = detect_faces(target_image)

    if not target_faces:
        raise ValueError("No faces detected in target image")

    target_face_features = extract_face_features(target_image, target_faces[0])

    found_files = []

    for file_name in os.listdir(images_directory):
        file_path = os.path.join(images_directory, file_name)

        if not os.path.isfile(file_path) or not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = load_image(file_path)
        faces = detect_faces(image)

        for face in faces:
            face_features = extract_face_features(image, face)
            if compare_faces(target_face_features, face_features):
                found_files.append(file_name)
                break

    return found_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Поиск человека на изображениях")
    parser.add_argument("target_image_path", help="Путь к изображению-образцу человека")
    parser.add_argument("images_directory", help="Путь к каталогу с изображениями для поиска")
    args = parser.parse_args()

    target_image_path = args.target_image_path
    images_directory = args.images_directory

    found_files = search_person(target_image_path, images_directory)

    print("Найденные файлы:", found_files)

    with open("founds.txt", "w") as founds_file:
        for file_name in found_files:
            founds_file.write(f"{file_name}\n")



