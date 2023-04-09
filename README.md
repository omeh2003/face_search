# Часть 1: Краткое описание реализации, план и используемые технологии

Для решения задачи поиска человека на изображениях мы можем использовать готовые библиотеки компьютерного зрения и машинного обучения, такие как OpenCV и Dlib.

План реализации:

1.  Загрузка изображения-образца человека и изображений из каталога для поиска.
2.  Обнаружение лиц на всех изображениях.
3.  Извлечение векторных признаков лиц с использованием предобученной модели (например, модель Dlib Face Recognition).
4.  Сравнение векторных признаков образца с признаками лиц на изображениях из каталога.
5.  Выбор изображений, на которых был найден человек, согласно установленному порогу сходства.
6.  Вывод списка файлов с найденными изображениями.

Технологии:

*   Python 3.10
*   OpenCV (для работы с изображениями и обнаружения лиц)
*   Dlib (для извлечения векторных признаков лиц и сравнения их)

# Часть 2: Структура проекта

Структура проекта будет состоять из следующих файлов и каталогов:

scss

```scss
face_search/
│
├── images/ (каталог для хранения фотографий для поиска)
│
├── target/ (каталог для хранения фотографии-образца человека)
│
├── face_search.py (основной скрипт для поиска человека на изображениях)
│
├── requirements.txt (файл с зависимостями проекта)
│
└── README.md (документация по использованию программы)
```

Часть 3: Установка зависимостей и программ

Для работы с изображениями и обнаружения лиц, а также для извлечения векторных признаков лиц, нам понадобятся библиотеки OpenCV и Dlib.

Установка зависимостей:

bash

```bash
pip install -r requirements.txt
```

Содержимое файла requirements.txt:

makefile

```makefile
opencv-python~=4.7.0.72
opencv-python-headless
dlib~=19.24.1
cmake
numpy~=1.24.2
```

Дополнительные программы на операционную систему:
# Windows

На Windows, для библиотек OpenCV и Dlib, возможно, потребуются дополнительные зависимости, такие как CMake и Visual Studio Build Tools, чтобы скомпилировать библиотеки из исходного кода.

Если вы сталкиваетесь с проблемами при установке dlib, вы можете попробовать установить предварительно скомпилированную версию dlib с помощью команды:

bash

```bash
pip install dlib-binary
```

Если проблемы продолжаются, убедитесь, что у вас установлены все необходимые компоненты Microsoft Visual C++ Redistributable. Вы можете загрузить их с официального сайта Microsoft: [https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0](https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0)

# Linux 

Не должно быть проблем с установкой.

# Распознование лиц
Для распознования лиц нам понадобятся еще два файла  файл `shape_predictor_5_face_landmarks.dat`  и  `dlib_face_recognition_resnet_model_v1.dat`

скачайте файл `shape_predictor_5_face_landmarks.dat` с официального репозитория dlib:

1.  Откройте ссылку: [http://dlib.net/files/shape\_predictor\_5\_face\_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2)
2.  Скачайте файл `shape_predictor_5_face_landmarks.dat.bz2`.
3.  Извлеките файл `shape_predictor_5_face_landmarks.dat` из скачанного архива, используя программу для работы с архивами (например, 7-Zip).
4.  Поместите файл `shape_predictor_5_face_landmarks.dat` в ту же папку, что и `face_search.py`.

 скачайте файл `dlib_face_recognition_resnet_model_v1.dat` с официального репозитория dlib:

1.  Откройте ссылку: [http://dlib.net/files/dlib\_face\_recognition\_resnet\_model\_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
2.  Скачайте файл `dlib_face_recognition_resnet_model_v1.dat.bz2`.
3.  Извлеките файл `dlib_face_recognition_resnet_model_v1.dat` из скачанного архива, используя программу для работы с архивами (например, 7-Zip).
4.  Поместите файл `dlib_face_recognition_resnet_model_v1.dat` в ту же папку, что и `face_search.py`.


# Запуск программы

Теперь чтобы запустить программу, вы должны указать аргументы командной строки, например:

bash

```bash
python face_search.py target/target_image.jpg images
```

target/target_image.jpg - Фотография с лицом которое ищем.

images  - каталог с фотогрфиями на которых ищем лицо.
