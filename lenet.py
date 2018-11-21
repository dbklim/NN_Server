# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для построения и обучения сети LeNet, и классификации изображений рукописных цифр.
'''

import sys
import numpy
import time
from PIL import Image as pilImage
from io import BytesIO
import tensorflow
from keras.models import model_from_json
from keras.preprocessing import image
from keras.datasets import mnist
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import utils
        
# Для повторяемости результатов (опорное значение для генератора случайных чисел)
numpy.random.seed(42)

class LeNet:
    ''' Предназначен для построения и обучения сети LeNet, и классификации изображений рукописных цифр.
    1. f_net_model - имя .json файла с описанием модели сети
    2. f_net_weights - имя .h5 файла с весами обученной сети
    3. f_test_classification_img - имя .jpg/.png/.bmp/.tiff файла с тестовым изображением цифры '''
    def __init__(self, f_net_model, f_net_weights, f_test_classification_img):
        self.f_net_model = f_net_model
        self.f_net_weights = f_net_weights
        self.f_test_classification_img = f_test_classification_img
        self.lenet_model = None
        self.init_classification()        
        self.is_training = False
        

    def train(self, f_source_data, for_out_in_file = False):
        ''' Обучение сети на наборе рукописных цифр MNIST с параметрами по умолчанию. 
        1. f_source_data - имя .npz файла с обучающей выборкой
        2. for_out_in_file - изменяет вывод при обучении модели для удобства записи в файл 
        3. возвращает строку с точностью классификации на тестовых данных из MNIST и затраченное на обучение время '''       
        
        self.is_training = True

        # Размер изображений в обучающей выборке
        img_width, img_height = 28, 28

        # Загрузка обучающей выборки mnist (X_train и X_test - наборы изображений, y_train и y_test - 
        # цифры, которые на них изображены (т.е. правильные ответы, метки классов))
        print('[i] Загрузка обучающей выборки из %s' % f_source_data)
        f = numpy.load(f_source_data)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()
        
        print('[i] Подготовка обучающей выборки...')
        # Добавление всех изображений в один массив
        # n x m x p x k, где n - номер изображения, m - столбец в изображении, p - строка в изображении, k - интенсивность цвета пикселя (0..255) 
        x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
        x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
        input_shape = (img_width, img_height, 1)

        # Приведение типа к float32 и нормализация (перевод интенсивности цвета пикселя в диапазон 0..1)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        # Преобразование меток изображений в категории (т.е. преобразование цифры в массив из 10 элементов (т.к. 10 классов), где все элементы 
        # равны 0 кроме того, порядковый номер которого равен исходной цифре, он равен 1). 
        # Например: была цифра 4, в результате преобразования получится массив - [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        Y_train = utils.to_categorical(y_train, 10)
        Y_test = utils.to_categorical(y_test, 10)

        print('[i] Построение сети...')
        # Создание модели однонаправленной (последовательной) сети
        model = Sequential()

        start_time = None
        end_time = None
        scores = None
        
        # Слой свертки, 75 карт признаков, размер ядра свертки: 5х5
        # Представляет из себя разбиение каждого изображения на матрицы 5х5, интенсивность цвета каждого пикселя умножается
        # на веса соответствующего данной части изображения нейрона, результаты суммируются и передаются дальше 
        # 75 карт признаков - т.е. 75 наборов слоёв, которые используют разные ядра свёртки (т.е. по сути, выделяют 75 признаков)
        model.add(Conv2D(75, kernel_size=(5, 5), activation='relu', input_shape=input_shape))

        # Слой подвыборки (субдискретизирующий слой), размер пула 2х2
        # Нужен что бы распознавать цифры разного масштаба на изображении, на этом слое берутся результаты матрицы нейронов 2х2 
        # предыдущего слоя и выбирается из них максимальное
        # Dropout - слой регуляризации, нужен что бы предотвратить переобучение (на этом слое каждый раз, когда подаётся новый 
        # объект (т.е. происходит передача сигналов между предыдущим слоем и следующим), случайным образом отключаются нейроны 
        # с вероятностью 0.2, т.е. вероятность 20% того, что нейрон не будет участвовать в обучении на следующем слое, он будет «отключён»)
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Слой свертки, 100 карт признаков, размер ядра свертки 5х5
        model.add(Conv2D(100, kernel_size=(5, 5), activation='relu'))

        # Слой подвыборки, размер пула 2х2
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        # Полносвязный слой, 500 нейронов, необходим для классификации
        # Flatten - преобразование из двумерного вида в одномерный
        # В слое типа Dense происходит соединение всех нейронов предыдущего уровня со всеми нейронами следующего уровня
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.5))

        # Полносвязный выходной слой, 10 нейронов, которые соответствуют классам рукописных цифр от 0 до 9
        # При успешном распознавании на выходе одного из этих нейронов будет значение, близкое к 1, а на остальных - близкие к 0
        model.add(Dense(10, activation='softmax'))

        # Компиляция модели и вывод данных о ней
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        # Обучение сети
        print('\n[i] Обучение сети...')
        start_time = time.time()
        if for_out_in_file:
            model.fit(x_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=2)
        else:
            model.fit(x_train, Y_train, batch_size=200, epochs=10, validation_split=0.2, verbose=1)

        # Оценка качества обучения на тестовых данных, loss - значение функции ошибки, acc - точность
        scores = model.evaluate(x_test, Y_test, verbose=0)
        end_time = time.time()
        print('[i] Точность работы на тестовой выборке: %.2f%%' % (scores[1]*100))
        print('[i] Время обучения: %.2f минут.' % ((end_time - start_time)/60.0))

        # Генерация описания модели в формате json и запись её в файл
        print('[i] Сохранение модели сети и весов в %s и %s' % (self.f_net_model, self.f_net_weights))
        model_json = model.to_json()
        json_file = open(self.f_net_model, 'w')
        json_file.write(model_json)
        json_file.close()

        # Сохранение весов в файл (для работы необходим пакет h5py (sudo pip3 install h5py) и libhdf5-dev (sudo apt-get install libhdf5-dev))
        model.save_weights(self.f_net_weights)

        self.is_training = False
        self.init_classification()
        return (round(scores[1]*100, 2), round((end_time - start_time)/60, 2)) # точность в %, время в минутах


    def init_classification(self):
        ''' Загрузка модели сети и её весов, компиляция полученной сети и проверка её работоспособности. '''
        # Загрузка модели из файла
        json_file = open(self.f_net_model, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.lenet_model = model_from_json(loaded_model_json)

        # Загрузка весов в модель
        self.lenet_model.load_weights(self.f_net_weights)

        # Компиляция модели
        self.lenet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Загрузка тестового изображения
        test_image_value = 0
        image_for_classification = pilImage.open(self.f_test_classification_img)
        image_for_classification = image_for_classification.convert('L')
        image_for_classification.thumbnail((28, 28), pilImage.ANTIALIAS)

        # Преобразование изображения в массив numpy
        x = image.img_to_array(image_for_classification)
        
        # Инвертирование цвета и нормализация
        x = 255 - x
        x = x / 255
        x = numpy.expand_dims(x, axis=0)

        # Классификация
        categories = self.lenet_model.predict(x)
        classified_digit = int(numpy.argmax(categories))
        if classified_digit == test_image_value:
            print('[i] Тестовое изображение классифицированно правильно.')
        else:
            print('[E] Тестовое изображение классифицировано неверно. Рекомендуется обучить сеть заново.')        
        

    def classify(self, image_for_classification):
        ''' Классификация изображения с помощью заранее обученной сети.
        1. image_for_classification - .jpg/.png/.bmp/.tiff файл с изображением цифры в виде бинарной строки
        2. возвращает строку с распознанной цифрой или: error_1 - если выполняется обучение сети, error_2 - если что-то не так с изоражением '''
        
        if self.is_training == True:
            return 'error_1'
        
        try:
            # Преобразование бинарной строки в PIL Image
            stream = BytesIO(image_for_classification)
            image_for_classification = pilImage.open(stream)

            # Если изображение прямоугольное, то преобразование его в квадратное
            width, height = image_for_classification.size
            if width > height:
                background = pilImage.new('RGB', (width, width), (255, 255, 255))
                background.paste(image_for_classification, (0, (width - height) // 2))
                image_for_classification = background
            elif width < height:
                background = pilImage.new('RGB', (height, height), (255, 255, 255))
                background.paste(image_for_classification, ((height - width) // 2, 0))
                image_for_classification = background

            # Преобразование изображение в чёрно-белое и уменьшение размера до 28х28
            image_for_classification = image_for_classification.convert('L')
            image_for_classification.thumbnail((28, 28), pilImage.ANTIALIAS)
        except Exception:
            return 'error_2'

        # Преобразование изображения в массив numpy
        x = image.img_to_array(image_for_classification)
        
        # Инвертирование цвета и нормализация
        x = 255 - x
        x = x / 255
        x = numpy.expand_dims(x, axis=0)

        # Распознавание
        categories = self.lenet_model.predict(x)
        classified_digit = int(numpy.argmax(categories))
        return str(classified_digit)




def main():
    f_source_data = 'data/mnist.npz'
    f_net_model = 'data/lenet_model.json'
    f_net_weights = 'data/lenet_weights.h5'
    f_test_classification_img = 'images/test_classification.jpg'
    lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img)
    lenet.train(f_source_data)


if __name__ == '__main__':
    main()