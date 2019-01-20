# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Предназначен для построения и обучения сети LeNet, и классификации изображений рукописных цифр.
'''

import sys
import os
import numpy
import time
import curses
from PIL import Image as pilImage
from io import BytesIO
import tensorflow
from keras.models import model_from_json
from keras.preprocessing import image as image_preproc
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
        if os.path.exists(f_net_model) and os.path.exists(f_net_weights) and os.path.exists(f_test_classification_img):
            self.__init_classification()
        self.is_training = False
        

    def train(self, f_source_data, for_out_in_file=False):
        ''' Обучение сети на наборе рукописных цифр f_source_data с параметрами по умолчанию. 
        1. f_source_data - имя .npz файла с обучающей выборкой
        2. for_out_in_file - изменяет вывод при обучении модели для удобства записи в файл
        4. возвращает строку с точностью классификации на тестовых данных и затраченное на обучение время '''       
        
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
        # Помещает значение интенсивности цвета каждого пикселя каждого изображения в отдельные массивы 
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
        self.__init_classification()
        return (round(scores[1]*100, 2), round((end_time - start_time)/60, 2)) # точность в %, время в минутах


    def __init_classification(self):
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
        x = image_preproc.img_to_array(image_for_classification)
        
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
        2. возвращает строку с распознанной цифрой или: error_1 - если выполняется обучение сети, eror_2 - если не найдены файлы с обученной сетью, error_3 - если что-то не так с изоражением '''
        
        if self.is_training == True:
            return 'error_1'
            
        if self.lenet_model == None:
            return 'error_2'
        
        try:
            # Преобразование бинарной строки в PIL Image
            stream = BytesIO(image_for_classification)
            image = pilImage.open(stream)

            # Конвертирование изображения в чёрно-белое и сохранение значений цвета каждого пикселя в массив numpy
            image = image.convert('L')
            image = numpy.array(image)

            # Удаление белых прямоугольников сверху и снизу изображения
            i = 0
            while i < image.shape[0]:
                if all(image[i] > 230):
                    image = numpy.delete(image, i, axis=0)
                else:
                    i += 1

            # Удаление белых прямоугольников слева и справа изображения
            i = 0
            while i < image.shape[1]:
                if all(image[:,i] > 230):
                    image = numpy.delete(image, i, axis=1)
                else:
                    i += 1
            image = pilImage.fromarray(image)

            # Если изображение прямоугольное, то преобразование его в квадратное
            width, height = image.size
            if width > height:
                background = pilImage.new('L', (width, width), (255))
                background.paste(image, (0, (width - height) // 2))
                image = background
            elif width < height:
                background = pilImage.new('L', (height, height), (255))
                background.paste(image, ((height - width) // 2, 0))
                image = background

            # Уменьшение разрешения до 28х28 пикселей и добавление белой рамки шириной 3 пикселя
            image.thumbnail((28, 28), pilImage.ANTIALIAS)
            background = pilImage.new('L', (28+6, 28+6), (255))
            background.paste(image, (3, 3))
            image = background
            image.thumbnail((28, 28), pilImage.ANTIALIAS)
        except Exception:
            return 'error_3'

        # Преобразование изображения в массив numpy по правилам keras
        x = image_preproc.img_to_array(image)
        
        # Инвертирование цвета и нормализация
        x = 255 - x
        x = x / 255
        x = numpy.expand_dims(x, axis=0)

        # Распознавание
        categories = self.lenet_model.predict(x)
        classified_digit = int(numpy.argmax(categories))
        return str(classified_digit)

    
    def create_training_sample(self, folder_source_images, f_training_sample='data/training_sample.npz'):
        ''' Создание обучающей выборки из набора изображений .jpg/.png/.bmp/.tiff. Изображения в своих именах должны содержать метку класса.
        Например: 0_1399.jpg - на изображении цифра 0; 7_81491.jpg - на изображении цифра 7
        1. folder_source_images - имя папки с изображениями
        2. f_training_sample - имя .npz файла, в который будет сохранена полученная обучающая выборка '''
        curses.setupterm()

        # Получение имён всех изображений, находящихся в folder_source_images    
        print('[i] Поиск изображений для создания обучающей выборки в %s' % folder_source_images)
        f_names_source_images = sorted(os.listdir(path=folder_source_images))
        if len(f_names_source_images) == 0:
            print('[E] Изображения не найдены!')
            return
        print('[i] Найдено %i изображений(-е)' % len(f_names_source_images))
        folder_source_images += '/'

        print('[i] Обработка изображений...')
        source_images = []
        classes = []
        k = 0
        for f_name_source_image in f_names_source_images:
            if k % 100 == 0 or k == len(f_names_source_images) - 1:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))
                print('[i] Обработка изображений... %i из %i' % (k, len(f_names_source_images)))
            image = pilImage.open(folder_source_images + f_name_source_image)

            # Конвертирование изображения в чёрно-белое и сохранение значений цвета каждого пикселя в массив numpy
            image = image.convert('L')
            image = numpy.array(image)

            # Удаление белых прямоугольников сверху и снизу изображения
            i = 0
            while i < image.shape[0]:
                if all(image[i] > 230):
                    image = numpy.delete(image, i, axis=0)
                else:
                    i += 1

            # Удаление белых прямоугольников слева и справа изображения
            i = 0
            while i < image.shape[1]:
                if all(image[:,i] > 230):
                    image = numpy.delete(image, i, axis=1)
                else:
                    i += 1
            image = pilImage.fromarray(image)

            # Если изображение прямоугольное, то преобразование его в квадратное
            width, height = image.size
            if width > height:
                background = pilImage.new('L', (width, width), (255))
                background.paste(image, (0, (width - height) // 2))
                image = background
            elif width < height:
                background = pilImage.new('L', (height, height), (255))
                background.paste(image, ((height - width) // 2, 0))
                image = background

            # Уменьшение разрешения до 28х28 пикселей и добавление белой рамки шириной 3 пикселя
            image.thumbnail((28, 28), pilImage.ANTIALIAS)
            background = pilImage.new('L', (28+6, 28+6), (255))
            background.paste(image, (3, 3))
            image = background
            image.thumbnail((28, 28), pilImage.ANTIALIAS)

            # Инвертирование цвета (что бы получить белые цифры на чёрном фоне)        
            image = numpy.array(image)
            image = 255 - image        

            source_images.append(image)
            
            # Сохранение меток изображений (т.е. какие цифры на них изображены)
            label = f_name_source_image[0]
            classes.append(label)
            k += 1

        # Перемешивание полученной обучающей выборки
        dataset = [ [source_images, classes] for source_images, classes in zip(source_images, classes) ]
        numpy.random.shuffle(dataset)
        source_images = [ source_images for [source_images, classes] in dataset ]
        classes = [ classes for [source_images, classes] in dataset ]
        
        print('[i] Сохранение в %s' % f_training_sample)
        source_images = numpy.array(source_images)
        classes = numpy.array(classes)
        numpy.savez_compressed(f_training_sample, x_train=source_images[:int(source_images.shape[0]*0.8)], y_train=classes[:int(classes.shape[0]*0.8)], 
                                x_test=source_images[int(source_images.shape[0]*0.8):], y_test=classes[int(classes.shape[0]*0.8):])




def main():
    f_training_sample_mnist = 'data/mnist.npz'
    f_net_model = 'data/lenet_model.json'
    f_net_weights = 'data/lenet_weights.h5'
    f_test_classification_img = 'images/test_classification.jpg'
    folder_source_images = 'data/source_images'
    f_training_sample = 'data/training_sample.npz'

    #lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img)    
    #lenet.train(f_training_sample)

    if len(sys.argv) > 1:
        if sys.argv[1] == '-c':
            if len(sys.argv) > 2:
                if os.path.exists(sys.argv[2]) and sys.argv[2].find('.npz') == -1:
                    if len(sys.argv) > 3:
                        if sys.argv[3].find('.npz') != -1:
                            lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
                            lenet.create_training_sample(sys.argv[2], sys.argv[3])
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                    else:
                        lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
                        lenet.create_training_sample(sys.argv[2])
                elif sys.argv[2].find('.npz') != -1:
                    lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
                    lenet.create_training_sample(folder_source_images, sys.argv[2])
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else:
                lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
                lenet.create_training_sample(folder_source_images)
        elif os.path.exists(sys.argv[1]) and sys.argv[1].find('.npz') != -1:
            lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
            lenet.train(f_training_sample)
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые варианты работы:')
            print('\tбез аргументов - запуск обучения сети на обучающей выборке MNIST')
            print('\ttraining_sample.npz - запуск обучения сети на обучающей выборке training_sample.npz')
            print('\t-c - создать обучающую выборку на основе изображений из data/source_images')
            print('\t-c my_data/source_images - создать обучающую выборку на основе изображений из my_data/source_images')
            print('\t-c training_sample.npz - создать обучающую выборку на основе изображений из data/source_images и сохранить в training_sample.npz')
            print('\t-c my_data/source_images training_sample.npz - создать обучающую выборку на основе изображений из my_data/source_images и сохранить в training_sample.npz\n')
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
    else:
        lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img) 
        lenet.train(f_training_sample_mnist)


if __name__ == '__main__':
    main()