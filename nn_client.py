#!/usr/bin/python3
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
Простой модуль-клиент для RESTful сервера с нейронными сетями. Необходимо импортировать access_to_nn_server().
Поддерживается: сеть LeNet для классификации рукописных цифр.

Что бы вручную протестировать клиент, выполните main().
'''

import time
import sys
import os
import subprocess
import curses
import requests
import base64


def access_to_nn_server(host, port, name_nn, type_operation, login = None, password = None, https = False, data = None):
    ''' Простой модуль-клиент для RESTful сервера с нейронными сетями. Аргументы:
    1. host - адрес хоста сервера
    2. port - порт сервера
    3. name_nn - имя запрашиваемой нейронной сети
    4. type_operation - тип операции над сетью
    5. login - логин для подключения к серверу
    6. password - пароль для подключения к серверу
    7. https - True, что бы включить режим https
    8. data - передаваемые данные для нейронной сети (например, бинарная строка с изображением)
    9. возвращает строку с ответом сервера, либо строку с ошибкой, либо при запросе lenet status - tuple с точность обучения сети в % и датой последнего обучения 
    
    Поддерживаемые значения для name_nn:
    1. list_nn - получить список имеющихся нейронных сетей и их адреса 
    2. lenet - действия над сетью LeNet

    Поддерживаемые значения для type_operation:
    1. classify - классифицировать изображение
    2. status - получить статус сети (точность классификации и дата последнего обучения) 
    3. train - запустить обучение сети (LeNet: на наборе данных MNIST или своём предварительно созданном наборе данных)
    4. about - получить информацию о сети

    data должна содержать:
    1. Для lenet classify - изображение .jpg/.png/.bmp/.tiff с рукописной цифрой в виде бинарной строки
    2. Для lenet train - значение mnist (набор данных MNIST) или other (свой предварительно созданный набор данных) для выбора соответствующей обучающей выборки '''

    addr = host + ':' + port
    protocol = 'http'
    if https:
        protocol = 'https'
    if login == None:
        login = 'test_nn'
    if password == None:
        password = 'lenet'
    auth = base64.b64encode((login + ':' + password).encode())
    headers = {'Authorization' : 'Basic ' + auth.decode()}

    if name_nn == 'list_nn': # Получить список имеющихся нейронных сетей и их адреса 
        response = requests.get(protocol + '://' + addr + '/list_nn', headers=headers)
        data = response.json()
        text = data.get('text')
        if text == None:
            error = data.get('error')
            return '[E] ' + error
        return text
    elif name_nn == 'lenet': # Нейронная сеть LeNet
        if type_operation == 'classify': # Классифицировать изображение с цифрой
            if data == None:
                return "[E] Argument 'data' is empty!"
            data = base64.b64encode(data)
            data = {'image' : data.decode()}
            response = requests.post(protocol + '://' + addr + '/lenet/classify', headers=headers, json=data)
            data = response.json()
            number = data.get('number')
            if number == None:
                error = data.get('error')
                return '[E] ' + error
            return number
        elif type_operation == 'status': # Получить статус сети (точность классификации и дата последнего обучения)
            response = requests.get(protocol + '://' + addr + '/lenet/train/status', headers=headers)
            data = response.json()
            accuracy = data.get('accuracy')
            datetime = data.get('datetime')
            if accuracy == None and datetime == None:
                error = data.get('error')
                return '[E] ' + error
            return (accuracy, datetime)
        elif type_operation == 'train': # Запустить обучение сети на наборе данных MNIST или предварительно созданном наборе данных
            if data == None:
                return "[E] Argument 'data' is empty!"
            response = requests.get(protocol + '://' + addr + '/lenet/train?training_sample=' + data, headers=headers)
            data = response.json()
            text = data.get('text')
            if text == None:
                error = data.get('error')
                return '[E] ' + error
            return text
        elif type_operation == 'about': # Получить информацию о сети
            response = requests.get(protocol + '://' + addr + '/lenet/about', headers=headers)
            data = response.json()
            text = data.get('text')
            if text == None:
                error = data.get('error')
                return '[E] ' + error
            return text
        elif type_operation == None:
            return "[E] Invalid value of argument 'type_operation': None!"
        else:
            return "[E] Invalid value of argument 'type_operation': " + type_operation + '!'
    elif name_nn == 'seq2seq':
        return 'in development...'
    else:
        return "[E] Invalid value of argument 'type_nn': " + name_nn + '!'




def get_address_on_local_network():
    ''' Определение адреса машины в локальной сети с помощью выполнения ifconfig | grep 'inet addr'
    1. возвращает строку с адресом или 127.0.0.1, если локальный адрес начинается не с 192.168.Х.Х или 172.17.Х.Х '''

    command_line = "ifconfig | grep 'inet addr'"
    proc = subprocess.Popen(command_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out = out.decode()
    if out.find('not found') != -1:
        print("\n[E] 'ifconfig' не найден.")
        sys.exit(0)
    i = 0
    host_192xxx = None
    host_172xxx = None
    while host_192xxx == None or host_172xxx == None:    
        out = out[out.find('inet addr:') + len('inet addr:'):]
        host = out[:out.find(' ')]
        out = out[out.find(' '):]
        if host.find('192.') != -1:
            host_192xxx = host
        elif host.find('172.') != -1:
            host_172xxx = host
        i += 1
        if i >= 10:
            break
    if host_192xxx:
        return host_192xxx
    elif host_172xxx:
        return host_172xxx
    else:
        print("\n[E] Неподдерживаемый формат локального адреса, требуется корректировка исходного кода. Выбран адрес 127.0.0.1.\n")
        return '127.0.0.1'


def main():
    host = '127.0.0.1'
    port = '5000'
    data = None
    type_operation = None
    name_nn = None
    if len(sys.argv) > 1:
        if sys.argv[1].find(':') != -1: # Задание адреса сервера через host:port
            if len(sys.argv) > 2: 
                if sys.argv[2] == 'list_nn': # Получить список имеющихся нейронных сетей и их адреса
                    name_nn = sys.argv[2]
                elif sys.argv[2] == 'lenet': # Нейронная сеть LeNet
                    if len(sys.argv) > 3:
                        if sys.argv[3] == 'classify': # Классифицировать изображение с цифрой
                            if len(sys.argv) > 4: # Задать изображение для классификации
                                if sys.argv[4].find('.jpg') != -1 or sys.argv[4].find('.png') != -1 or sys.argv[4].find('.bmp') != -1 or sys.argv[4].find('.tiff') != -1:
                                    img_path = sys.argv[4]
                                    try:
                                        with open(img_path, "rb") as fh:
                                            data = fh.read()
                                    except FileNotFoundError as e:
                                        print('\n[E] ' + str(e) + '\n')
                                        return
                                else:
                                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[4] + "'. Введите help для помощи.\n")
                                    return
                            type_operation = sys.argv[3]
                        elif sys.argv[3] == 'status': # Получить статус сети (точность классификации и дата последнего обучения)
                            type_operation = sys.argv[3]
                        elif sys.argv[3] == 'train': # Запустить обучение сети на наборе данных MNIST или предварительно созданном наборе данных
                            type_operation = sys.argv[3]
                            if len(sys.argv) > 4:
                                if sys.argv[4] == 'other':
                                    data = 'other'
                                else:
                                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[4] + "'. Введите help для помощи.\n")
                                    return
                            else:
                                data = 'mnist'
                        elif sys.argv[3] == 'about': # Получить информацию о сети
                            type_operation = sys.argv[3]
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                            return
                    else:
                        print("\n[E] Ожидается ещё один параметр после '" + sys.argv[2] + "'. Введите help для помощи.\n")
                        return
                    name_nn = sys.argv[2]
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
                    return
            host = sys.argv[1][:sys.argv[1].find(':')]
            port = sys.argv[1][sys.argv[1].find(':') + 1:]          
        elif sys.argv[1] == 'list_nn': # Получить список имеющихся нейронных сетей и их адреса
            name_nn = sys.argv[1]
            host = get_address_on_local_network()
        elif sys.argv[1] == 'lenet': # Нейронная сеть LeNet
            if len(sys.argv) > 2:
                if sys.argv[2] == 'classify': # Классифицировать изображение с цифрой
                    if len(sys.argv) > 3: # Задать изображение для классификации
                        if sys.argv[3].find('.jpg') != -1 or sys.argv[3].find('.png') != -1 or sys.argv[3].find('.bmp') != -1 or sys.argv[3].find('.tiff') != -1:
                            img_path = sys.argv[3]
                            try:
                                with open(img_path, "rb") as fh:
                                    data = fh.read()
                            except FileNotFoundError as e:
                                print('\n[E] ' + str(e) + '\n')
                                return
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                            return
                    type_operation = sys.argv[2]
                elif sys.argv[2] == 'status': # Получить статус сети (точность классификации и дата последнего обучения)
                    type_operation = sys.argv[2]
                elif sys.argv[2] == 'train': # Запустить обучение сети на наборе данных MNIST или предварительно созданном наборе данных
                    type_operation = sys.argv[2]
                    if len(sys.argv) > 3:
                        if sys.argv[3] == 'other':
                            data = 'other'
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                            return
                    else:
                        data = 'mnist'
                elif sys.argv[2] == 'about': # Получить информацию о сети
                    type_operation = sys.argv[2]
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
                    return
            else:
                print("\n[E] Ожидается ещё один параметр после '" + sys.argv[1] + "'. Введите help для помощи.\n")
                return
            name_nn = sys.argv[1]
            host = get_address_on_local_network()
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые варианты работы:')
            print('\tбез аргументов - запуск с автоопределением адреса машины в локальной сети и портом 5000')
            print('\thost:port - запуск с подключением к host:port')
            print('\thost:port list_nn - получить список имеющихся нейронных сетей и их адреса')
            print('\thost:port lenet classify - классифицировать изображение с цифрой с помощью сети LeNet')
            print('\thost:port lenet classify image.jpg - классифицировать image.jpg/.png/.bmp/.tiff с помощью сети LeNet')
            print('\thost:port lenet status - получить статус сети LeNet')
            print('\thost:port lenet about - получить информацию о сети LeNet')
            print('\thost:port lenet train - запустить обучение сети LeNet на наборе данных MNIST')
            print('\thost:port lenet train other - запустить обучение сети LeNet на предварительно созданном наборе данных')
            print('\tlist_nn - получить список имеющихся нейронных сетей и их адреса')
            print('\tlenet classify - классифицировать изображение с цифрой с помощью сети LeNet (host определяется автоматически, port = 5000)')
            print('\tlenet classify image.jpg - классифицировать image.jpg/.png/.bmp/.tiff с помощью сети LeNet (host определяется автоматически, port = 5000)')
            print('\tlenet status - получить статус сети LeNet (host определяется автоматически, port = 5000)')
            print('\tlenet about - получить информацию о сети LeNet (host определяется автоматически, port = 5000)')
            print('\tlenet train - запустить обучение сети LeNet на наборе данных MNIST (host определяется автоматически, port = 5000)')
            print('\tlenet train other - запустить обучение сети LeNet на предварительно созданном наборе данных (host определяется автоматически, port = 5000)\n')
            return
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
            return
    else: # Определение host автоматически
        host = get_address_on_local_network()
    
    if type_operation == None and name_nn == None: # Если не были переданы аргументы командной строки
        curses.setupterm()
        print('[i] Выберите вариант работы клиента:')
        print('\t1. list_nn - получить список имеющихся нейронных сетей и их адреса')
        print('\t2. lenet classify - классифицировать изображение с цифрой с помощью сети LeNet')
        print('\t3. lenet status - получить статус сети LeNet')
        print('\t4. lenet about - получить информацию о сети LeNet')
        print('\t5. lenet train - запустить обучение сети LeNet на наборе данных MNIST')
        print('\t6. lenet train other - запустить обучение сети LeNet на предварительно созданном наборе данных')
        name_nn = 'lenet'
        while True:
            choice = input('Введите цифру: ')
            if choice == '1':
                name_nn = 'list_nn'
                break
            elif choice == '2':
                type_operation = 'classify'
                if data == None:
                    for_exit = True
                    while for_exit:
                        number_image = input('Введите номер изображения из директории image с цифрой для распознавания (0..9): ')
                        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            if number_image == digit:
                                for_exit = False
                                break
                        if for_exit:
                            print('Необходимо ввести цифру от 0 до 9!')
                    # Загрузка изображения
                    img_path = 'images/' + number_image + '.jpg'
                    data = None
                    with open(img_path, 'rb') as f_image:
                        data = f_image.read()
                break
            elif choice == '3':
                type_operation = 'status'
                break
            elif choice == '4':
                type_operation = 'about'
                break
            elif choice == '5':
                choice = input('Вы уверены?(д/н) ')
                if choice == 'д' or choice == 'y':
                    type_operation = 'train'
                    data = 'mnist'
                break
            elif choice == '6':
                choice = input('Вы уверены?(д/н) ')
                if choice == 'д' or choice == 'y':
                    type_operation = 'train'
                    data = 'other'
                break
            else:
                os.write(sys.stdout.fileno(), curses.tigetstr('cuu1'))

    if data == None and type_operation == 'classify': # Если не был передан адрес изображения
        for_exit = True
        while for_exit:
            number_image = input('Введите номер изображения из директории image с цифрой для распознавания (0..9): ')
            for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                if number_image == digit:
                    for_exit = False
                    break
            if for_exit:
                print('Необходимо ввести цифру от 0 до 9!')
        # Загрузка изображения
        img_path = 'images/' + number_image + '.jpg'
        data = None
        with open(img_path, 'rb') as f_image:
            data = f_image.read()

    start_time = time.time()
    try:
        result = access_to_nn_server(host, port, name_nn, type_operation, data=data)
    except requests.exceptions.RequestException as e:
        print('\n[E] ' + str(e) + '\n')
        return
    end_time = time.time()
    if isinstance(result, tuple):
        print('Результат обработки запроса: точность классификации %s%%, дата последнего обучения %s' % (result[0], result[1]))
    elif isinstance(result, list):
        print('Результат обработки запроса: %s' % result)
    elif result.find('[E]') != -1:
        print(result)
    else:
        print('Результат запроса: ' + result)
    print('Обработано за %.4f c' % (end_time - start_time))


if __name__ == '__main__':
    main()