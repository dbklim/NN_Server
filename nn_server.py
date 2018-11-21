#!/usr/bin/python3
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#       OS : GNU/Linux Ubuntu 16.04 
# COMPILER : Python 3.5.2
#   AUTHOR : Klim V. O.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

'''
REST-сервер для взаимодействия с нейронными сетями. Используется Flask и WSGIServer.
Поддерживается: сеть LeNet для классификации рукописных цифр.
'''

import os
import sys
import signal
import platform
import base64
import json
import subprocess
import threading
import socket
from logging.config import dictConfig
from datetime import datetime
from functools import wraps

from flask import Flask, redirect, jsonify, abort, request, make_response, __version__ as flask_version
from flask_httpauth import HTTPBasicAuth
from gevent import __version__ as wsgi_version
from gevent.pywsgi import WSGIServer 

from tensorflow import get_default_graph

from lenet import LeNet


# Создание папки для логов, если она была удалена
if os.path.exists('log') == False:
    os.makedirs('log')

# Удаление старых логов
if os.path.exists('log/server.log'):
    os.remove('log/server.log')
    for i in range(1,6):
        if os.path.exists('log/server.log.' + str(i)):
            os.remove('log/server.log.' + str(i))

# Конфигурация логгера
dictConfig({
    'version' : 1,
    'formatters' : {
        'simple' : {
            'format' : '%(levelname)-8s | %(message)s'
        }
    },
    'handlers' : {
        'console' : {
            'class' : 'logging.StreamHandler',
            'level' : 'DEBUG',
            'formatter' : 'simple',
            'stream' : 'ext://sys.stdout'
        },
        'file' : {
            'class' : 'logging.handlers.RotatingFileHandler',
            'level' : 'DEBUG',
            'maxBytes' : 16 * 1024 * 1024,
            'backupCount' : 5,
            'formatter' : 'simple',
            'filename' : 'log/server.log'
        }
    },
    'loggers' : {
        'console' : {
            'level' : 'DEBUG',
            'handlers' : ['console'],
            'propagate' : 'no'
        },
        'file' : {
            'level' : 'DEBUG',
            'handlers' : ['file'],
            'propagate' : 'no'
        }
    },
    'root' : {
        'level' : 'DEBUG',
        'handlers' : ['console', 'file']
    }
})

app = Flask(__name__)
auth = HTTPBasicAuth()
max_content_length = 16 * 1024 * 1024
f_source_data = 'data/mnist.npz'
f_net_model = 'data/lenet_model.json'
f_net_weights = 'data/lenet_weights.h5'
f_datetime_training_networks = 'data/last_datetime_training_networks.txt'
f_test_classification_img = 'images/test_classification.jpg'


def limit_content_length():
    ''' Декоратор для ограничения размера передаваемых клиентом данных. '''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if request.content_length > max_content_length:                
                log('превышен максимальный размер передаваемых данных ({:.2f} кБ)'.format(request.content_length/1024), request.remote_addr, 'error')
                return make_response(jsonify({'error': 'Maximum data transfer size exceeded, allowed only until {: .2f} kB.'.format(max_content_length/1024)}), 413)
            elif request.content_length == 0:
                log('тело запроса не содержит данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains no data.'}), 400)
            elif request.json == None:
                log('тело запроса содержит неподдерживаемый тип данных', request.remote_addr, 'error')
                return make_response(jsonify({'error': 'The request body contains an unsupported data type. Only json is supported'}), 415)
            return f(*args, **kwargs)
        return wrapper
    return decorator


def log(message, addr=None, level='info'):
    ''' Запись сообщения в лог файл с уровнем INFO или ERROR. По умолчанию используется INFO.
    1. addr - строка с адресом подключённого клиента
    2. message - сообщение
    3. level - уровень логгирования, может иметь значение либо 'info', либо 'error' '''
    if level == 'info':
        if addr == None:
            app.logger.info(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.info(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
    elif level == 'error':
        if addr == None:
            app.logger.error(datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)
        else:
            app.logger.error(addr + ' - - ' + datetime.strftime(datetime.now(), '[%Y-%m-%d %H:%M:%S]') + ' ' + message)


lenet = None
result_train_lenet = (99.32, '2018-11-21 17:39:08') # точность в %, дата и время последнего обучения
last_datetime_of_training_lenet = None
thread_for_train = None
http_server = None
# Получение графа вычислений tensorflow по умолчанию (для последующей передачи в другой поток)
graph = get_default_graph()


def train_nn():
    ''' Выполняет обучение сети с перенаправлением всего вывода в файл log/train.log. Выполняется в отдельном потоке. '''
    log('запущено обучение сети')
    global result_train_lenet
    in_terminal = sys.stdout
    sys.stdout = open('log/train.log', 'w')
    with graph.as_default():
        result_train_lenet = lenet.train(f_source_data, True)
    sys.stdout = in_terminal
    log('обучение сети завершено, точность классификации: {:.2f}%, затраченное время: {:.2f} мин'.format(result_train_lenet[0], result_train_lenet[1]))
    global last_datetime_of_training_lenet
    last_datetime_of_training_lenet = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') # Сохранение даты последнего обучения
    with open(f_datetime_training_networks, 'w') as f_datetime:
        f_datetime.write('lenet=' + last_datetime_of_training_lenet)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'The requested URL was not found on the server.'}), 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return make_response(jsonify({'error': 'The method is not allowed for the requested URL.'}), 405)


@app.errorhandler(500)
def internal_server_error(error):
    print(error)
    return make_response(jsonify({'error': 'The server encountered an internal error and was unable to complete your request.'}), 500)


@auth.get_password
def get_password(username):
    # login test_nn, password lenet
    if username == 'test_nn':
        return 'lenet'


@auth.error_handler
def unauthorized():
    return make_response(jsonify({'error': 'Unauthorized access.'}), 401)


@app.route('/')
def root():
    return redirect('/list_nn')


@app.route('/list_nn')
@auth.login_required
def list_nn():
    ''' Возвращает список имеющихся нейронных сетей и их адреса. '''
    return jsonify({'text':['LeNet url:/lenet',
                            'Планируется добавление модели seq2seq...']})


@app.route('/lenet')
@app.route('/lenet/about')
@auth.login_required
def lenet_about():
    ''' Возвращает информацию о сети LeNet. '''
    return jsonify({'text':'Сеть LeNet предназначена для классификации рукописных цифр. В базовом варианте обучена на наборе данных MNIST. ' + \
                           'Cостоит из 5 слоёв:\n' + \
                           '1. Слой свертки: 75 карт признаков, ядро свертки 5х5, функция активации relu\n' + \
                           '2. Слой подвыборки (субдискретизации): размер пула 2х2, метод наибольшего значения\n' + \
                           '3. Слой свертки: 100 карт признаков, ядро свертки 5х5, функция активации relu\n' + \
                           '4. Слой подвыборки (субдискретизации): размер пула 2х2, метод наибольшего значения\n' + \
                           '5. Полносвязный слой: 500 нейронов, преобразует двумерный результат работы сети в одномерный, функция активации ' + \
                           'relu\n' + \
                           '6. Полносвязный выходной слой: 10 нейронов, которые соответствуют классам рукописных цифр от 0 до 9, функция ' + \
                           'активации softmax\n' + \
                           'При обучении ипользовался оптимизатор adam и метод вычисления значения функции потери categorical_crossentropy.'})


@app.route('/lenet/train', methods=['GET', 'POST'])
@auth.login_required
def lenet_train():
    ''' GET-запрос инициирует обучение сети LeNet на наборе рукописных цифр MNIST. '''
    if request.method == 'GET':
        global thread_for_train
        if thread_for_train == None:
            thread_for_train = threading.Thread(target=train_nn, daemon=True)
            thread_for_train.start()
            log('запуск обучения модели LeNet на наборе рукописных цифр MNIST', request.remote_addr)
            return jsonify({'text':'Запущено обучение сети.'})
        elif thread_for_train.isAlive() == False:
            thread_for_train = threading.Thread(target=train_nn, daemon=True)
            thread_for_train.start()
            log('запуск обучения модели LeNet на наборе рукописных цифр MNIST', request.remote_addr)
            return jsonify({'text':'Запущено обучение сети.'})
        else:
            log('обучение сети ещё не завершено', request.remote_addr)
            return make_response(jsonify({'error':'Network training is not yet complete.'}), 406)

    elif request.method == 'POST':
        return jsonify({'text':'В разработке...'})


@app.route('/lenet/train/status')
@auth.login_required
def lenet_train_status():
    ''' Возвращает последний результат обучения сети LeNet. '''
    global thread_for_train
    if thread_for_train == None:
        log('точность классификации: {:.2f}%, дата обучения: {}'.format(result_train_lenet[0], last_datetime_of_training_lenet), request.remote_addr)
        return jsonify({'accuracy':str(result_train_lenet[0]), 'datetime':last_datetime_of_training_lenet})
    elif thread_for_train.isAlive() == False:
        log('точность классификации: {:.2f}%, дата обучения: {}'.format(result_train_lenet[0], last_datetime_of_training_lenet), request.remote_addr)
        return jsonify({'accuracy':str(result_train_lenet[0]), 'datetime':last_datetime_of_training_lenet})
    else:
        log('обучение сети ещё не завершено', request.remote_addr)
        return make_response(jsonify({'error':'Network training is not yet complete.'}), 406)


@app.route('/lenet/classify', methods=['POST'])
@auth.login_required
@limit_content_length()
def lenet_classify():
    ''' Принимает .jpg/.png/.bmp/.tiff файл с изображением цифры, классифицирует её с помощью сети LeNet и возвращает распознанную цифру. '''
    data = request.json
    image = data.get('image')
    if image == None:
        log('json в теле запроса имеет неправильную структуру', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Json in the request body has an invalid structure.'}), 415)
    image = base64.b64decode(image)
    log('принято изображение размером {:.2f} кБ'.format(len(image)/1024), request.remote_addr)

    with graph.as_default():
        number = lenet.classify(image)

    if number.find('error_1') != -1:
        log('сеть недоступна, выполняется обучение сети', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Network is unavailable. Network training is performed.'}), 406)
    elif number.find('error_2') != -1:
        log('изображение имеет неподдерживаемый формат', request.remote_addr, 'error')
        return make_response(jsonify({'error': 'Image has an unsupported format. Supported: jpeg, png, bmp, tiff.'}), 415)
    log("распознана цифра '" + number + "'", request.remote_addr)
    return jsonify({'number':number})

# Всего 5 запросов:
# 1. GET-запрос на /list_nn вернёт информацию об имеющихся нейронных сетях и их адресах
# 2. GET-запрос на /lenet/about вернёт информацию о сети LeNet
# 3. GET-запрос на /lenet/train запустит обучение сети на наборе рукописных цифр MNIST
# 4. GET-запрос на /lenet/train/status вернёт последний результат обучения сети LeNet
# 5. POST-запрос на /lenet/classify принимает .jpg/.png/.bmp/.tiff файл и возвращает распознанную цифру в виде строки

def run(host, port, wsgi = False, https_mode = False):
    ''' Автовыбор доступного порта (если указан порт 0), загрузка языковой модели и нейронной сети и запуск сервера.
    1. wsgi - True: запуск WSGI сервера, False: запуск тестового Flask сервера
    2. https - True: запуск в режиме https (сертификат и ключ должны быть в cert.pem и key.pem), False: запуск в режиме http
    
    Самоподписанный сертификат можно получить, выполнив: openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 '''
    
    if port == 0: # Если был введён порт 0, то автовыбор любого доступного порта
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, 0))
            port = sock.getsockname()[1]
            log('выбран порт ' + str(port))
            sock.close()
        except socket.gaierror:
            log('адрес ' + host + ':' + str(port) + ' некорректен', level='error')
            sock.close()
            return
        except OSError:
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
            sock.close()
            return

    log('Flask v.' + flask_version + ', WSGIServer v.' + wsgi_version)    
    log('установлен максимальный размер принимаемых данных: {:.2f} Кб'.format(max_content_length/1024))
    
    log('загрузка сети LeNet...')
    global lenet
    print()
    lenet = LeNet(f_net_model, f_net_weights, f_test_classification_img)
    print()

    global last_datetime_of_training_lenet
    with open(f_datetime_training_networks, 'r') as f_datetime:
        last_datetimes = f_datetime.readlines()
    last_datetime_of_training_lenet = last_datetimes[0][last_datetimes[0].find('lenet=') + 6:last_datetimes[0].find('\n')]

    if wsgi:
        global http_server
        if https_mode:
            log('WSGI сервер запущен на https://' + host + ':' + str(port) + ' (нажмите Ctrl+C или Ctrl+Z для выхода)')
        else:
            log('WSGI сервер запущен на http://' + host + ':' + str(port) + ' (нажмите Ctrl+C или Ctrl+Z для выхода)')
        try:
            if https_mode:
                http_server = WSGIServer((host, port), app, log=app.logger, error_log=app.logger, keyfile='key.pem', certfile='cert.pem')
            else:
                http_server = WSGIServer((host, port), app, log=app.logger, error_log=app.logger)
            http_server.serve_forever()
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')
    else:
        log('запуск тестового Flask сервера...')
        try:
            if https_mode:
                app.run(host=host, port=port, ssl_context=('cert.pem', 'key.pem'), threaded=True, debug=False)
            else:
                app.run(host=host, port=port, threaded=True, debug=False)
        except OSError:
            print()
            log('адрес ' + host + ':' + str(port) + ' недоступен', level='error')


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
        print("\n[E] Неподдерживаемый формат локального адреса, требуется корректировка исходного кода.\n")
        return '127.0.0.1'


def main():
    host = '127.0.0.1'
    port = 5000
    
    # Аргументы командной строки имеют следующую структуру: [ключ(-и)] [адрес:порт]
    # nn_server.py - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000
    # nn_server.py host:port - запуск WSGI сервера на host:port
    # nn_server.py -d - запуск тестового Flask сервера на 127.0.0.1:5000
    # nn_server.py -d host:port - запуск тестового Flask сервера на host:port    
    # nn_server.py -d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port
    # nn_server.py -s - запуск WSGI сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом 5000
    # nn_server.py -s host:port - запуск WSGI сервера c поддержкой https на host:port
    # nn_server.py -s -d - запуск тестового Flask сервера c поддержкой https на 127.0.0.1:5000
    # nn_server.py -s -d host:port - запуск тестового Flask сервера c поддержкой https на host:port    
    # nn_server.py -s -d localaddr:port - запуск тестового Flask сервера c поддержкой https, автоопределением адреса машины в локальной сети и портом port
    # Что бы выбрать доступный порт автоматически, укажите в host:port или localaddr:port порт 0

    #run(host, port, wsgi=False)

    if len(sys.argv) > 1:
        if sys.argv[1] == '-s': # запуск в режиме https
            if len(sys.argv) > 2:
                if sys.argv[2] == '-d': # запуск тестового Flask сервера
                    if len(sys.argv) > 3:
                        if sys.argv[3].find('localaddr') != -1 and sys.argv[3].find(':') != -1: # localaddr:port
                            host = get_address_on_local_network()
                            port = int(sys.argv[3][sys.argv[3].find(':') + 1:])
                            run(host, port, https_mode=True)
                        elif sys.argv[3].count('.') == 3 and sys.argv[3].count(':') == 1: # host:port                        
                            host = sys.argv[3][:sys.argv[3].find(':')]
                            port = int(sys.argv[3][sys.argv[3].find(':') + 1:])
                            run(host, port, https_mode=True)                
                        else:
                            print("\n[E] Неверный аргумент командной строки '" + sys.argv[3] + "'. Введите help для помощи.\n")
                    else:
                        run(host, port, https_mode=True)

                elif sys.argv[2].count('.') == 3 and sys.argv[2].count(':') == 1: # запуск WSGI сервера на host:port              
                    host = sys.argv[2][:sys.argv[2].find(':')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port, wsgi=True, https_mode=True)               

                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else: 
                host = get_address_on_local_network()
                run(host, port, wsgi=True, https_mode=True)

        elif sys.argv[1] == '-d': # запуск тестового Flask сервера
            if len(sys.argv) > 2:
                if sys.argv[2].find('localaddr') != -1 and sys.argv[2].find(':') != -1: # localaddr:port
                    host = get_address_on_local_network()
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port)
                elif sys.argv[2].count('.') == 3 and sys.argv[2].count(':') == 1: # host:port
                    host = sys.argv[2][:sys.argv[2].find(':')]
                    port = int(sys.argv[2][sys.argv[2].find(':') + 1:])
                    run(host, port)                
                else:
                    print("\n[E] Неверный аргумент командной строки '" + sys.argv[2] + "'. Введите help для помощи.\n")
            else:
                run(host, port)

        elif sys.argv[1].count('.') == 3 and sys.argv[1].count(':') == 1: # запуск WSGI сервера на host:port
            host = sys.argv[1][:sys.argv[1].find(':')]
            port = int(sys.argv[1][sys.argv[1].find(':') + 1:])
            run(host, port, wsgi=True)
        elif sys.argv[1] == 'help':
            print('\nПоддерживаемые варианты работы:')
            print('\tбез аргументов - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000')
            print('\thost:port - запуск WSGI сервера на host:port')
            print('\t-d - запуск тестового Flask сервера на 127.0.0.1:5000')
            print('\t-d host:port - запуск тестового Flask сервера на host:port')
            print('\t-d localaddr:port - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом port')
            print('\t-s - запуск WSGI сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом 5000')
            print('\t-s host:port - запуск WSGI сервера с поддержкой https на host:port')
            print('\t-s -d - запуск тестового Flask сервера с поддержкой https на 127.0.0.1:5000')
            print('\t-s -d host:port - запуск тестового Flask сервера с поддержкой https на host:port')
            print('\t-s -d localaddr:port - запуск тестового Flask сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом port\n')
        else:
            print("\n[E] Неверный аргумент командной строки '" + sys.argv[1] + "'. Введите help для помощи.\n")
    else: # запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом 5000
        host = get_address_on_local_network()
        run(host, port, wsgi=True)




def on_stop(*args):
    print()
    log('сервер остановлен')
    if http_server != None:
        http_server.close()
    sys.exit(0)


if __name__ == '__main__':
    # При нажатии комбинаций Ctrl+Z, Ctrl+C либо закрытии терминала будет вызываться функция on_stop() (Работает только на linux системах!)
    if platform.system() == "Linux":
        for sig in (signal.SIGTSTP, signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, on_stop)
    main()





# NeuralNetworks_Server

Проект состоит из нескольких частей - RESTful сервер для взаимодействия с нейронными сетями, нейронная сеть LeNet и простой клиент для общения с сервером и его тестирования.

Полный список всех необходимых для работы пакетов:
1. Для Python3.5: decorator, Flask v1.0.2, Flask-HTTPAuth v3.2.4, gevent v1.3.7, h5py, Keras v2.2.4, numpy, pillow, requests, tensorflow[-gpu].
2. Для Ubuntu: python3.5-dev, python3-pip, python3-tk, net-tools.

Если вы используете Ubuntu 16.04 или выше, для установки всех пакетов можно воспользоваться `install_packages.sh` (проверено в Ubuntu 16.04). По умолчанию будет установлен [TensorFlow](https://www.tensorflow.org/install/) для CPU. Если у вас есть видеокарта nvidia, вы можете установить [TensorFlowGPU](https://www.tensorflow.org/install/gpu). Для этого необходимо при запуске `install_packages.sh` передать параметр `gpu`. Например:
```
./install_packages.sh gpu
```

# RESTful-сервер

Данный сервер предоставляет REST-api для взаимодействия с нейронными сетями. На данный момент имеется только одна сеть - LeNet для классификации рукописных цифр. Сервер реализован с помощью [Flask](http://flask.pocoo.org/), а многопоточный режим (production-версия) с помощью [gevent.pywsgi.WSGIServer](http://flask.pocoo.org/docs/1.0/deploying/wsgi-standalone/). Также сервер имеет ограничение на размер принимаемых данных в теле запроса равное 16 Мб. Реализация находится в модуле `nn_server.py`.

Запустить WSGI сервер можно выполнив `run_nn_server.sh` (запуск без аргументов командной строки).

Сервер поддерживает аргументы командной строки, которые немного упрощают его запуск. Аргументы имеют следующую структуру: `[ключ(-и)] [адрес:порт]`.

Возможные ключи:
1. `-d` - запуск тестового Flask сервера (если ключ не указывать - будет запущен WSGI сервер)
2. `-s` - запуск сервера с поддержкой https (используется самоподписанный сертификат, получен с помощью openssl)

Допустимые варианты `адрес:порт`:
1. `host:port` - запуск на указанном `host` и `port`
2. `localaddr:port` - запуск с автоопределением адреса машины в локальной сети и указанным `port`
3. `host:0` или `localaddr:0` - если `port = 0`, то будет выбран любой доступный порт автоматически

Список возможных комбинаций аргументов командной строки и их описание: 
1. без аргументов - запуск WSGI сервера с автоопределением адреса машины в локальной сети и портом `5000`. Например: ```python3 rest_server.py```
2. `host:port` - запуск WSGI сервера на указанном `host` и `port`. Например: ```python3 nn_server.py 192.168.2.102:5000```
3. `-d` - запуск тестового Flask сервера на `127.0.0.1:5000`. Например: ```python3 nn_server.py -d```
4. `-d host:port` - запуск тестового Flask сервера на указанном `host` и `port`. Например: ```python3 nn_server.py -d 192.168.2.102:5000```
5. `-d localaddr:port` - запуск тестового Flask сервера с автоопределением адреса машины в локальной сети и портом `port`. Например: ```python3 nn_server.py -d localaddr:5000```
6. `-s` - запуск WSGI сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом `5000`. Например: ```python3 nn_server.py -s```
7. `-s host:port` - запуск WSGI сервера с поддержкой https на указанном `host` и `port`. Например: ```python3 nn_server.py -s 192.168.2.102:5000```
8. `-s -d` - запуск тестового Flask сервера с поддержкой https на `127.0.0.1:5000`. Например: ```python3 nn_server.py -s -d```
9. `-s -d host:port` - запуск тестового Flask сервера с поддержкой https на указанном `host` и `port`. Например: ```python3 nn_server.py -s -d 192.168.2.102:5000```
10. `-s -d localaddr:port` - запуск тестового Flask сервера с поддержкой https, автоопределением адреса машины в локальной сети и портом `port`. Например: ```python3 nn_server.py -s -d localaddr:5000```

Сервер может сам выбрать доступный порт, для этого нужно указать в `host:port` или `localaddr:port` порт `0` (например: ```python3 nn_server.py -d localaddr:0```).

Всего поддерживается 5 запросов:
1. GET-запрос на `/list_nn` вернёт информацию об имеющихся нейронных сетях и их адресах
2. GET-запрос на `/lenet/about` вернёт информацию о сети LeNet
3. GET-запрос на `/lenet/train` запустит обучение сети на наборе рукописных цифр MNIST
4. GET-запрос на `/lenet/train/status` вернёт последний результат обучения сети LeNet (точность классификации и дата последнего обучения)
5. POST-запрос на `/lenet/classify` принимает .jpg/.png/.bmp/.tiff файл и возвращает распознанную цифру в виде строки

---

**Описание сервера**

1. Сервер имеет базовую http-авторизацию. Т.е. для получения доступа к серверу надо в каждом запросе добавить заголовок, содержащий логин:пароль, закодированный с помощью `base64` (логин: `test_nn`, пароль: `lenet`). Пример на python:
```
import requests
import base64

auth = base64.b64encode('test_nn:lenet'.encode())
headers = {'Authorization' : "Basic " + auth.decode()}
```

Выглядеть это будет так:
```
Authorization: Basic dGVzdF9ubjpsZW5ldA==
```

2. В запросе на классификацию (который под номером 5) сервер ожидает .jpg/.png/.bmp/.tiff файл (цветной либо чёрно-белый, размером от 28х28 пикселей) с изображением рукописной цифры (цифра должна занимать большую часть изображения, примеры можно найти в `images/*.jpg`), который передаётся в json с помощью кодировки `base64` (т.е. открывается .jpg/.png/.bmp/.tiff файл, читается в массив байт, потом кодирутеся `base64`, полученный массив декодируется из байтовой формы в строку `utf-8` и помещается в json), в python это выглядит так:
```
# Формирование запроса
auth = base64.b64encode('test_nn:lenet'.encode())
headers = {'Authorization' : "Basic " + auth.decode()}

with open('images/1.jpg', 'rb') as f_image:
    img_data = f_image.read()
data = base64.b64encode(data)
data = {'image' : data.decode()}

# Отправка запроса серверу
response = requests.post('http://' + addr + '/lenet/classify', headers=headers, json=data)

# Разбор ответа
data = response.json()
number = data.get('number')
print(number)
```

---

**Передаваемые данные в каждом запросе**

Все передаваемые данные обёрнуты в json (в том числе и ответы с ошибками).
1. Сервер передаёт клиенту:
```
{
'text' : ['LeNet url:/lenet',
          'Сеть1 url:/сеть1',
          'Сеть2 url:/сеть2']
}
```
2. Сервер передаёт клиенту:
```
{
'text' : 'Описание сети LeNet'
}
```
3. Сервер передаёт клиенту:
```
{
'text' : 'Запущено обучение сети.'
}
```
4. Сервер передаёт клиенту:
```
{
'accuracy' : '99.32', 
'datetime' : '2018-11-21 17:39:08'
}
```
5. Клиент в теле запроса должен отправить:
```
{
'image' : '/9j/4AAQSkZJRgABAQE...'
}
```
Сервер ему передаст:
```
{
'number' : '1'
}
```

В случае возникновения ошибки, сервер передаст клиенту, например (код ответа 401):
```
{
'error': 'Unauthorized access.'
}
```
Переопределены следующие коды ответов: 400, 401, 404, 405, 406, 415, 500

---

**Примеры запросов**

1. GET-запрос на `/list_nn`

Пример запроса, который формирует `python-requests`:
```
GET /list_nn HTTP/1.1
Host: 192.168.2.83:5000
Connection: keep-alive
Accept-Encoding: gzip, deflate
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: python-requests/2.9.1
```

Пример запроса, который формирует curl (`curl -v -u test_nn:lenet -i http://192.168.2.83:5000/list_nn`):
```
GET /list_nn HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: curl/7.47.0
```

В обоих случаях сервер ответил:
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 305
Date: Fri, 21 Nov 2018 15:13:21 GMT

{
'text' : ['LeNet url:/lenet',
          'Сеть1 url:/сеть1',
          'Сеть2 url:/сеть2']
}
```

---

2. GET-запрос на `/lenet/about`

Пример запроса, который формирует `python-requests`:
```
GET /lenet/about HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: python-requests/2.9.1
Connection: keep-alive
Accept-Encoding: gzip, deflate
```

Пример запроса, который формирует curl (`curl -v -u test_nn:lenet -i http://192.168.2.83:5000/lenet/about`):
```
GET /lenet/about HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: curl/7.47.0
```

В обоих случаях сервер ответил:
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1086
Date: Fri, 21 Nov 2018 15:43:06 GMT

{
'text' : 'Описание сети LeNet'
}
```

---

3. GET-запрос на `/lenet/train`

Пример запроса, который формирует `python-requests`:
```
GET /lenet/train HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: python-requests/2.9.1
Connection: keep-alive
Accept-Encoding: gzip, deflate
```

Пример запроса, который формирует curl (`curl -v -u test_nn:lenet -i http://192.168.2.83:5000/lenet/train`):
```
GET /lenet/train HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: curl/7.47.0
```

В обоих случаях сервер ответил:
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1086
Date: Fri, 21 Nov 2018 15:43:06 GMT

{
'text' : 'Запущено обучение сети.'
}
```

---

4. GET-запрос на `/lenet/train/status`

Пример запроса, который формирует `python-requests`:
```
GET /lenet/train/status HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: python-requests/2.9.1
Connection: keep-alive
Accept-Encoding: gzip, deflate
```

Пример запроса, который формирует curl (`curl -v -u test_nn:lenet -i http://192.168.2.83:5000/lenet/train/status`):
```
GET /lenet/train/status HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdF9ubjpsZW5ldA==
User-Agent: curl/7.47.0
```

В обоих случаях сервер ответил:
```
HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 1086
Date: Fri, 21 Nov 2018 15:43:06 GMT

{
'accuracy' : '99.32', 
'datetime' : '2018-11-21 17:39:08'
}
```

5. POST-запрос на `/lenet/classify`

Пример запроса, который формирует `python-requests`:
```
POST /lenet/classify HTTP/1.1
Host: 192.168.2.83:5000
User-Agent: python-requests/2.9.1
Accept: */*
Content-Length: 10739
Connection: keep-alive
Content-Type: application/json
Authorization: Basic dGVzdGJvdDp0ZXN0
Accept-Encoding: gzip, deflate

{
'image' : '/9j/4AAQSkZJRgABAQE...'
}
```

Пример запроса, который формирует curl (`curl -v -u test_nn:lenet -i -H "Content-Type: application/json" -X POST -d '{'image' : '/9j/4AAQSkZJRgABAQE...'}' http://192.168.2.83:5000/lenet/classify`):
```
POST /lenet/classify HTTP/1.1
Host: 192.168.2.83:5000
Authorization: Basic dGVzdGJvdDp0ZXN0
User-Agent: curl/7.47.0
Accept: */*
Content-Type: application/json
Content-Length: 10739

{
'image' : '/9j/4AAQSkZJRgABAQE...'
}
```

Сервер ответил:
```
HTTP/1.1 200 OK
Content-Length: 81
Date: Fri, 02 Nov 2018 15:57:13 GMT
Content-Type: application/json

{
'number' : '1'
}
```

---


