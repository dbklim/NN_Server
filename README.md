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
3. GET-запрос на `/lenet/train` запустит обучение сети в фоне (т.е. в отдельном потоке) на наборе рукописных цифр MNIST (после запуска, на запросы 3, 4 и 5 сервер будет отвечать ошибкой с пояснением, что сеть на данный момент обучается и недоступна для использования)
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

---

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

# Нейронная сеть LeNet

Описание сети


# Клиент для общения с RESTful сервером

Это простой клиент, использующий модуль requests для отправки запросов серверу и получения ответов. Реализация находится в модуле `nn_client.py`.

Данный клиент можно использовать двумя способами:
1. Как самостоятельное консольное приложение.
2. В составе другого приложения из кода python.

**Использование клиента как самостоятельное консольное приложение**

В этом варианте клиент можно запустить, выполнив в терминале `python3 nn_client.py` и после этого выбрав нужный пункт из предложенного меню. Так же клиент поддерживает аргументы командной строки. Они имеют следующую структуру: 
`[host:port] [имя_сети операция_над_сетью]`

Возможные значения `аргумента имя_сети`:
1. `list_nn` - получить список имеющихся нейронных сетей и их адреса
2. `lenet` - использовать сеть LeNet, которая предназначена для классификации рукописных цифр

Возможные значения аргумента `операция_над_сетью`:
1. `classify` - классифирует изображение (или другой объект, в случае появления новых нейронных сетей в составе сервера), если сеть это поддерживает
2. `status` - получить статус сети (точность работы и дату последнего обучения)
3. `about` - получить информацию о сети
4. `train` - запустить обучение сети (при этом на большинство запросов к этой сети сервер будет отвечать ошибкой, что сеть на данный момент недоступна, пока не закончится процесс обучения)

Список возможных комбинаций аргументов командной строки и их описание:
1. без аргументов - запуск с автоопределением адреса машины в локальной сети и портом `5000`. Например: `python3 nn_client.py`
2. `host:port` - запуск с подключением к указанному `host` и `port`. Например: `python3 nn_client.py 192.168.2.102:5000`
3. `host:port list_nn` - получить список имеющихся нейронных сетей и их адреса. Например: `python3 nn_client.py 192.168.2.102:5000 list_nn`
4. `host:port lenet classify` - классифицировать изображение с цифрой с помощью сети LeNet (будет предложено выбрать цифру из каталога `images`). Например: `python3 nn_client.py 192.168.2.102:5000 lenet classify`
5. `host:port lenet classify image.jpg` - классифицировать изображение из файла `image.jpg/.png/.bmp/.tiff` с помощью сети LeNet. Например: `python3 nn_client.py 192.168.2.102:5000 lenet classify my_folder/my_number.jpg` 
6. `host:port lenet status` - получить статус сети LeNet. Например: `python3 nn_client.py 192.168.2.102:5000 lenet status`
7. `host:port lenet about` - получить информацию о сети LeNet. Например: `python3 nn_client.py 192.168.2.102:5000 lenet status`
8. `host:port lenet train` - запустить обучение сети LeNet на наборе данных MNIST. Например: `python3 nn_client.py 192.168.2.102:5000 lenet train`
9. `list_nn` - получить список имеющихся нейронных сетей и их адреса. Например: `python3 nn_client.py list_nn`
10. `lenet classify` - классифицировать изображение с цифрой с помощью сети LeNet (`host` определяется автоматически, `port = 5000`). Например: `python3 nn_client.py lenet classify`
11. `lenet classify image.jpg` - классифицировать `image.jpg/.png/.bmp/.tiff` с помощью сети LeNet (`host` определяется автоматически, `port = 5000`). Например: `python3 nn_client.py lenet classify my_folder/my_number.jpg`
12. `lenet status` - получить статус сети LeNet (`host` определяется автоматически, `port = 5000`). Например: `python3 nn_client.py lenet status`
13. `lenet about` - получить информацию о сети LeNet (`host` определяется автоматически, `port = 5000`). Например: `python3 nn_client.py lenet about`
14. `lenet train` - запустить обучение сети LeNet на наборе данных MNIST (`host` определяется автоматически, `port = 5000`). Например: `python3 nn_client.py lenet train`
            
---

**Использование клиента в составе другого приложения из кода python**



---

Если у вас возникнут вопросы, можете написать мне на почту: vladsklim@gmail.com
