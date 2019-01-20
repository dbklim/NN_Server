FROM ubuntu:16.04
MAINTAINER Vlad Klim 'vladsklim@gmail.com'

# Установка необходимых пакетов для Ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y
RUN apt-get install -y tzdata python3.5 python3.5-dev python3-pip python3-tk locales locales-all net-tools
RUN ldconfig

# Установка часового пояса хост-машины
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN dpkg-reconfigure -f noninteractive tzdata

# Копирование файлов проекта
COPY . /NN_Server
WORKDIR /NN_Server

# Установка модулей для Python3
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow==1.7.0 decorator flask==1.0.2 flask-httpauth==3.2.4 gevent==1.3.7 h5py keras numpy pillow requests
RUN ldconfig

# Изменение локализации для вывода кириллицы в терминале
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Очистка кеша
RUN apt-get -y autoremove
RUN apt-get -y autoclean
RUN apt-get -y clean

#ENTRYPOINT ["python3"]
#CMD ["nn_server.py"]


