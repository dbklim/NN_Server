#!/bin/bash
apt-get -y update
apt-get -y dist-upgrade

# Установка Docker
apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
apt-add-repository 'deb https://apt.dockerproject.org/repo ubuntu-xenial main' -y
apt-get -y update
apt-get install -y docker-engine
# Для проверки статуса докера: sudo systemctl status docker	

# Очистка кеша
apt-get -y autoremove
apt-get -y autoclean
apt-get -y clean


