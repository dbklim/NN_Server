#!/bin/bash
apt-get -y update
apt-get -y dist-upgrade

# Установка пакетов Ubuntu
PACKAGES="build-essential python3.5 python3.5-dev python3-pip net-tools"
apt-get -y install $PACKAGES

# Установка пакетов Python3
yes | pip3 install --upgrade pip
yes | pip3 install requests

# Очистка кеша
apt-get -y autoremove
apt-get -y autoclean
apt-get -y clean


