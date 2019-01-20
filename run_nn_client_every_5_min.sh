#!/bin/sh

echo "INFO | `date` | Start..."
i=0
while (true) 
do
 echo "INFO | `date` | $i";
 i=$((i + 1));
 python3 nn_client.py nn_server.serveo.net:80 lenet status
 sleep 300; # пауза 5 минут
done;
