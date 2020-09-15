docker build -t cifar10-classification .
docker run -v `pwd`/../:/opt/ -t cifar10-classification cifar10_pretrained_classifier.py
