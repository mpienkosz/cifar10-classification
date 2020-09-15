docker build -t cifar10-classification .
docker run -v `pwd`/../:/opt/ -t cifar10-classification cifar10_shallow_classifier.py
