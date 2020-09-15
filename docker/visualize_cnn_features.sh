docker build -t cifar10-classification .
docker run -v `pwd`/../:/opt/ -t cifar10-classification visualize_cnn_features.py
