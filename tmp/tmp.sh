sudo docker build -t tmp .
sudo docker container rm tmp-rst
sudo nvidia-docker run --name tmp-rst -v /home/yeongsoo/datas/VoxCeleb1:/data -t tmp:latest