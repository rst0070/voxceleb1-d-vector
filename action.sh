sudo docker container rm vox-rst-adam2
sudo docker build -t vox_adam .
sudo nvidia-docker run --name vox-rst-adam2 --shm-size=50gb -v /home/yeongsoo/datas/VoxCeleb1:/data -t vox_adam:latest