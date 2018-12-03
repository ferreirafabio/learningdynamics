# Graph networks for robotic interaction

## run the GN docker image
1. install docker + nvidia-docker (https://github.com/NVIDIA/nvidia-docker)
2. ```sudo docker pull ferreirafabio/gn:latest```
3. ```sudo nvidia-docker run -it ferreirafabio/gn:latest /bin/bash```
4. ```cd /repos/GNforInteraction/mains```
5. Run the code with: ```python3 singulation_graph_nets.py --c "../configs/singulation2.json"```



Pulling, Commiting and Pushing:
- ```sudo docker ps -l```
- ```sudo docker commit <id> ferreirafabio/gn```
- sudo docker push ferreirafabio/gn```
