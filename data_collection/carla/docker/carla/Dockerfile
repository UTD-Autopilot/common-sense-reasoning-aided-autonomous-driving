FROM carlasim/carla:0.9.13

USER root

RUN apt-get update ; \
  apt-get install -y xdg-user-dirs

USER carla
WORKDIR /home/carla
