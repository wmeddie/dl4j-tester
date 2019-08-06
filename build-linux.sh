#!/usr/bin/env bash

docker run \
  -v $PWD:/src \
  -w /src \
  -v $HOME/.m2:/var/maven/.m2 \
  -it \
  --rm \
  -u $UID \
  -e MAVEN_CONFIG=/var/maven/.m2 \
  maven \
  mvn -Duser.home=/var/maven -Pcross-cuda-build clean package

