#!/usr/bin/env bash

java \
    -verbose:gc \
    -Xms256m \
    -Xmx256m \
    -XX:+UseG1GC \
    -Dorg.bytedeco.javacpp.maxbytes=2G \
    -Dorg.bytedeco.javacpp.maxphysicalbytes=2G \
    -Dorg.bytedeco.javacpp.maxretries=1 \
    -cp target/dl4j-tester_2.11-1.0-SNAPSHOT-jar-with-dependencies.jar \
    io.skymind.Main
