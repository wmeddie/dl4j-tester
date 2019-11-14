#!/usr/bin/env bash

java \
    -da \
    -verbose:gc \
    -Xms8m \
    -Xmx256m \
    -Xverify:none \
    -XX:+UnlockExperimentalVMOptions \
    -XX:+UseG1GC \
    -XX:+UseCompressedOops \
    -XX:+UseNUMA \
    -XX:MaxGCPauseMillis=3 \
    -XX:+DisableExplicitGC \
    -XX:+AlwaysPreTouch \
    -Dorg.bytedeco.javacpp.maxbytes=2G \
    -Dorg.bytedeco.javacpp.maxphysicalbytes=2G \
    -Dorg.bytedeco.javacpp.maxretries=1 \
    -cp target/dl4j-tester_2.11-1.0-SNAPSHOT-jar-with-dependencies.jar \
    io.skymind.Main
