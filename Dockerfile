FROM openjdk:12.0.1-jdk

COPY . /usr/src/dl4j-tester
WORKDIR /usr/src/dl4j-tester

CMD ["sh", "run.sh"]

