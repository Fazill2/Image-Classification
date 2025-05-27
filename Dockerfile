FROM ubuntu:latest
LABEL authors="kubat"

ENTRYPOINT ["top", "-b"]