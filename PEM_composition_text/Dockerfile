FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install --no-install-recommends -y -q \
        software-properties-common \
        apt-utils \
        build-essential \
        pkg-config \
        gnupg2 \
        ca-certificates \
        curl \
        git \
        wget \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools

RUN apt-get update && pip3 install --upgrade pip
RUN pip3 install poetry~=1.2 && poetry config virtualenvs.create false
RUN git config --global --add safe.directory /src

########################################
# Application dependencies
WORKDIR /src/
COPY pyproject.toml /src/
RUN poetry install --no-root --no-interaction --no-ansi

########################################
# Add Jupyter notebook
RUN poetry add jupyter pickleshare
ADD https://github.com/krallin/tini/releases/download/v0.6.0/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
EXPOSE 8888
ENTRYPOINT ["/usr/bin/tini", "--"]
