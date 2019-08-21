FROM osrf/ros2:devel  

# this is already set in ros2 anyway
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && \
    apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    zsh \
    # install lsofa dependencies \
    build-essential \
    cmake \
    ccache \
    libboost-atomic-dev \
    libboost-chrono-dev \
    libboost-date-time-dev \
    libboost-filesystem-dev \
    libboost-locale-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    python2.7-dev \
    python-numpy \
    python-scipy \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    libglew-dev \
    # plugin repos \
    libxml2-dev \
    libcgal-dev \
    libblas-dev \
    liblapack-dev \
    libsuitesparse-dev \
    libassimp-dev  \
    wget \
    # ca-certificates \
    # python3-pip \&& \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CMD ["/bin/bash -c" "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"]
RUN wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O zshinstaller.sh \
    && chmod +x zshinstaller.sh \
    && ./zshinstaller.sh 

# git clone sofa
RUN "$(git clone https://github.com/lakehanne/sofa -b v19.06)" \
    && cd sofa \
    && git checkout -b v19.06 
    # && git pull origin v19.06 \
RUN mkdir -p ~/sofa/build \
    && cd ~/sofa/build \
    && ./../sofa-cmake.sh

# RUN apt-get install wget
# Cleanup

RUN apt-get update && \
    purge -y --auto-remove gcc wget \
    && apt-get clean 

RUN wget https://raw.githubusercontent.com/lakehanne/Shells/master/bash_aliases_docker_ros -O ~/.bash_aliases
RUN echo "source ~/.bash_aliases" >> ~/.zshrc
ENTRYPOINT ["/usr/bin/zsh"]


