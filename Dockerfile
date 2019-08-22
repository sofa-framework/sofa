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
    libqt5core5a \
    libqt5charts5 \
    libqt5charts5-dev \
    # xorg \
    # libxll \
    libglew-dev \
    # qt depends
    libfontconfig1-dev \
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
    && ./zshinstaller.sh && rm zshinstaller.sh

# install qt5
# RUN wget http://qt.mirror.constant.com/archive/qt/5.13/5.13.0/qt-opensource-linux-x64-5.13.0.run -O qt.run \
    # && chmod +x qt.run \
    # && ./qt.run && rm qt.run

# git clone sofa
RUN "$(git clone --recursive https://github.com/lakehanne/sofa -b v19.06 /sofa)"
RUN mkdir -p /sofa/build
# COPY . /sofa

RUN cd /sofa/build \
    && cmake -DPLUGIN_IMAGE=ON \
    -DPLUGIN_CGALPLUGIN=ON \
    -DPLUGIN_MANUALMAPPING=ON \
    -DPLUGIN_SOFAALLCOMMONCOMPONENTS=ON \
    -DPLUGIN_SOFADISTANCEGRID=ON \
    -DMODULE_SOFASPARSESOLVER=ON \
    -DPLUGIN_SOFAEULERIANFLUID=ON \
    -DPLUGIN_DIFFUSIONSOLVER=ON \
    -DPLUGIN_SOFAPYTHON=ON \
    -DPLUGIN_VOLUMETRICRENDERING=ON \
    -DSOFA_BUILD_METIS=ON \
    -DPLUGIN_STLIB=ON \
    -DPLUGIN_SOFTROBOTS=ON \
    -DPLUGIN_SOFAROSCONNECTOR=ON \
    -DPLUGIN_ZYROSCONNECTOR=ON \
    -DSOFA_BUILD_MINIFLOWVR=ON \
    -DPLUGIN_MULTITHREADING=ON ../ \
    && echo -e "\nNow making\n\n" \
    && make -j$nproc \
    && echo -e "\n\nNow installing in build folder\n\n" \
    && make install

# cleanup
RUN apt-get update && \
    apt-get purge -y --auto-remove wget \
    && apt-get clean

# RUN wget https://raw.githubusercontent.com/lakehanne/Shells/master/bash_aliases_docker_ros -O ~/.bash_aliases
# RUN echo "source ~/.bash_aliases" >> ~/.zshrc
# ENTRYPOINT ["/usr/bin/zsh"]
