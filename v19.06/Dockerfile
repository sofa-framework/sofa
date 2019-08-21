FROM debian:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

# SHELL ["/bin/bash", "-c"]
# RUN cp /bin/bash /bin/sh
# RUN source ~/.bashrc
RUN mkdir -p /root/home
# WORKDIR /root/home

# this for anaconda
# SHELL ["/bin/bash", "-c"]
RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget \
    apt-utils \
    bzip2 \
    ca-certificates \
    curl \
    git \
    gcc \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# from the repo
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/anaconda3 && \
    rm ~/miniconda.sh && \
    ~/anaconda3/bin/conda clean -tipsy && \
    ln -s ~/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "~/anaconda3/bin/conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

RUN ~/anaconda3/bin/conda update -n base -c defaults conda

RUN ~/anaconda3/bin/conda create -n 36 python=3.6

RUN git clone https://github.com/quick2wire/quick2wire-python-api

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" "cd quick2wire-python-api"]
CMD [ "/bin/bash" "~/anaconda3/bin/conda deactivate"]
CMD [ "/bin/bash" "~/anaconda3/bin/conda activate 36"]
CMD [ "/bin/bash" "python3" "setup.py install --user"]
CMD [ "/bin/bash" "cd ../ && rm -rf quick2wire-python-api"]
CMD [ "/bin/bash" "git clone https://github.com/Roboteurs/slushengine"]
CMD ["/bin/bash" "cd slushengine"]
CMD ["/bin/bash" "python setup.py install --user"]
CMD ["/bin/bash" "cd ../ && rm -rf slushengine"]
CMD ["wget" "https://raw.githubusercontent.com/lakehanne/Shells/master/bash_aliases_docker_ros -O ~/.bash_aliases"]

RUN git clone git://github.com/lakehanne/hexapod.git  ~/home/Documents
CMD ["/bin/bash" " cd ~/home/Documents/Hexapod/python"]
CMD ["/bin/bash" " for pkg in $(cat requirements.txt); do conda install $pkg; done \"]
CMD ["/bin/bash" " cd ~" ]

CMD [ "/bin/bash" "source ~/.bashrc" ]
CMD [ "/bin/bash" "~/anaconda3/bin/conda activate 36"]
CMD [ "/bin/bash" "pip3 install spidev --user" ]

# remove unneeded items
RUN apt-get update \
	&& rm -rf /var/lib/apt/lists/* \
	&& apt-get purge -y --auto-remove gcc wget \
    && apt-get clean \
	&& rm -rf ~/.ssh

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD ["/bin/bash"]
