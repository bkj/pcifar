FROM nvidia/cuda:8.0-cudnn5-devel

# Install dependencies
RUN apt-get -y update
RUN apt-get install -y git wget

# Install anaconda
RUN wget https://repo.continuum.io/archive/Anaconda2-4.4.0-Linux-x86_64.sh
RUN bash Anaconda2-4.4.0-Linux-x86_64.sh -b -p $HOME/.anaconda
RUN rm Anaconda2-4.4.0-Linux-x86_64.sh

# Install packages
RUN pip install tqdm keras
RUN conda install -y pytorch torchvision cuda80 -c soumith
RUN conda install -y tensorflow-gpu
RUN conda install -y -c conda-forge awscli
RUN apt install -y jq

# Install pcifar
ADD ./install-pcifar.sh /root/install-pcifar.sh
RUN chmod -x /root/install-pcifar.sh
RUN sh /root/install-pcifar.sh

CMD /bin/bash
