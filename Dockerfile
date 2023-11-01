FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y build-essential \
        cmake \
        curl \
        fasd \
        git \
        htop \
        ncdu \
        powerline \
        python3-dev \
        rsync \
        vim \
        wget

# Default powerline10k theme, no plugins installed
# Uses "git", "ssh-agent" and "history-substring-search" bundled plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -p git -p ssh-agent -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

RUN pip install \
    torch_geometric==2.2 \
    yacs  \
    torchmetrics \
    performer-pytorch \
    ogb \
    wandb \
    torch_sparse \
    torch_scatter \
    fire \
    ranky \
    loguru \
    pysnooper \
    lightning \
    numba \
    diffsort