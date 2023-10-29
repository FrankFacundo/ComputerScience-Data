FROM ubuntu:23.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

RUN apt-get update && apt-get install -y \
    zsh \
    wget \
    git \
    curl \
    byobu \
    fonts-powerline

RUN apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

# RUN curl -SLO https://deb.nodesource.com/nsolid_setup_deb.sh \
#     && chmod 500 nsolid_setup_deb.sh \
#     && ./nsolid_setup_deb.sh 21 \
#     && apt-get install nodejs -y

# RUN npm install xterm

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
    && sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="powerlevel10k/powerlevel10k"|' ~/.zshrc \
    && chsh -s $(which zsh)

SHELL ["/usr/bin/zsh", "-c"]

RUN pip install --upgrade pip \
    && pip install jupyterlab

WORKDIR /workspace

EXPOSE 8888

CMD ["jupyter", "lab", "--ip='0.0.0.0'", "--port=8888", "--no-browser", "--allow-root"]