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

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
    && git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}/themes/powerlevel10k \
    && sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="powerlevel10k/powerlevel10k"|' ~/.zshrc \
    && chsh -s $(which zsh)

RUN mkdir -p ~/.gitstatus/bin
RUN curl -L https://github.com/romkatv/gitstatus/releases/download/v1.5.1/gitstatusd-linux-x86_64.tar.gz -o ~/.gitstatus/bin/gitstatusd.tar.gz
RUN rm ~/.gitstatus/bin/gitstatusd.tar.gz
RUN echo 'source ~/.gitstatus/gitstatus.plugin.sh' >> ~/.bashrc

SHELL ["/usr/bin/zsh", "-c"]

# Install Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

WORKDIR /workspace

COPY extensions /extensions
RUN code-server --install-extension /extensions/eamodio.gitlens-2023.10.3105.vsix
RUN code-server --install-extension /extensions/ms-python.python-2023.21.13041006.vsix
RUN code-server --install-extension /extensions/ms-toolsai.jupyter-2023.10.1003011100@linux-x64.vsix
RUN code-server --install-extension /extensions/ms-python.pylint-2023.9.12961008.vsix
RUN code-server --install-extension /extensions/esbenp.prettier-vscode-10.1.0.vsix

# Expose the default port for Code Server
EXPOSE 40000

# Command to run Code Server
CMD ["code-server", "--bind-addr", "0.0.0.0:40000", "--auth", "none"]