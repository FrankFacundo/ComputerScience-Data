FROM ubuntu:23.04

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

RUN apt-get update && apt-get install -y \
    zsh \
    wget \
    git \
    curl \
    byobu \
    htop \
    fonts-powerline \
    fontconfig \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

# Set up virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Oh My Zsh and themes
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY fonts/ /usr/share/fonts/
# Visual Studio Code: Open File → Preferences → Settings, 
# enter "terminal.integrated.fontFamily" in the search box and set the value to "MesloLGS NF".
# p10k configure
RUN fc-cache -fv
ADD powerlevel10k /powerlevel10k
RUN echo 'source /powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc \
    && sed -i.bak '/^ZSH_THEME=/d' ~/.zshrc \
    && chsh -s $(which zsh)

# Install Anaconda
ENV ANACONDA_VERSION=2023.09
RUN wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-0-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh
ENV PATH="/opt/conda/bin:$PATH"

SHELL ["/usr/bin/zsh", "-c"]

# Clone the Zsh plugins
# Clone the desired Zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions.git ~/.zsh/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.zsh/zsh-syntax-highlighting && \
    git clone https://github.com/agkozak/zsh-z.git ~/.zsh/zsh-z && \
    git clone https://github.com/zsh-users/zsh-completions.git ~/.zsh/zsh-completions

# Add the plugin's source to .zshrc
RUN echo "source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc && \
    echo "source ~/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc && \
    echo "source ~/.zsh/zsh-z/zsh-z.plugin.zsh" >> ~/.zshrc && \
    echo "fpath+=(~/.zsh/zsh-completions/src)" >> ~/.zshrc && \
    echo "autoload -U compinit && compinit" >> ~/.zshrc

# Install Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

WORKDIR /workspace

COPY extensions /extensions
RUN code-server --install-extension /extensions/eamodio.gitlens-2023.10.3105.vsix
RUN code-server --install-extension /extensions/ms-python.python-2023.21.13041006.vsix
RUN code-server --install-extension /extensions/ms-toolsai.jupyter-2023.10.1003011100@linux-x64.vsix
RUN code-server --install-extension /extensions/ms-toolsai.jupyter-keymap-1.1.2.vsix
RUN code-server --install-extension /extensions/ms-toolsai.jupyter-renderers-1.0.17.vsix
RUN code-server --install-extension /extensions/ms-toolsai.vscode-jupyter-cell-tags-0.1.8.vsix
RUN code-server --install-extension /extensions/ms-toolsai.vscode-jupyter-slideshow-0.1.5.vsix
RUN code-server --install-extension /extensions/ms-python.pylint-2023.9.12961008.vsix
RUN code-server --install-extension /extensions/esbenp.prettier-vscode-10.1.0.vsix
RUN code-server --install-extension /extensions/eeyore.yapf-0.1.13.vsix

# Expose the default port for Code Server
EXPOSE 40000

# Command to run Code Server
CMD ["code-server", "--bind-addr", "0.0.0.0:40000", "--auth", "none"]
