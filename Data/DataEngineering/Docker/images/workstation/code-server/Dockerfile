FROM ubuntu:23.04

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm-256color

RUN apt-get update && apt-get install -y \
    zsh \
    wget \
    git \
    curl \
    byobu \
    htop \
    mlocate \
    fonts-powerline \
    fontconfig \
    iputils-ping \
    nano \
    vim \
    net-tools \
    tldr \
    zip \
    tree \
    procps \
    nvtop \
    build-essential \
    libssl-dev \
    poppler-utils \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    openjdk-11-jdk \
    postgresql \
    postgresql-contrib

RUN apt-get install -y nvidia-cuda-toolkit

# Set up virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Oh My Zsh and themes
# p10k configure
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY fonts/ /usr/share/fonts/
RUN fc-cache -fv
RUN git clone --depth=1 --recurse-submodules https://github.com/romkatv/powerlevel10k.git /powerlevel10k
RUN echo 'source /powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc \
    && sed -i.bak '/^ZSH_THEME=/d' ~/.zshrc \
    && chsh -s $(which zsh)

SHELL ["/usr/bin/zsh", "-c"]

# Clone the Zsh plugins
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

# Install Anaconda
ENV ANACONDA_VERSION=2024.02-1
RUN wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda init zsh
RUN zsh -l

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Scala
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://downloads.lightbend.com/scala/2.13.8/scala-2.13.8.deb && \
    dpkg -i scala-2.13.8.deb && \
    apt-get install -f && \
    rm scala-2.13.8.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Scala and Java paths
ENV SCALA_HOME=/usr/share/scala
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin:$SCALA_HOME/bin

# Install Spark
ENV SPARK_VERSION=3.5.1
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget -qO- "https://dlcdn.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz" | tar -xz -C /opt/ \
    && mv /opt/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION $SPARK_HOME

# # Configure Spark
# COPY spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf
# COPY log4j.properties $SPARK_HOME/conf/log4j.properties


# Install Docker
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates software-properties-common lsb-release gnupg
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

# Adding your user to the Docker group (Optional, depends on your use case)
RUN usermod -aG docker root

# Expose Docker socket
# This allows the container to communicate with the host's Docker daemon.
# Note: This will be mounted when running the container using the `-v /var/run/docker.sock:/var/run/docker.sock` flag.
VOLUME ["/var/run/docker.sock"]

# Install Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

ADD settings.json /root/.local/share/code-server/User/settings.json
WORKDIR /workspace

RUN code-server --install-extension eamodio.gitlens
RUN code-server --install-extension ms-python.python
RUN code-server --install-extension ms-toolsai.jupyter
RUN code-server --install-extension charliermarsh.ruff
# RUN code-server --install-extension njpwerner.autodocstring
RUN code-server --install-extension streetsidesoftware.code-spell-checker
RUN code-server --install-extension esbenp.prettier-vscode
RUN code-server --install-extension mechatroner.rainbow-csv
# RUN code-server --install-extension ms-vscode.cpptools
RUN code-server --install-extension ms-azuretools.vscode-docker
RUN code-server --install-extension zhuangtongfa.Material-theme
RUN code-server --install-extension PKief.material-icon-theme
RUN code-server --install-extension pomdtr.excalidraw-editor
RUN code-server --install-extension ms-python.debugpy
# RUN code-server --install-extension mathematic.vscode-pdf
RUN code-server --install-extension Continue.continue
RUN code-server --install-extension GitHub.github-vscode-theme

RUN rm -rf /var/lib/apt/lists/*

# Expose Spark UI port (master and worker) and Code server
EXPOSE 8080 8081 40000

# Command to run Code Server
CMD ["code-server", "--bind-addr", "0.0.0.0:40000", "--auth", "none"]
