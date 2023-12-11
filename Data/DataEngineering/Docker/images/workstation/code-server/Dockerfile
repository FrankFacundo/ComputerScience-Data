FROM ubuntu:23.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=xterm-256color

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
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    openjdk-11-jdk

# Set up virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Oh My Zsh and themes
# p10k configure
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
COPY fonts/ /usr/share/fonts/
RUN fc-cache -fv
ADD powerlevel10k /powerlevel10k
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
ENV ANACONDA_VERSION=2023.09
RUN wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-0-Linux-x86_64.sh -O /tmp/anaconda.sh \
    && /bin/bash /tmp/anaconda.sh -b -p /opt/conda \
    && rm /tmp/anaconda.sh
ENV PATH="/opt/conda/bin:$PATH"

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
ENV SPARK_VERSION=3.5.0
ENV HADOOP_VERSION=3
ENV SPARK_HOME=/opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

RUN wget -qO- "https://dlcdn.apache.org/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION.tgz" | tar -xz -C /opt/ \
    && mv /opt/spark-$SPARK_VERSION-bin-hadoop$HADOOP_VERSION $SPARK_HOME

# # Configure Spark
# COPY spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf
# COPY log4j.properties $SPARK_HOME/conf/log4j.properties

# Install Code Server
RUN curl -fsSL https://code-server.dev/install.sh | sh

ADD settings.json /root/.local/share/code-server/User/settings.json
WORKDIR /workspace

COPY extensions /extensions
RUN code-server --install-extension /extensions/eamodio.gitlens-2023.12.904.vsix
RUN code-server --install-extension /extensions/ms-python.python-2023.23.13391009.vsix
RUN code-server --install-extension /extensions/charliermarsh.ruff-2023.50.0@linux-x64.vsix
RUN code-server --install-extension /extensions/esbenp.prettier-vscode-10.1.0.vsix
RUN code-server --install-extension /extensions/mechatroner.rainbow-csv-3.8.0.vsix
RUN code-server --install-extension /extensions/vscode-icons-team.vscode-icons-12.6.0.vsix
RUN code-server --install-extension /extensions/ms-vscode.cpptools-1.18.3@linux-x64.vsix
RUN code-server --install-extension /extensions/ms-azuretools.vscode-docker-1.28.0.vsix

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

# Expose Spark UI port (master and worker)
EXPOSE 8080
EXPOSE 8081

# Expose the default port for Code Server
EXPOSE 40000

# Command to run Code Server
CMD ["code-server", "--bind-addr", "0.0.0.0:40000", "--auth", "none"]
