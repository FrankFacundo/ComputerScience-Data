FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    jq \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libx11-6 \
    libxrandr2 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxi6 \
    libxtst6 \
    libcups2 \
    libxss1 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libpango-1.0-0 \
    fonts-liberation \
    xdg-utils \
    ca-certificates \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN CHROME_DRIVER_URL=$(wget -qO- https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json | jq -r '.channels.Stable.downloads.chromedriver[] | select(.platform=="linux64") | .url') && \
    echo "Downloading ChromeDriver from: $CHROME_DRIVER_URL" && \
    wget -q --show-progress -O /tmp/chromedriver.zip "$CHROME_DRIVER_URL" && \
    mkdir -p /tmp/chromedriver_extracted && \
    unzip /tmp/chromedriver.zip -d /tmp/chromedriver_extracted/ && \
    mv /tmp/chromedriver_extracted/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf /tmp/chromedriver.zip /tmp/chromedriver_extracted

RUN echo "Chrome version: $(google-chrome --version)" && \
    echo "ChromeDriver version: $(chromedriver --version)"

CMD ["/bin/sh"]
