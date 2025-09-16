# Multi-stage build for Nuitka compilation
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install --user .

FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && apt-get install -y curl docker.io && \
    # Install NodeJS
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install ESLint from package.json
COPY package*.json ./
COPY eslint.config.mjs ./
RUN npm install

COPY --from=builder /root/.local /root/.local

# Add node_modules/.bin to PATH for npx
ENV PATH="/app/node_modules/.bin:${PATH}"

COPY . .

EXPOSE 5003

CMD ["python", "main.py", "--env_name", "prod"]
