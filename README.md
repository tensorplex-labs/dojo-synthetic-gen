# dojo-synthetic-gen

## Summary

Dojo Synthetic-gen produces coding question and answer pairs which are used by [Dojo Subnet](https://github.com/tensorplex-labs/dojo) for gathering human preference data.

## Setup

Before running the synthetic-gen, you will need the following keys:

- Openrouter (from https://openrouter.ai/)

Copy the .env.example file to a .env file and fill in the blanks, here we will use Openrouter as our LLM API provider.

Docker will create a redis instance using the specified `REDIS_USERNAME` and `REDIS_PASSWORD`. You will need these to manually interact with your redis on docker.

```bash
cp .env.example .env

# env vars that need to be filled
REDIS_USERNAME=
REDIS_PASSWORD=
OPENROUTER_API_KEY=
```

## Run with docker-compose

Install docker and docker-compose:

```bash
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
sudo mkdir -m 0755 -p /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Run docker to launch synthetic-gen:

```bash
# Run the service
docker compose up -d
```

## run without docker

```bash
 # run in dev mode with trace logging enabled.
 # when dev mode is enabled, questions will not be popped from redis.
 python main.py --trace --env_name dev

```
