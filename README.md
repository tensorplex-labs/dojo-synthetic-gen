# dojo-synthetic-api

## v2 architecture

- `run_dojo_v2_process()` runs as a background task. it continuously generates questions and answers until redis contains 4 completed questions. (controlled by `buffer_size` in `config.py`)
- v2 endpoints implement a asynchronous request-reply pattern (eg. dojo requests a code answer, syn-api returns a uid that can later be used to retrieve the code answer.)
- eg.
  - dojo can query the `"/generate-question"` route to retrieve a question + qa_id
  - dojo and syn-api share the same redis instance, dojo can directly retrieve answers from redis using the qa_id as a key.
  - alternatively, ive also made a `"/generate-answer"` endpoint to query for answers
- new endpoints were added to `synthetic_gen.py`. Documentation for each endpoint exists in-file.
- `test_routes.py` is a very hacky script that queries the endpoints for testing.
- <span style="color:red; font-weight:bold">**after running `python main.py` navigate to `http://localhost:5003/docs#/` to view endpoint documentation**</span>

### v2 to-dos:

- error handling when answer or augment fails to generate. Need to store a negative value in redis so dojo can skip that element.
- clean up and enforce types in redis class and `synthetic_gen.py`
- more specific logging
- review and update this README to remove outdated info.
- add proper return types to new functions.
- deprecate un-needed syn-api v1 code.

## Summary

Dojo Synthetic-API produces coding question and answer pairs which are used by [Dojo Subnet](https://github.com/tensorplex-labs/dojo) for gathering human preference data.

Synthetic-API is intended to be run programatically by validators on dojo Subnet. For testing purposes it can be run standalone.

## Setup

Before running the synthetic-api, you will need the following keys:

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

Run docker to launch synthetic-api:

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
