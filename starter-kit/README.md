# Agentic Search

Use historical trajectories to help a fixed agent solve new tasks. Your algorithm returns guidance; the official agent chooses actions and the environment scores the result.

| Dataset | Public trajectories | Development tasks |
| --- | ---: | ---: |
| ALFWorld | 1,000 | 50 |
| WebShop | 1,000 | 50 |
| HotpotQA | 1,000 | 50 |

## 1. Get the code

Use **Linux x86_64**, Docker Engine, Docker Compose v2, Git, Python 3.10+, and unzip. Docker installs the project dependencies, including Java and third-party environment code. No GPU is needed.

```bash
git clone https://github.com/wsdm2027-cup/wsdm2027-cup.github.io.git
cd wsdm2027-cup.github.io/starter-kit
```

## 2. Download the data

Download the data bundle from [Hugging Face](https://huggingface.co/datasets/agentic-search-review-team/agentic-search).

Move `data/` and `resources/` into `starter-kit/`, beside `run.py`. Allow about 18 GB for the extracted bundle, plus space for Docker images. The supplied indexes are ready to use.

## 3. Run a baseline

From `starter-kit/`:

```bash
python3 docker/launch.py build
python3 docker/launch.py check
```

The launcher manages both containers: one for the official agent and environments, one for your algorithm. Builds need Internet access; `check` checks the data without calling the model.

The default benchmark model is **Qwen3.5-9B**. Fill in `model.base_url` and your provider's exact model ID in `model.name`; examples are in `config.yaml`. Use an OpenAI-compatible Chat Completions API with tool calling and non-streaming support; keep `provider: OpenAI`. Organizers use their own configuration for evaluation.

Official evaluation uses the model with thinking mode disabled. Disable it for comparable local runs using your provider's documented settings.

Set your key in Bash, then run BM25 on the 50 ALFWorld development tasks:

```bash
read -rsp 'Model API key: ' MODEL_API_KEY
export MODEL_API_KEY
python3 docker/launch.py run --dataset alfworld --method algorithms/baselines/bm25.py
```

Configuration changes take effect on the next run without rebuilding the images. Never include API keys in your submission.

Choose `alfworld`, `webshop`, `hotpotqa` or `all` (all 150 development tasks). Other baselines in `algorithms/baselines/` are `no_retrieval.py`, `random.py` and `dense_static.py`.

Results are under `results/containers_*/`; see each dataset's `summary.json` for its score. To use data stored elsewhere, add `--data /path/to/data --resources /path/to/resources` to check and run commands.

## 4. Develop your algorithm

Start with [algorithms/methods/example.py](algorithms/methods/example.py). Implement `create_algorithm(config)` and the `prepare`, `reset` and `get_guidance` methods described in [base.py](algorithms/base.py). Your method receives the task and observations and returns `Guidance(text, retrieved_ids)`.

- Place all custom code, model files and supporting assets under `algorithms/`, with your method entry point under `algorithms/methods/`. Keep the supplied interface and service files unchanged.
- Add Python dependencies to `docker/requirements.algorithm.txt`, and system dependencies to `docker/Dockerfile.algorithm`.
- Extra models must support CPU and be included during the build. Runtime is offline.
- Keep the template's container settings. Write runtime caches and indexes to `/cache` or `/tmp`; the application directory is read-only.

Rebuild after changing code, dependencies or models, then run your image:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.algorithm -t my-algorithm:cpu .
python3 docker/launch.py run --dataset hotpotqa --algorithm-image my-algorithm:cpu --method algorithms/methods/example.py
```

Replace the example path with your own method path.

## 5. Submit your method

Export your algorithm image:

```bash
mkdir -p submission
docker save my-algorithm:cpu -o submission/algorithm-image.tar
```

Include:

```text
submission/
├── algorithm-image.tar
├── algorithms/
├── docker/
│   ├── Dockerfile.algorithm
│   └── requirements.algorithm.txt
├── .dockerignore
└── README.md
```

Include all custom supporting files under `algorithms/`. Your submission README should give the image tag, method path, build command and details of any extra models or training data. Do not include API keys, the official image or benchmark data. See the [challenge website](https://wsdm2027-cup.github.io/agenticsearch/) for submission announcements.

## 6. Scores and limits

ALFWorld uses task success (0/1), WebShop its official reward (0–1), and HotpotQA normalized answer exact match (0/1). The final score is the average of the three dataset means.

Each task attempt allows **300 seconds, 50 actions and 50 model calls**. Local defaults are **4 CPUs and 8 GB RAM**.

## Third-party sources

This project builds on ALFWorld, WebShop, and HotpotQA. See [third-party sources and licenses](licenses/README.md) for attribution and processing details.

