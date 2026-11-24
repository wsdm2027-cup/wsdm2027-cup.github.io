# Agentic Search over Instruction Trajectories

[![Community: join the Discord](https://img.shields.io/badge/Community-Join%20the%20Discord-5865F2?logo=discord&logoColor=white)](https://discord.com/invite/XebBpa4fT) [![GitHub: jiuwenswarm](https://img.shields.io/badge/GitHub-jiuwenswarm-181717?logo=github&logoColor=white)](https://github.com/openJiuwen-ai/jiuwenswarm) [![WeChat: openJiuwen](https://img.shields.io/badge/WeChat-openJiuwen-07C160?logo=wechat&logoColor=white)](assets/openjiuwen-wechat.png)

**WSDM Cup 2027** · [ACM WSDM 2027](https://www.wsdm-conference.org/2027/), February 15–19, 2027, Hong Kong

The agent, its tools, its model and its execution limits are fixed. Your method decides what the agent
reads before each model call: it retrieves historical trajectories, adapts them, and returns guidance —
so better retrieval of past experience makes a better agent. Prize pool **$10,000** across six teams.

> **New here? Read the [challenge site](https://wsdm2027-cup.github.io/agenticsearch/)** — it is the
> authoritative version of the rules. This README is the practical entry point: what you download,
> what you build, and what you submit.

## Quick links

| I want to | Go to |
|---|---|
| Read the rules, task definition, evaluation and dates | <https://wsdm2027-cup.github.io/agenticsearch/> |
| Register my team (opens Nov 18, 2026 · closes Feb 5, 2027) | [Registration form](https://docs.google.com/forms/d/e/1FAIpQLSddTIzBP3PdGSzp6jAiMErPn8EvGdzsxvKf4Ra8H5TgE26DmA/viewform?pli=1) |
| Download trajectories, development tasks and resources | [Hugging Face dataset](https://huggingface.co/datasets/agentic-search-review-team/agentic-search) |
| Get the runtime, baselines and installation guide | [`starter-kit/`](starter-kit/README.md) |
| Submit my image (portal opens Dec 4, 2026) | Portal link TBD — announced on the challenge site |
| Ask a question | [Open an issue](https://github.com/wsdm2027-cup/wsdm2027-cup.github.io/issues) or email <wsdm_cup_2027@163.com> |

## What you are building

At every step of an episode your method is called with the task goal, the current observation, the
available actions and the interaction history. It returns `Guidance(text, retrieved_ids)`. The official
agent appends that text to its prompt and then chooses the next action — you never act, and you never
choose the model.

```
task + history  ──▶  retrieve and adapt  ──▶  guidance in the prompt  ──▶  agent acts
```

Only the middle step is yours. Practical constraints:

- **CPU only.** Any extra model, index or file must be baked into your image at build time.
- **Offline at runtime.** No network calls, no API keys, no downloading during evaluation.
- **Per-task budget:** 300 seconds, 50 actions, 50 model calls.
- **Three domains, equal weight:** ALFWorld, WebShop, HotpotQA.

## At a glance

| | |
|---|---|
| Domains | ALFWorld · WebShop · HotpotQA |
| Public data | 1,000 trajectories + 50 development tasks per domain |
| Hidden test set | 150 tasks per domain |
| Scoring | mean of the three domain means |
| Submission limit | **5 attempts in total**, never reset |
| Prize pool | $10,000 — 1 gold ($3,000), 2 silver ($2,000 each), 3 bronze ($1,000 each) |
| Registration deadline | **Feb 5, 2027** (same day as the submission deadline) |

## Important dates

The timeline is maintained on the challenge site so that it never drifts from the official schedule:

**<https://wsdm2027-cup.github.io/agenticsearch/#dates>**

Three dates you cannot miss: registration opens **Nov 18, 2026**, the submission portal opens
**Dec 4, 2026**, and both the final submission deadline and registration close on **Feb 5, 2027**.

## Data

The bundle ships as `data-resources.zip` (about 5.6 GB) with a matching `SHA256SUMS`. It is a **gated**
dataset: sign in with a Hugging Face account and request access, which is approved automatically.
Extracted, it needs about **18 GB**, plus room for Docker images.

| Dataset | Public trajectories | Development tasks | Hidden test tasks |
|---|---:|---:|---:|
| ALFWorld | 1,000 | 50 | 150 |
| WebShop | 1,000 | 50 | 150 |
| HotpotQA | 1,000 | 50 | 150 |

A trajectory is a task goal, an initial observation, and the action/observation pairs that followed.
Development tasks are disjoint from the trajectory corpus. The same released corpus is available during
hidden-test evaluation, and you may collect or generate extra trajectories for your own training and
local experiments. Indexes are supplied and ready to use; the environment resources (WebShop and
HotpotQA search backends) come with the bundle.

See [`starter-kit/README.md` §2](starter-kit/README.md#2-download-the-data) for fetch, verify and unpack
steps.

## Quick start

Requirements: Linux x86_64, Docker Engine, Docker Compose v2, Git, Python 3.10+, unzip. No GPU. Docker
installs the project dependencies, including Java and third-party environment code.

```bash
git clone https://github.com/wsdm2027-cup/wsdm2027-cup.github.io.git
cd wsdm2027-cup.github.io/starter-kit
# unzip data-resources.zip here so that data/ and resources/ sit beside run.py
python3 docker/launch.py build     # builds the official container and your algorithm container
python3 docker/launch.py check     # validates the data without calling the model
```

Run a baseline on the 50 ALFWorld development tasks:

```bash
read -rsp 'OpenRouter API key: ' OPENROUTER_API_KEY
export OPENROUTER_API_KEY
python3 docker/launch.py run --dataset alfworld --method algorithms/baselines/bm25.py
```

`--dataset` also accepts `webshop`, `hotpotqa` or `all`. Scores land in `results/containers_*/<dataset>/summary.json`.
Of the four supplied baselines, `bm25.py`, `random.py`, `dense_static.py` retrieve one reference
trajectory and `no_retrieval.py` is the floor.

Write your method in `algorithms/methods/` (start from `example.py`), implement `create_algorithm` plus
`prepare`, `reset` and `get_guidance`, then:

```bash
docker build --platform linux/amd64 -f docker/Dockerfile.algorithm -t my-algorithm:cpu .
python3 docker/launch.py run --dataset hotpotqa --algorithm-image my-algorithm:cpu \
  --method algorithms/methods/example.py
```

Dependencies go in `docker/requirements.algorithm.txt` and `docker/Dockerfile.algorithm`; extra model
files belong under `algorithms/`. Write temporary files to `/tmp` or `/cache` — the application
directory is read-only at evaluation time.

## How you are scored

Each domain mean is computed over its hidden test tasks:

- **ALFWorld** — task success, 0 or 1.
- **WebShop** — the environment reward, 0 to 1.
- **HotpotQA** — normalized answer exact match, 0 or 1.

```
overall score = (ALFWorld success rate + WebShop mean reward + HotpotQA accuracy) / 3
```

Tie-break: if two teams share the same mean score, the team with the **smaller standard deviation across
the three domain scores** ranks higher.

Validation baselines from the starter kit; beating Dense Static is the first milestone:

| Method | Overall validation score |
|---|---:|
| No Retrieval | 0.1988 |
| Random | 0.3282 |
| BM25 | 0.4141 |
| Dense Static | 0.4278 |

## What you submit

One exported Docker image plus the code and build files needed to rebuild it:

```text
submission/
├── algorithm-image.tar          # docker save my-algorithm:cpu -o submission/algorithm-image.tar
├── algorithms/                  # your method, plus any model files you ship
├── docker/
│   ├── Dockerfile.algorithm
│   └── requirements.algorithm.txt
├── .dockerignore
└── README.md                    # image tag, method path, build command, extra models or training data
```

Rules that get submissions rejected:

- No API keys, credentials or secrets anywhere in the image or the code.
- Do not include the official agent image or any benchmark/test data.
- Keep the supplied interface and service files; keep the template's container settings.
- The image must rebuild from the files you submit, and must run offline on CPU.

Each team gets **5 attempts in total** for the whole competition, counted from Dec 4, 2026 and never
reset. See [`starter-kit/README.md` §5](starter-kit/README.md#5-submit-your-method) for the checklist.

## Organizers and contact

Huawei Technologies, Co., Ltd. — Zheng Wang, Cong Wang, Didan Deng, Jie Wang, Yangkai Ding, Qijun Li.

For any question about the challenge, [open an issue](https://github.com/wsdm2027-cup/wsdm2027-cup.github.io/issues)
or write to <wsdm_cup_2027@163.com>.

## Third-party sources

The environments build on ALFWorld, WebShop and HotpotQA. Attribution and processing details are in
[`starter-kit/licenses/`](starter-kit/licenses/).
