\# LLMCache-Bench



LLMCache-Bench is a benchmarking framework to evaluate \*\*semantic caching strategies\*\* for Large Language Models (LLMs).

It allows running controlled experiments with/without cache, collecting metrics such as latency, hit/miss rates, and accuracy.



\## Goals

* Compare caching strategies ('none', 'vanilla-exact', 'vanilla-approx', 'extended')
* Provide reproducible experiment configs
* Log raw results and aggregated metrics



\## Quick Start (placeholder)

1. Clone this repo
2. Install dependencies: 'pip install -e .'
3. Define your experiment in 'experiments/experiment.yaml'
4. Run benchmark (scripts will be added later)



\## Project Structure

* 'src/' -> main source code
* 'scripts/' -> runner/plotting scripts
* 'experiments/' -> configs
* 'results/' -> raw + aggregated results
* 'docs/' -> documentation
* 'tests/' -> tests



\## Running Tests



We use \[pytest](https://docs.pytest.org/) for unit and integration tests.



\### Recommended way

Always run tests through Python’s `-m` flag to ensure you’re using the correct virtual environment:



```bash

python -m pytest -q



