# OMNI: Open-endedness via Models of human Notions of Interestingness

This is the source code repository for the [OMNI: Open-endedness via Models of human Notions of Interestingness]() paper.

## Code Layout
- `crafter/` folder contains Crafter environment code
- `envs/` folder contains different environment settings presented in the paper
    - `env_<tasks in env>_<method>.py` scripts
    - tasks in env can be: `tr` (repetitive), `trc` (repetitive and compounds), `trs` (repetitive and synonyms)
    - methods can be: `uni` (uniform sampling), `lp` (learning progress), `omni` (OMNI), `omoi` (oracle)
- `moi_saved/` folder contains the processed data from cached GPT-3 predictions
- `evaluate.py` script is used for visualizations and evaluations of a trained agent
- `generate_plots.py` script is used to generate the plots presented in the paper
- `model.py` specifies the model architecture
- `train.py` script is used to train the reinforcement learning agent

## Setup
Clone the repository with `git clone <repo_url> && cd omni_code`.\
Create python virtual environment `python3 -m venv venv`.\
Activate python virtual environment `source venv/bin/activate`.\
Install dependencies `pip -r install requirements.txt`.

## Training
Run `train.py` script with the necessary args:
```
python train.py --model <model_name> --env <env_name>
```

## Evaluation
Run `evaluate.py` script with the necessary args:
```
python evaluate.py --model <model_name> --env <env_name>
```

## Crafter GUI
Run `crafter/run_gui.py` script with the necessary args:
```
python crafter/run_gui.py --env <env_name>
```

## Acknowledgements
This codebase draws inspiration from the following codebases:
- [Crafter](https://github.com/danijar/crafter)
- [torch-ac](https://github.com/lcswillems/torch-ac)
