# OMNI: Open-endedness via Models of human Notions of Interestingness  [[Arxiv]](https://arxiv.org/abs/2306.01711) [[Website]](http://www.jennyzhangzt.com/omni/) [[Tweet]](https://twitter.com/jeffclune/status/1666082258888056834)

https://github.com/jennyzzt/omni/assets/53294998/a681f581-58ad-4b7f-b365-3c8505d697cf

This is the source code repository for the [OMNI: Open-endedness via Models of human Notions of Interestingness](https://arxiv.org/abs/2306.01711) paper. OMNI utilizes large (language) models as a model of interestingness, because they already internalize human concepts of interestingness from training on vast amounts of human-generated data. This repository implements OMNI on a procedurally generated 2D gridworld gomain Crafter.

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
Install dependencies `pip install -r requirements.txt`.

## Training
Run `train.py` script with the necessary args:
```
python train.py --model <model_name> --env <env_name>
```
For example, in the repetitive Crafter task setting  
Uniform: `python train.py --model tr_uni-1 --env tr_uni --seed 1`  
LP: `python train.py --model tr_lp-1 --env tr_lp --seed 1`  
OMNI: `python train.py --model tr_omni-1 --env tr_omni --seed 1`  

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

## Citation
```
@article{zhang2023omni,
  title={OMNI: Open-endedness via Models of human Notions of Interestingness},
  author={Jenny Zhang and Joel Lehman and Kenneth Stanley and Jeff Clune},
  year={2023},
  journal={arXiv preprint arXiv:2306.01711},
}
```
