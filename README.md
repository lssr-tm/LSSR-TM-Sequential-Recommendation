# LSSR-TM Sequential Recommendation

This repository provides the source code for the paper:

**Long- and Short-Term Sequential Recommendation Model Based on Transformer and Mixture of Experts**

## Structure

.
├── README.md
└── LSSR-TM/
    ├── requirements.txt
    ├── train/lssr.py
    ├── LSSR/model.py
    ├── datasets/
    └── data/

- `LSSR-TM/train/lssr.py`: training and evaluation script.
- `LSSR-TM/LSSR/model.py`: model implementation.
- `LSSR-TM/datasets/`: dataset preprocessing and loading scripts.
- `LSSR-TM/data/`: directory for local raw datasets.

## Environment

Install dependencies with:

    cd LSSR-TM
    pip install -r requirements.txt

## Datasets

The experiments use MovieLens-1M, Amazon Beauty, Amazon Games, Steam, and XLong.

Raw datasets are not included in this repository. Please download them from their official sources and place them in the local data directory.

The default raw data paths in `LSSR-TM/train/lssr.py` can be modified according to your local environment.

## Run

Example on MovieLens-1M:

    cd LSSR-TM
    python train/lssr.py --dataset ml-1m --ablation full --raw_path ./data/ratings.dat

Supported datasets:

- `ml-1m`
- `beauty`
- `games`
- `steam`
- `xlong`

Supported ablation settings:

- `full`
- `wo_timestamp`
- `wo_moe`
- `wo_gate`
- `wo_st2lt`

## XLong

The public release of XLong does not provide real interaction timestamps. Therefore, timestamp-related inputs are disabled automatically in XLong experiments.

## Output

Experimental results are saved under:

    LSSR-TM/results/

## Citation

Chang Guo, Yunfei Du, Jun Wang, Ziyao Geng, Xiaoyang Guo, and Bo Zhang.  
Long- and Short-Term Sequential Recommendation Model Based on Transformer and Mixture of Experts.  
Expert Systems with Applications.