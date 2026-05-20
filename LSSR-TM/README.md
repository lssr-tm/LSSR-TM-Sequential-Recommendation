# LSSR-TM

## Structure

- `train/lssr.py`: training and evaluation entry
- `LSSR/model.py`: model definition
- `datasets/`: dataset preprocessing and loading scripts
- `data/`: place raw datasets here before running

## Environment

```bash
pip install -r requirements.txt
```

## Example

```bash
python train/lssr.py --dataset ... --ablation ... --raw_path ./data/ratings_....csv
```

## Batch experiment plans

```bash
python train/lssr.py --run_plan plan_a
python train/lssr.py --run_plan plan_b
```

## Notes

- Raw datasets are not included in this repository.
- For XLong, the public release does not provide real timestamps.
