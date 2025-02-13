First, create conda environment from requirements.txt file.
```zsh
conda create --name flowerenv --file requirements.txt
```

Then to run the FL-MNAR algorithm, run the following command:
```zsh
python3 main.py --dataset <name of dataset> --data_dir <directory of dataset if available> --scheduler <value of T_s> --alpha <local learning rate>
```
