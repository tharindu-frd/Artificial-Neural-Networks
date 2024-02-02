## Commands

### Create any empty file using git bash

```
touch filename
```

### Create an environment ( we can use any of them , second one will create the env in the current working directory )

```
conda create -n envName python=3.10 -y
conda create --prefix ./env python=3.10 -y
```

### Activate the env

```
conda activate ./env
conda env export > environment.yaml

```

## Now inside the environment.yaml file under dependencies add cudatoolkit=11.2 , cudnn=8.1.0

## Install the requirements

```
pip install -r requirements.txt
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

## make a directory and go inside that

```
mkdir Research_env
cd Research_env/
```

## Start the jupyter notebook

```
jupyter notebook
```

### Once the project is done

```
python src/training.py
tensorboard --logdir logs_dir
```
