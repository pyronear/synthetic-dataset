# synthetic-dataset

This is a temporary dataset to organize research on the change-detection project in data for good season 10. It will gather our ideas to create a synthetic change detection dataset and our solutions to detect them. The content of this repo will be shared between the pyro-vision and pyro-dataset repo when we have a stable solution

## Requirements


You first need to install Python dependencies : 
```shell
pip install -r requirements.txt
```

Finally you need to add videos of background and smoke in dedicated folders :
- `videos\background`
- `videos\smoke`

## Create dataset

If you want to generate 3 set please use

```shell
python synthetic-dataset\make_dataset.py --set 3
```

If you whant all possible combinaisons:

```shell
python synthetic-dataset\make_dataset.py
```
