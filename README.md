# Assignment4

## Author
- Jere Perisic

## Goal 
- Using Hugging Face to fine-tune the model and test it's predictions

## Input
- Lyrics data which are cleaned as train corpus and validation
- Also, use the same corpus for the test
- Data split -> 80/10/10

## Output

Table 1: F1 scores on the validation set across epochs with varying learning rates.

| Epoch | 0.01   | 0.001  | 1e-5   | 5e-5   |
|-------|--------|--------|--------|--------|
| 1     | 0.25665| 0.48713| 0.52368| 0.51712|
| 2     | 0.36377| 0.41470| 0.51999| 0.51712|
| 3     | 0.24098| 0.41830| 0.53386| 0.52581|
| 4     | 0.16549| 0.48184| 0.52480| 0.50137|

![Plot](plot.png)

Table 2: Testing the model to determine the best learning rate.

| Learning Rate | Blues  | Country | Metal  | Pop    | Rap    | Rock   |
|---------------|--------|---------|--------|--------|--------|--------|
| 0.01          | 0.0    | 0.0     | 0.31612| 0.0    | 0.54054| 0.18518|
| 0.001         | 0.18705| 0.35616 | 0.66990| 0.48591| 0.77659| 0.28025|
| 1e-5          | 0.57875| 0.42253 | 0.66976| 0.49122| 0.78260| 0.20481|
| 5e-5          | 0.56140| 0.43333 | 0.63636| 0.49420| 0.78918| 0.19117|

![Bar Chart](barchart.png)

## Run
```bash
python3 classifier.py
```
