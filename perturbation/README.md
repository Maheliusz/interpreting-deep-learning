## Explaining Deep Learning through meaningful perturbation
### Input layer perturbation
`python explain.py <image_path> [--model=<model_path>]`  
where  
`image_path` is path to image to explain  
`model_path` (optional) is path to .pth torch model file. If not provided, script will use VGG19 model.
### Inner layers perturbation
`python inner_explain.py <image_path> [--model=<model_path>]`
### Results analysis
`python analyse.py <.ndy file(s) path> [--filter] [--stats]`
`python input_mask_inner_analysis.py <.ndy file(s) path>`

Based on https://github.com/jacobgil/pytorch-explain-black-box