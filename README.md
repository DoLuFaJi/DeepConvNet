Face recognition project with CNN
=================================

Projet was made with python3.7

Install
> `pip install -r requirements.txt`

Run program
> `python extractor.py [-b 16] [-lr 0.01] [-m 0.2] [-i 30] [-n model_name]`

Load models
> `python extractor.py -l model_name`

For face dectetor in an image
> `python extractor.py -d path_to_image [-c confidence]`
The image resulting will be [./rectangle]

Optional arguments:
- b B        Batch size
- lr LR      Learning rate
- m M        Momentum
- i I        Iterations
- n N        Model name
- l L        Load model
- t T        Use pytorch tutorial net
- d D        Detect faces in an image (path)
- c C        confidence when searching face in an image
