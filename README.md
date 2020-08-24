# Causal Future Prediction in a Minkowski Space-Time

## Where Physics and Machine Learning Meet

### A. Vlontzos, H.B. Rocha, D. Rueckert, B. Kainz


## Setup 

This repo is based of Matthieu et al. Poincare VAE [git link](https://github.com/emilemathieu/pvae)

> pip install -r -U requirements.txt

## Use
To train a model run: 
>main.py --gpu 0 --parameters

In inference mode the our algorithm will automatically 

Populate the function below with your own method of choosing the next point in the latent space. 
This part of the algorithm depends on the target application, see paper for details. 
Given a "count all the white pixels" naive method. 
>compare_images @ utils.py 

 
 ## Citation
````
@article{vlontzos2020causal,
    title={Causal Future Prediction in a Minkowski Space-Time},
    author={Athanasios Vlontzos and Henrique Bergallo Rocha and Daniel Rueckert and Bernhard Kainz},
    year={2020},
    eprint={2008.09154},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

