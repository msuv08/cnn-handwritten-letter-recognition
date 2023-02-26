# Handwritten Letter Recognition Using CNN's
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/torch"> <img src="https://img.shields.io/badge/License-MIT-blue)">

The task of letter recognition, a sub-task of text recognition, is growing in popularity
due to emerging applications in robotics, computer vision, and much more. While
improvement in this field has been made in the recent years, there is still room
for more. Many of the previously developed models have been complex networks
with multiple levels. This paper presents a simple, shallow convolutional neural
network for the task of letter recogntion. This work also explores how to better
regularize and increase the accuracy of the model while keeping it as simple as
possible. It is found that even a simple, one-layer convolutional neural network can
be very effective at letter recognition given that a good dataset is used for training
the model.

## Read our NeurIPS Conference Paper!

- [Click here for PDF](https://drive.google.com/file/d/1Ue-gkw28zc_4sjh_6IPJXJ8qQ5CRBaPF/view?usp=sharing)

- [Open-source LaTeX (Overleaf Project)](https://www.overleaf.com/project/626aff8056d3514ce994b2df)

- [Check out the docs](https://docs.google.com/document/d/1AzhL3GwHnokODFME3JMoKhNrin2qoaYS6MHlrjVUo7k/edit)

<img src="https://user-images.githubusercontent.com/49384703/221442575-a9da3f91-0c65-4559-ab8b-5fe1ff33bdd2.png" width=900>


### Deployment for TTF_to_PNG.py

This script works as follows:

```python
python3 ttf_to_png.py "relative/path/to/font" size
```
Here's an example of it being used:

```python
python3 ttf_to_png.py "data/tffs/P1-Regular.ttf" 28   
```
