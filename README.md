# SGD_ICP_PY
Final project for the class Nuages de points 3D. Python implementation of sgd_icp from the article "Speeding up ICP using Stochastic Gradient Descent" F. Afzal Maken, F. Ramos, L. Ott IEEE International Conference on Robotics Automation, 2019 available [here](https://arxiv.org/abs/1907.09133)

The link to the original implementation is [here](https://bitbucket.org/fafz/sgd_icp/src/master/).  

The original and the python implementations are tested in the main.ipynb. It can be accesed using the following badge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/lucasgneccoh/SGD_ICP/blob/main/main.ipynb)


# Remarks

`pypcd` is not mine. I took it from [this](https://github.com/dimatura/pypcd) repository, but I had to modify the code to be able to use it.
To be able to use `utils_pcd.py` you need to enter the `pypcd` folder and from there run the following commands

```
setup.py build
setup.py install
```

This should install the library using the file I modified.
