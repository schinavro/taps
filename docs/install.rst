=======
Install
=======

Requirements
============

For the basic use, Python and Atomic Simulation Environment (ASE) is all it need.
It should work with latest version of python, but the dependt package related to
ase was not able to build with Python >=3.9. I've checked it worked on python 3.7,
Since it relies on the ase, and it seems depends on the version of the python,
I would recommand ver 3.7.

- python <=3.7
- ase
- Julia (optional)
- sbdesc (optional)


Simple installation
===================

With conda one needs python 3.7 environment.

```conda create --new py37 python=3.7```

To activate the environment,

```conda activate py37```

and,
```conda install -c schinavro tapse ```
