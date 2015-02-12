The Plato repo contains all the work of my PhD.

To sort out the mess into submesses, I've organized things into packages as follows:

**plato** - Contains a very nice framework built on top of theano, primarily for doing deep learning stuff.  Things in here should in theory be nice, clean, and tested.

**utils** - Various utilities that don't involve theano (for loading datasets, graphs, etc).

**misc** - Various scripts and experiments.  Don't expect much hygene here.

**general** - Generally useful things that don't necessairily relate to machine learning.  Things here should be nice and individually useful.


Dependency:
misc --> plato --> utils --> general
