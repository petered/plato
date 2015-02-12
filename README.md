The Plato repo contains all the work of my PhD so far.  The intent is for it to evolve into a clean, well-tested library, built on top of theano, containing standard components in deep learning (MLPs, DBNs, etc), so that people don't have to keep reinventing the wheel every time they do a project in deep learning.

To separate the messy, experimental stuff from the clean, reuseable stuff, I've separated the repo into several packages, which should be split off into separate repos once things mature.  The packages are as follows:

**plato** - Contains a very nice framework built on top of theano, primarily for doing deep learning stuff.  Things in here should in theory be nice, clean, and tested.

**utils** - Various utilities that don't involve theano (for loading datasets, doing numpy operations, comparing predictors, etc)

**misc** - Various scripts and experiments.  Don't expect much hygene here.

**general** - Generally useful things that don't necessairily relate to machine learning.  Things here should be nice and individually useful, and probably just included in python/numpy one day.

**plotting** - Code related to plotting - this should be moved out to a separate repo once it's satisfactory and stable.

Dependency:
```
misc --> plato --> utils --> general
  |                            ^
  '--> plotting ---------------|
```
Arrow from A to B indicates "A imports from B, but B doesn't import from A"


**Getting Started**
To get started:

# Open a terminal.
# First, make sure you have virtualenv (run `virtualenv` to check).  If you do not, run 
# sudo pip install virtualenv
# Now, cd to whatever folder you store your projects in, and run the following commands in terminal
```
git clone https://github.com/petered/plato.git
cd plato
setup.sh
```
You should be in the virtual env now.  To see if everything worked, try running the RBM demo: 
```
python plato/tools/demo_restricted_boltzmann_machine.py 
```
The next time you return to the project, you just need to do the following:
```
cd path/to/plato/project  # Which you obviously replace by your actual path to the project
source venv/bin/activate  # This will get you inside the venv.
```
And then you're inside the venv and ready to go.
