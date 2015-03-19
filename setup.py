

from setuptools import setup
import plato
import utils
import plotting
import general


setup(name='plato',
      author='Plato',
      author_email='poconn4@gmail.com',
      url='https://github.com/petered/plato',
      long_description='Deep Learning Library built on top of Theano',
      version=0,
      packages=['plato'],
      scripts=['bin/plato'])
