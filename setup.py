from setuptools import setup

setup(name='pyriemann',
      version='0.1',
      description='Riemannian Geometry for python',
      url='',
      author='Alexandre Barachant',
      author_email='alexandre.barachant@gmail.com',
      license='GPLv3',
      packages=['pyriemann'],
      install_requires=['numpy','scipy','scikit-learn'],
      zip_safe=False)