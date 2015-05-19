from setuptools import setup,find_packages

setup(name='pyriemann',
      version='0.2',
      description='Riemannian Geometry for python',
      url='',
      author='Alexandre Barachant',
      author_email='alexandre.barachant@gmail.com',
      license='GPLv3',
      packages=find_packages(),
      install_requires=['numpy','scipy','scikit-learn'],
      zip_safe=False)