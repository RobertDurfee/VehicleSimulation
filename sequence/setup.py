from setuptools import setup, find_packages

required_packages = [ 'Keras==2.2.4' ]

setup(
    name='sequence',
    description='vehicle simulation sequence model',
    version='0.1.0',
    install_requires=required_packages,
    packages=find_packages,
    includ_package_data=True
)