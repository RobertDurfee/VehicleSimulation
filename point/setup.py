from setuptools import setup, find_packages

required_packages = [ 'Keras==2.2.4' ]

setup(
    name='point',
    description='vehicle simulation point model',
    version='0.1.0',
    install_requires=required_packages,
    packages=find_packages(),
    include_package_data=True
)