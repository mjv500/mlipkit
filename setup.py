from setuptools import setup, find_packages

setup(
    name='MlipKit',
    version='0.0.0',
    description='A simple python wrapper for Machine Learning Interatomic Potentials',
    author='Samuel Longo',
    author_email='longo.samuel@outlook.it',
    url='https://github.com/chewingram/mlipkit',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ase',
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
    ],
)
