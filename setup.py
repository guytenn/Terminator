from distutils.core import setup
from setuptools import find_packages

install_requires = [
    'GPUtil>=1.4.0',
    'numpy>=1.22.2',
    'opencv-python==4.5.5.64',
    'wandb==0.9.4',
    'gym==0.21.0',
    'protobuf==3.17.3',
    'dm-tree==0.1.5',
    'pandas==1.0.5',
    'scipy==1.4.1',
    'torch>=1.10.0',
    'mlagents_envs==0.28.0',
    'tabulate==0.8.9',
    'ray==1.12.1'
]

setup(
    name='terminator',
    packages=find_packages(),
    version='0.0.1',
    description='Reinforcement Learning with a Termionator',
    long_description=open('./README.md').read(),
    author='Guy Tennenholtz',
    author_email='guytenn@gmail.com',
    url='https://arxiv.org/abs/2005.13239',
    install_requires=install_requires,
    license='MIT'
)