# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

REQUIRES=[
    'setuptools == 21.0.0',
    'mistk'
]

setup(
    install_requires=REQUIRES,
    name='yolov5-model',
    packages=find_packages()
)
