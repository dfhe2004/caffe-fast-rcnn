from setuptools import setup, find_packages

from distutils.extension import Extension

setup(
    name = "caffe",
    version = "0.2",
    packages = find_packages(),
    #cmdclass = {'build_ext': build_ext},
    #ext_modules = ext_modules
)

