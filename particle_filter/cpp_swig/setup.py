#!/usr/bin/env python

"""
setup.py file for SWIG particleFilter
"""

from distutils.core import setup, Extension


particleFilter_module = Extension('_particleFilter',
                           sources=['particleFilter_wrap.cxx', 'particleFilter.cpp'],
                           )

setup (name = 'particleFilter',
       version = '0.1',
       author      = "Stefan Caldararu",
       description = """particleFilter wrapper""",
       ext_modules = [particleFilter_module],
       py_modules = ["particleFilter"],
       )