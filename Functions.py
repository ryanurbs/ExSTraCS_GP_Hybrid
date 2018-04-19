"""
Name:        Functions.py
Authors:     Ryan Urbanowicz - Written at Dartmouth College, Hanover, NH, USA
Contact:     ryan.j.urbanowicz@darmouth.edu
Created:     February 20, 2018 (Siddharth Verma - Netaji Subhas Institute of Technology, Delhi, India.)
Description: This module contains all the tasks related to creation of user defined functions for GP_Trees. Only the function
             create_function is available publicly which can be used to make own function. A dictionary of common
             functions (_common_functions) is created to have a pool of commonly used functions to use them directly
             without creating one.

---------------------------------------------------------------------------------------------------------------------------------------------------------
ExSTraCS V2.0: Extended Supervised Tracking and Classifying System - An advanced LCS designed specifically for complex, noisy classification/data mining tasks,
such as biomedical/bioinformatics/epidemiological problem domains.  This algorithm should be well suited to any supervised learning problem involving
classification, prediction, data mining, and knowledge discovery.  This algorithm would NOT be suited to function approximation, behavioral modeling,
or other multi-step problems.  This LCS algorithm is most closely based on the "UCS" algorithm, an LCS introduced by Ester Bernado-Mansilla and
Josep Garrell-Guiu (2003) which in turn is based heavily on "XCS", an LCS introduced by Stewart Wilson (1995).

Copyright (C) 2014 Ryan Urbanowicz
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABLILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation,
Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
---------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np

''''''

__all__ = ['create_function', 'get_arity']


# Class for declaring own functions
class _Function:

    def __init__(self, function, name, arity, ret_type=None):
        self.function = function
        # A callable function of the form function(a, *args)

        self.name = name
        # Alias name of the function as it would appear everywhere in
        # code and visualisation.

        self.arity = arity
        # No. of inputs of the function

        self.ret_type = ret_type


# This function is called when a new function is to be made.
def create_function(function, name, arity):
    if not isinstance(arity, int):
        raise ValueError('arity of the function %s should be int, got %s' % (name, type(arity)))

    return _Function(function, name, arity);


def _protectedDiv(left, right):
    try:
        # Converting the np primitives to ordinary primitives so as to avoid any numpy related error.
        left = np.float64(left).item()
        right = np.float64(right).item()

        return left / right
    except ZeroDivisionError:
        return 1


def _protectedSqrt(arg):
    return np.sqrt(np.abs(arg))


def get_arity(func):
    return _common_functions[func].arity


def get_function(func):
    return _common_functions[func].function


# Making some common function to be used in the tree. More functions can be created here.
add1 = create_function(np.add, "add", 2)
sub1 = create_function(np.subtract, "sub", 2)
mul1 = create_function(np.multiply, "mul", 2)
div1 = create_function(_protectedDiv, "div", 2)
less1 = create_function(np.less, "lt", 2)
great1 = create_function(np.greater, "gt", 2)
max1 = create_function(np.maximum, "max", 2)
min1 = create_function(np.minimum, "min", 2)
sin1 = create_function(np.sin, "sin", 1)
cos1 = create_function(np.cos, "cos", 1)
tan1 = create_function(np.tan, "tan", 1)
neg1 = create_function(np.negative, "neg", 1)
abs1 = create_function(np.abs, "abs", 1)
sqrt1 = create_function(_protectedSqrt, "sqrt", 1)

_common_functions = {'add': add1,
                     'sub': sub1,
                     'mul': mul1,
                     'div': div1,
                     'lt': less1,
                     'gt': great1,
                     'sqrt': sqrt1,
                     'max': max1,
                     'min': min1,
                     'sin': sin1,
                     'cos': cos1,
                     'tan': tan1,
                     'neg': neg1,
                     'abs': abs1}



