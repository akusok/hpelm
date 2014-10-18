# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 16:34:56 2014

@author: akusok
"""

class ELMError(Exception):
    """Expected errors in ELM toolbox, i.e. for incorrect input.
    """
    def __init__(self, code):
        self.code = code
    def __str__(self):
        return repr(self.code)