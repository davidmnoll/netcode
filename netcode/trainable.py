
import numpy as np
import ctypes
from functools import wraps
from typing import Any, Callable, Type, get_args, get_origin, get_overloads
import inspect
from netcode.network import Network
from pprint import pprint as pp


import hello

TYPE_SIZES = {
  int: 'variable',
  float: 64,  # This is platform dependent
  # Add other built-in types as needed
  bool: 1,
}

def get_cast_to_from_hint(func):
  hints = func.__annotations__
  for arg_name, hint in hints.items():

    if hint in TYPE_SIZES:
      return TYPE_SIZES[hint]
    elif hasattr(hint, 'itemsize'):  # Might be a numpy type
      return 8 * hint().itemsize  # Convert byte size to bit size
    elif inspect.isclass(hint): 
      if issubclass(hint, ctypes._SimpleCData):  # Check if it's a ctypes type
        return 8 * ctypes.sizeof(hint)  # Convert byte size to bit size
      else: 
        raise Exception(f"Class {hint} is not supported")
    elif (get_origin(hint) == tuple):
      args = get_args(hint)
      pp(get_origin(hint) == tuple)
      pp(get_overloads(hint))
      pp(args[0])
      pp(type(hint) )
    else:
      raise Exception(f"Type {hint} is not supported")
        #   size = get_size_from_hint(arg_type)
        # pp(arg_type.__args__[0])
        # pp(size)
        # print(f"Argument {arg_name} of type {arg_type} has size: {size} bits")
        # if size == 'unknown':
        #   raise Exception(f"Argument {arg_name} of type {arg_type} has unknown size")


def get_cast_from_from_hint(func):
  pass

class Trainable():
  
  class NetworkFunction():
    def __init__(self, func: Callable):
      self.output_type = None
      self.input_type = None
      self.func = func
      import os
      hello.say_hello()
      self.cast_to = get_cast_to_from_hint(func)
      self.cast_from = get_cast_from_from_hint(func)
      self.network = Network()
      

    def cast_input(self, input_args):
      self.network.set_input(self.cast_to(input_args))

    

    def cast_output(self, output):
      return self.cast_from(output)

    def __call__(self, *args, **kwargs):
      return self.cast_output(self.network.get_output())

    def train(self): 
      self.network.train(self.func)       



  def __init__(self, *args, **kwargs):
    self.network_function = None
    print(f"Trainable.__init__ called with args: {args} and kwargs: {kwargs}")

  def __call__(self, func):
    pp(func.__name__)
    pp(func.__annotations__)
    raise Exception("Not implemented")
    self.network_function = Trainable.NetworkFunction(func)
    self.network_function.train()
    return self.network_function

