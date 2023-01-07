"""
The third prototype
"""

from manim import *
import numpy as np
import math

class spring(Scene):
  def construct(self):
    spring = SpringMaob(UR * 2 + UP, 2, width=0.5)

    self.add(NumberPlane())
    # self.add(spring.get_spring())
    # self.wait()

    self.play(spring.animate(6, run_time=4))

def spring_interp (x: float) -> float:
  """
  Interpolates a spring animation
  """
  nx = x - 0.16
  return (pow(2, -10 * nx) * math.sin(50 * nx) + 1.476) * 0.335 + 0.50645

class SpringMaob:
  def __init__(self, position: np.ndarray, height: float, width = 1):
    """
    Creates a spring that has its top fixes to a certain position 
    and the height can be set accordingly
    """
    self.__position: np.ndarray = position
    self.__height = 1

    self.spring = FunctionGraph(lambda t: np.sin(t), color=RED, x_range=[0, 10 * PI], stroke_width=5)
    self.spring.rotate(90 * DEGREES)
    self.spring.stretch(1 / (10 * PI), 1)
    self.spring.stretch(0.5 * width, 0)
    self.spring.move_to(position)
    self.spring.shift(DOWN * 0.5)

    self.set_height(height)

  def set_height(self, height: float) -> None:
    if (height == self.__height):
      return None

    ratio = height / self.__height
    center = self.__position + DOWN * self.__height / 2
    top = height / 2 * UP + center

    self.__height = height
    self.spring.stretch(ratio, 1)
    self.spring.shift(self.__position - top)

  def animate(self, height: float, run_time = 1):
    ratio = height / self.__height
    center = self.__position + DOWN * self.__height / 2
    top = height / 2 * UP + center

    self.__height = height
    return self.spring.animate(rate_func=spring_interp, run_time=run_time).stretch(ratio, 1).shift(self.__position - top)

  def get_spring(self) -> FunctionGraph:
    return self.spring
