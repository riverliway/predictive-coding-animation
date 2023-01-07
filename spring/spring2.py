"""
The third prototype
"""

from typing import Literal
from manim import *
import numpy as np
import math

POLE_COLOR = '#606060'
INACTIVE_COLOR = '#1e3b69'

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
    self.__pin = 'top'

    self.spring = FunctionGraph(lambda t: np.sin(t), color=RED, x_range=[0, 10 * PI], stroke_width=5)
    self.spring.rotate(90 * DEGREES)
    self.spring.stretch(1 / (10 * PI), 1)
    self.spring.stretch(0.5 * width, 0)
    self.spring.move_to(position)
    self.spring.shift(DOWN * 0.5)

    self.set_height(height)

  def set_height(self, height: float) -> None:
    """
    Sets the spring to the desired height while keeping the pinned end stationary.
    """
    self.__move(height, self.spring)

  def animate(self, height: float, run_time = 1, rate_func = spring_interp):
    """
    Animates the spring moving to the desired height while keeping the pinned end stationary.
    """
    return self.__move(height, self.spring.animate(rate_func=rate_func, run_time=run_time))

  def get_spring(self) -> FunctionGraph:
    """
    Gets the manim object under the hood.
    """
    return self.spring

  def set_pin(self, pin: Literal['top', 'bot']) -> None:
    """
    The pinned end of the spring does not move during animations/setting of height.
    The non-pinned end is allowed to move freely.
    """
    self.__pin = pin

  def __move(self, height, obj):
    """
    Internal function that abstracts the logic for both `set_height` and `animate`.
    Keeps the pinned location in place while extending the non-pinned location to the appropriate height.
    """
    if (height == self.__height):
      return None

    ratio = height / self.__height
    center = self.__position + DOWN * self.__height / 2
    direction = UP if self.__pin == 'top' else DOWN
    pinned_pos = height / 2 * direction + center
    target_pos = self.__position if self.__pin == 'top' else self.__position + DOWN * self.__height

    self.__height = height
    return obj.stretch(ratio, 1).shift(target_pos - pinned_pos)


class spring(Scene):
  def construct(self):
    neuron_locations = self.construct_baseplates()
    self.label = None
    self.train(3, neuron_locations)

  def train(self, step: int, neuron_locs: list[np.ndarray]):
    movement = 0.25
    neuron_locations = [loc for loc in neuron_locs]
    neuron_locations[1] += step * DOWN * movement
    neuron_locations[2] += step * DOWN * movement * 2

    self.change_label('Prediction')
    neurons, weights = self.create_neurons(neuron_locations)

    self.change_label('Calculate Error')
    ghost, spring = self.add_error(neurons[2], step * movement * 2)

    self.wait()

    self.create_pins([n.get_center() for n in neurons])

    self.change_label('Inference')
    self.wait()
    self.inference(neurons, ghost, weights, spring)
    
    self.wait()

  def change_label(self, new):
    if (self.label is not None):
      self.play(FadeOut(self.label))

    self.label = Paragraph(new, alignment='left').shift(LEFT * 6.5 + UP * 3)
    self.label.shift(self.label.get_center() - self.label.get_left())
    self.play(Write(self.label))

  def construct_baseplates(self) -> list[np.ndarray]:
    """
    Constructs the base rectangles and the scales for each neuron
    Returns: a list of coords for where the neurons go
    """
    base = RoundedRectangle(corner_radius=0.1, color=DARK_GRAY, width=1.5, height=2).set_opacity(1).apply_matrix([[1, 1], [0, 2]])
    rect1 = base.copy().shift(1.3 * DOWN + 3 * LEFT)
    rect2 = base.copy().shift(1.3 * DOWN + 1 * LEFT)
    rect3 = base.copy().shift(1.3 * DOWN, 3 * RIGHT)
    self.add(rect1, rect2, rect3)

    pole = Rectangle(color=POLE_COLOR, width=0.2, height=3).set_opacity(1).shift(rect1.get_center()).shift(UP * 1.5)
    dLine1 = DashedLine(rect1.get_center() + 0.5 * LEFT, rect1.get_center() + 0.5 * RIGHT).set_color(POLE_COLOR)
    dLine2 = dLine1.copy().shift(UP * 1.5)
    dLine3 = dLine2.copy().shift(UP * 1.5)

    scale1 = VGroup(pole, dLine1, dLine2, dLine3)
    scale2 = scale1.copy().shift(RIGHT * 2)
    scale3 = scale2.copy().shift(RIGHT * 4.75 + 1.5 * UP)
    scale4 = scale2.copy().shift(RIGHT * 3.25 + 1.5 * DOWN)

    self.add(scale1, scale2, scale3, scale4)

    return [s.get_center() + UP * 1.5 for s in [scale1, scale2, scale3, scale4]]

  def create_neurons(self, neuron_locations: list[np.ndarray]) -> tuple[list[Circle], list[Line]]:
    """
    Creates the neurons and weights on the top of their posts

    Returns: all of the neurons and all of the weights
    """
    neuron = Circle(radius=0.5, color=WHITE, z_index=20).set_opacity(1)
    neuron0 = neuron.copy().shift(neuron_locations[0])
    neuron1 = neuron.copy().shift(neuron_locations[1])
    neuron2 = neuron.copy().shift(neuron_locations[2])
    neuron3 = neuron.copy().shift(neuron_locations[3])
    neurons = [neuron0, neuron1, neuron2, neuron3]

    self.play(AnimationGroup(*[Create(n) for n in neurons], lag_ratio=0.1))

    weight0 = Line(start=neuron0.get_center(), end=neuron1.get_center(), color=YELLOW)
    weight1 = Line(start=neuron1.get_center(), end=neuron2.get_center(), color=YELLOW)
    weight2 = Line(start=neuron1.get_center(), end=neuron3.get_center(), color=YELLOW)

    self.play(AnimationGroup(Create(weight0), Create(VGroup(weight1, weight2), lag_ratio=0), lag_ratio=0.3))

    return neurons, [weight0, weight1, weight2]

  def add_error(self, neuron: Circle, offset: float) -> tuple[Circle, SpringMaob]:
    """
    Adds the error by stretching the neuron to reveal the ghost.
    The height refers to how far down we are on the current training index

    Returns: the ghost neuron and the error spring
    """
    ghost = neuron.copy().set_opacity(0.5)
    self.add(ghost)

    height = 3
    spring = SpringMaob(ghost.get_center(), 0.1, width=0.2)
    stretchAnim = spring.animate(height, rate_func=rate_functions.smooth)
    neuronAnim = neuron.animate().set_fill_color(INACTIVE_COLOR).shift(DOWN * height)
    self.play(AnimationGroup(stretchAnim, neuronAnim))

    return ghost, spring

  def create_pins(self, neuron_locations: list[np.ndarray]):
    """
    Adds three pins, one for each of the 
    """
    pin = SVGMobject('./pin.svg', height=0.6, width=0.6, z_index=26, stroke_color=BLACK).apply_matrix([[-1, 0], [0, 1]])
    pin.shift(UR * 0.1)

    pin0 = pin.copy().shift(neuron_locations[0])
    pin1 = pin.copy().shift(neuron_locations[3])
    pin2 = pin.copy().shift(neuron_locations[2])
    c0 = Create(pin0)
    c1 = Create(pin1)
    c2 = Create(pin2)

    # It is kinda stupid that we have to reset the z-index, but it breaks if we don't
    pin0.set_z_index(26)
    pin1.set_z_index(26)
    pin2.set_z_index(26)

    self.play(AnimationGroup(c0, c1, c2))

  def inference(self, neurons: list[Circle], ghost1: Circle, weights: list[Line], spring: SpringMaob):
    ghost0 = neurons[1].copy().set_opacity(0.5)
    ghost2 = neurons[3].copy().set_opacity(0.5)
    self.add(ghost0)
    self.add(ghost2)

    displacement = DOWN * 1.5
    targetColor = interpolate_color(WHITE, INACTIVE_COLOR, 0.5)
    leftNeuronAnim = neurons[1].animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    spring0 = SpringMaob(ghost0.get_center(), 0.1, width=0.2)
    stretchAnim0 = spring0.animate(1.5)

    spring1 = SpringMaob(ghost2.get_center(), 0.1, width=0.2)
    stretchAnim1 = spring1.animate(1.5)

    # Set this spring animation to be pinned at the bottom
    spring.set_pin('bot')
    stretchAnim2 = spring.animate(1.5)

    topWeightAnim = weights[1].animate(rate_func=spring_interp).shift(displacement)
    bottomWeightAnim = weights[2].animate(rate_func=spring_interp).shift(displacement)

    topGhostAnim = ghost1.animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)
    bottomGhostAnim = ghost2.animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    allAnims = [leftNeuronAnim, topWeightAnim, bottomWeightAnim, topGhostAnim, bottomGhostAnim, stretchAnim0, stretchAnim1, stretchAnim2]

    self.play(AnimationGroup(*allAnims), run_time=4)
