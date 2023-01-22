"""
The fourth prototype - this is a single round of inference because the weights update continously
manim render -pqh continous_weights.py spring
"""

from typing import Literal
from manim import *
import numpy as np
import math

POLE_COLOR = '#606060'
BASEPLATE_COLOR = '##0f0f0f'
BASEPLATE_OUTLINE = '#1F1F1F'
POLE_HEIGHT = 3
INACTIVE_COLOR = '#ee7c31'
WEIGHT_COLOR = '#5b9bd5'
FONT = 'Lato'
FONT_SIZE = 26

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

    self.spring = FunctionGraph(lambda t: np.sin(t), color=RED, x_range=[0, 10 * PI], stroke_width=2)
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

  def get_height(self) -> float:
    return self.__height

  def set_pin(self, pin: Literal['top', 'bot']) -> None:
    """
    The pinned end of the spring does not move during animations/setting of height.
    The non-pinned end is allowed to move freely.
    """
    self.__pin = pin

  def __move(self, new_height, obj):
    """
    Internal function that abstracts the logic for both `set_height` and `animate`.
    Keeps the pinned location in place while extending the non-pinned location to the appropriate height.
    """
    if (new_height == self.__height):
      return obj

    height = 0.1 if new_height == 0 else new_height

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
    self.speedup = 1
    
    self.train(neuron_locations)

    self.wait()

  def train(self, neuron_locs: list[np.ndarray]):
    neurons, weights = self.create_neurons(neuron_locs, [0, 0, 0, 0])

    self.add_error(neurons[2], neurons[1], weights[1])
    self.create_pins([n.get_center() for n in neurons])

    self.wait()
    self.inference(neurons, weights)

  def construct_baseplates(self) -> list[np.ndarray]:
    """
    Constructs the base rectangles and the scales for each neuron
    Returns: a list of coords for where the neurons go
    """
    base = RoundedRectangle(corner_radius=0.1, sheen_factor=0.2, sheen_direction=[-1, -1, 0], color=BLACK, stroke_color=BASEPLATE_OUTLINE, stroke_width=2, width=1.5, height=2)
    base = base.set_opacity(1).apply_matrix([[1, 1], [0, 2]])
    rect1 = base.copy().shift(1.3 * DOWN + 3 * LEFT)
    rect2 = base.copy().shift(1.3 * DOWN + 0 * LEFT)
    rect3 = base.copy().shift(1.3 * DOWN, 3 * RIGHT)
    self.add(rect1, rect2, rect3)

    pole = Rectangle(color=POLE_COLOR, width=0.1, height=POLE_HEIGHT, stroke_width=0).set_opacity(0.6).shift(rect1.get_center()).shift(UP * POLE_HEIGHT / 2)
    dLine1 = DashedLine(rect1.get_center() + 0.5 * LEFT, rect1.get_center() + 0.5 * RIGHT).set_color(POLE_COLOR)
    dLine2 = dLine1.copy().shift(UP * POLE_HEIGHT / 2)
    dLine3 = dLine2.copy().shift(UP * POLE_HEIGHT / 2)

    # scale1 = VGroup(pole, dLine1, dLine2, dLine3)
    # In the most recent revision, the client asked for 
    scale1 = VGroup(pole)
    scale2 = scale1.copy().shift(RIGHT * 3)
    scale3 = scale2.copy().shift(RIGHT * 3.75 + POLE_HEIGHT / 2 * UP)
    scale4 = scale2.copy().shift(RIGHT * 2.25 + POLE_HEIGHT / 2 * DOWN)

    self.add(scale1, scale2, scale3, scale4)

    return [s.get_center() + UP * POLE_HEIGHT / 2 for s in [scale1, scale2, scale3, scale4]]

  def create_neurons(self, neuron_locations: list[np.ndarray], activations: list[float]) -> tuple[list[Circle], list[Line]]:
    """
    Creates the neurons and weights on the top of their posts.
    The activations is for settings a color

    Returns: all of the neurons and all of the weights
    """
    neuron = Circle(radius=0.5, color=WHITE, z_index=20, stroke_width=2, stroke_color=INACTIVE_COLOR).set_opacity(1)
    neuron0 = neuron.copy().shift(neuron_locations[0]).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, activations[0]))
    neuron1 = neuron.copy().shift(neuron_locations[1]).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, activations[1]))
    neuron2 = neuron.copy().shift(neuron_locations[2]).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, activations[2]))
    neuron3 = neuron.copy().shift(neuron_locations[3]).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, activations[3]))
    neurons = [neuron0, neuron1, neuron2, neuron3]

    self.play(AnimationGroup(*[Create(n) for n in neurons], lag_ratio=0.1))

    weight0 = Line(start=neuron0.get_center(), end=neuron1.get_center(), color=WEIGHT_COLOR, stroke_width=2)
    weight1 = Line(start=neuron1.get_center(), end=neuron2.get_center(), color=WEIGHT_COLOR, stroke_width=2)
    weight2 = Line(start=neuron1.get_center(), end=neuron3.get_center(), color=WEIGHT_COLOR, stroke_width=2)

    self.play(AnimationGroup(Create(weight0), Create(VGroup(weight1, weight2), lag_ratio=0), lag_ratio=0.3))

    return neurons, [weight0, weight1, weight2]

  def add_error(self, neuron: Circle, midNeuron: Circle, weight: Line):
    """
    Adds the error by stretching the neuron to reveal the ghost.
    The height refers to how far down we are on the current training index

    Returns: the ghost neuron and the error spring
    """

    neuronAnim = neuron.animate().set_fill_color(INACTIVE_COLOR).shift(DOWN * POLE_HEIGHT)
    weightAnim = weight.animate.put_start_and_end_on(midNeuron.get_center(), neuron.get_center() + DOWN * POLE_HEIGHT)
    self.play(AnimationGroup(neuronAnim, weightAnim, run_time=self.speedup))

  def create_pins(self, neuron_locations: list[np.ndarray]) -> list[SVGMobject]:
    """
    Adds three pins, one for each of the 
    """
    pin = SVGMobject('./pin2.svg', height=0.6, width=0.6, z_index=26).scale(0.5)

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

    self.play(AnimationGroup(c0, c1, c2, run_time=self.speedup))

    return [pin0, pin1, pin2]

  def inference(self, neurons: list[Circle], weights: list[Line]):

    displacement = DOWN * POLE_HEIGHT / 2
    targetColor = interpolate_color(WHITE, INACTIVE_COLOR, 0.5)
    leftNeuronAnim = neurons[1].animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    weightAnim0 = weights[0].animate(rate_func=spring_interp).put_start_and_end_on(neurons[0].get_center(), neurons[1].get_center() + displacement)
    weightAnim1 = weights[1].animate(rate_func=spring_interp).put_start_and_end_on(neurons[1].get_center() + displacement, neurons[2].get_center())
    weightAnim2 = weights[2].animate(rate_func=spring_interp).put_start_and_end_on(neurons[1].get_center() + displacement, neurons[3].get_center())

    self.play(AnimationGroup(leftNeuronAnim, weightAnim0, weightAnim1, weightAnim2), run_time=4 * self.speedup)
