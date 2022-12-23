from typing import TypeVar
from manim import *
import numpy as np
import math

POLE_COLOR = '#606060'
INACTIVE_COLOR = '#1e3b69'

class spring(Scene):
  def construct(self):
    neuron_locations = self.construct_baseplates()
    neurons, weights = self.create_neurons(neuron_locations)
    ghost, spring = self.add_error(neurons[2])
    self.create_pins([n.get_center() for n in neurons])

    self.inference(neurons, ghost, weights, spring)
    
    self.wait()

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

  def add_error(self, neuron: Circle) -> tuple[Circle, FunctionGraph]:
    """
    Adds the error by stretching the neuron to reveal the ghost

    Returns: the ghost neuron and the error spring
    """
    ghost = neuron.copy().set_opacity(0.5)
    self.add(ghost)

    spring = FunctionGraph(lambda t: np.sin(t), color=RED, x_range=[0, 10 * PI], stroke_width=5)
    spring.rotate(90 * DEGREES).scale(0.1).move_to(ghost.get_center()).stretch(0.1, 1).shift(DOWN * 0.1)
    stretchAnim = spring.animate.stretch(10, 1).shift(1.5 * DOWN)
    neuronAnim = neuron.animate().set_fill_color(INACTIVE_COLOR).shift(DOWN * 3)
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

  def inference(self, neurons: list[Circle], ghost1: Circle, weights: list[Line], spring: FunctionGraph):
    # Since Manim doesn't allow for rate functions to go outside of [0, 1]
    # We need to have the spring rate function terminate not at 1
    # and subsequently extend all animations so they end at the right distance
    # The 1.19 value was found by brute force
    extender = 1.19
    ghost0 = neurons[1].copy().set_opacity(0.5)
    ghost2 = neurons[3].copy().set_opacity(0.5)
    self.add(ghost0)
    self.add(ghost2)

    leftNeuronMove = DOWN * 1.5
    leftNeuronAnim = neurons[1].animate.shift(leftNeuronMove * extender)

    self.play(AnimationGroup(leftNeuronAnim, rate_func=spring_interp), run_time=2)

    neurons[1].shift(leftNeuronMove * (1 - extender))
    self.add(neurons[1])

def spring_interp (x: float) -> float:
  """
  Interpolates a spring animation
  """
  nx = x - 0.06
  return (pow(2, -10 * nx) * math.sin(30 * nx) + 1.476) * 0.45