"""
The fourth prototype - this is a single round of inference because the weights update continously
manim render -pqh continous_transition.py spring
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
    
    neurons, weights = self.train(neuron_locations)

    self.wait()

    self.transition(neurons, weights)

  def train(self, neuron_locs: list[np.ndarray]):
    neurons, weights = self.create_neurons(neuron_locs, [0, 0, 0, 0])

    self.add_error(neurons[2], neurons[1], weights[1])
    self.create_pins([n.get_center() for n in neurons])

    self.wait()
    self.inference(neurons, weights)

    return neurons, weights

  def construct_baseplates(self) -> list[np.ndarray]:
    """
    Constructs the base rectangles and the scales for each neuron
    Returns: a list of coords for where the neurons go
    """
    base = RoundedRectangle(corner_radius=0.1, sheen_factor=0.2, sheen_direction=[-1, -1, 0], color=BLACK, stroke_color=BASEPLATE_OUTLINE, stroke_width=2, width=1.5, height=2)
    base = base.set_opacity(1).apply_matrix([[1, 1], [0, 2]])
    global rect1
    rect1 = base.copy().shift(1.3 * DOWN + 3 * LEFT)
    global rect2
    rect2 = base.copy().shift(1.3 * DOWN + 0 * LEFT)
    global rect3
    rect3 = base.copy().shift(1.3 * DOWN, 3 * RIGHT)
    self.add(rect1, rect2, rect3)

    global pole1
    pole1 = Rectangle(color=POLE_COLOR, width=0.1, height=POLE_HEIGHT, stroke_width=0).set_opacity(0.6).shift(rect1.get_center()).shift(UP * POLE_HEIGHT / 2)
    pole1.set_z_index(6)
    global pole2
    pole2 = pole1.copy().shift(RIGHT * 3)
    global pole3
    pole3 = pole2.copy().shift(RIGHT * 3.75 + POLE_HEIGHT / 2 * UP)
    global pole4
    pole4 = pole2.copy().shift(RIGHT * 2.25 + POLE_HEIGHT / 2 * DOWN)

    self.add(pole1, pole2, pole3, pole4)

    return [s.get_center() + UP * POLE_HEIGHT / 2 for s in [pole1, pole2, pole3, pole4]]

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

    weight0 = Line(start=neuron0.get_center(), end=neuron1.get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12)
    weight1 = Line(start=neuron1.get_center(), end=neuron2.get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12)
    weight2 = Line(start=neuron1.get_center(), end=neuron3.get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12)

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

    global pin0
    pin0 = pin.copy().shift(neuron_locations[0])
    global pin1
    pin1 = pin.copy().shift(neuron_locations[3])
    global pin2
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

  def transition(self, neurons: list[Circle], weights: list[Line]):
    """
    Transitions to the top-down view set up for the network animation scene
    """
    targetPole1 = Rectangle(color=pole1.get_color(), height=0.1, width=0.1, stroke_width=0).set_opacity(0.6).shift(LEFT * 3)
    targetPole2 = targetPole1.copy().shift(RIGHT * 3)
    targetPole3 = targetPole2.copy().shift(RIGHT * 3 + UP * 2)
    targetPole4 = targetPole3.copy().shift(DOWN * 4)
    poleAnim1 = Transform(pole1, targetPole1)
    poleAnim2 = Transform(pole2, targetPole2)
    poleAnim3 = Transform(pole3, targetPole3)
    poleAnim4 = Transform(pole4, targetPole4)

    neuronAnim1 = neurons[0].animate.move_to(targetPole1.get_center())
    neuronAnim2 = neurons[1].animate.move_to(targetPole2.get_center())
    neuronAnim3 = neurons[2].animate.move_to(targetPole3.get_center())
    neuronAnim4 = neurons[3].animate.move_to(targetPole4.get_center())

    pinAnim1 = pin0.animate.move_to(targetPole1.get_center())
    pinAnim2 = pin1.animate.move_to(targetPole4.get_center())
    pinAnim3 = pin2.animate.move_to(targetPole3.get_center())

    baseAnim1 = rect1.animate.apply_matrix([[1, -0.5], [0, 0.5]]).move_to(targetPole1.get_center()).stretch(2.6, 1)
    baseAnim2 = rect2.animate.apply_matrix([[1, -0.5], [0, 0.5]]).move_to(targetPole2.get_center()).stretch(2.6, 1)
    baseAnim3 = rect3.animate.apply_matrix([[1, -0.5], [0, 0.5]]).move_to(targetPole2.get_center() + RIGHT * 3).stretch(2.6, 1)

    weightAnim1 = weights[0].animate.put_start_and_end_on(targetPole1.get_center(), targetPole2.get_center())
    weightAnim2 = weights[1].animate.put_start_and_end_on(targetPole2.get_center(), targetPole3.get_center())
    weightAnim3 = weights[2].animate.put_start_and_end_on(targetPole2.get_center(), targetPole4.get_center())

    anims = [
      poleAnim1, poleAnim2, poleAnim3, poleAnim4,
      neuronAnim1, neuronAnim2, neuronAnim3, neuronAnim4,
      pinAnim1, pinAnim2, pinAnim3,
      baseAnim1, baseAnim2, baseAnim3,
      weightAnim1, weightAnim2, weightAnim3
    ]
    self.play(AnimationGroup(*anims), run_time=2)

    newNeuron1 = neurons[0].copy().shift(UP * 2)
    newNeuron2 = neurons[0].copy().shift(DOWN * 2)
    newNeuron3 = newNeuron1.copy().shift(RIGHT * 3)
    newNeuron4 = newNeuron2.copy().shift(RIGHT * 3)
    newNeuron5 = newNeuron3.copy().shift(RIGHT * 3 + DOWN * 2)

    newNeuron1.set(fill_color=active_color(0.75))
    newNeuron2.set(fill_color=active_color(0.25))
    newNeuron5.set(fill_color=active_color(0.5))

    newNeurons = [newNeuron1, newNeuron2, newNeuron3, newNeuron4, newNeuron5]

    layer1 = [newNeuron1, neurons[0], newNeuron2]
    layer2 = [newNeuron3, neurons[1], newNeuron4]
    layer3 = [neurons[2], newNeuron5, neurons[3]]
    newWeights1 = [Line(start=ln.get_center(), end=rn.get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12) for ln in layer1 for rn in layer2]
    newWeights2 = [Line(start=ln.get_center(), end=rn.get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12) for ln in layer2 for rn in layer3]

    neuronAnims = [Create(n) for n in newNeurons]
    self.play(AnimationGroup(*neuronAnims, FadeOut(rect1), FadeOut(rect2), FadeOut(rect3)))

    weightAnims1 = [Create(w) for w in newWeights1]
    weightAnims2 = [Create(w) for w in newWeights2]
    weightGroup1 = AnimationGroup(*weightAnims1, rate_func=rate_functions.ease_in_quad)
    weightGroup2 = AnimationGroup(*weightAnims2, rate_func=rate_functions.ease_out_quad)
    self.play(AnimationGroup(weightGroup1, weightGroup2, lag_ratio=0.8, run_time=2))

pole1 = None
pole2 = None
pole3 = None
pole4 = None
rect1 = None
rect2 = None
rect3 = None
pin0 = None
pin1 = None
pin2 = None

def active_color(alpha: float):
  """
  Get the color of a neuron based on how active it is
  """
  return interpolate_color(WHITE, INACTIVE_COLOR, alpha)