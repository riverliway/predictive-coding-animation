"""
The third prototype - exactly the same as the second but orange
manim render -pqh fast.py spring
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
    self.label = None
    self.speedup = 1
    neurons = None
    weights = None
    ghosts = None
    springs = None

    for i in range(6):
      self.speedup -= 0.1
      self.step = i

      if i == 0:
        neurons, weights, ghosts, springs = self.train(i, neuron_locations, neurons, weights)
      else:
        self.speedup = 0.2
        self.train_fast(neurons, weights, ghosts, springs)

    self.wait()

  def train(self, step: int, neuron_locs: list[np.ndarray], neurons, weights) -> tuple[list[Circle], list[Line]]:
    movement = POLE_HEIGHT / 12
    neuron_locations = [loc for loc in neuron_locs]
    first_offset = step * movement
    second_offset = step * movement * 2
    neuron_locations[1] += first_offset * DOWN
    neuron_locations[2] += second_offset * DOWN

    if neurons is None:
      neurons, weights = self.create_neurons(neuron_locations, [0, first_offset / POLE_HEIGHT, second_offset / POLE_HEIGHT, 0])

    ghost, spring = self.add_error(neurons[2], step * movement * 2)

    if self.step == 0:
      self.wait()

    self.create_pins([n.get_center() for n in neurons])

    if self.step == 0:
      self.wait()
    ghost_neurons, springs = self.inference(neurons, ghost, weights, spring, first_offset)
    
    if self.step == 0:
      self.wait()

    self.update_weights(weights, movement, ghost_neurons, springs)

    return neurons, weights, ghost_neurons, springs

  def train_fast(self, neurons, weights, ghosts, springs):
    """
    Similar to the other training method, but skips a lot of the animations to make it faster
    """
    displacement = ghosts[0].get_center() - neurons[1].get_center()
    leftNeuronAnim = neurons[1].animate.shift(displacement).set(fill_color=ghosts[0].get_color())

    springs[0].set_pin('top')
    leftSpringAnim = springs[0].animate(0.1, rate_func=rate_functions.smooth)

    springs[2].set_pin('top')
    botSpringAnim = springs[2].animate(0.1, rate_func=rate_functions.smooth)

    springs[1].set_pin('bot')
    topSpringAnim = springs[1].animate(springs[1].get_height() + displacement[1], rate_func=rate_functions.smooth)

    topWeightAnim = weights[1].animate.shift(displacement)
    botWeightAnim = weights[2].animate.shift(displacement)

    botGhostAnim = ghosts[1].animate.shift(displacement).set(fill_color=active_color(get_activation(ghosts[1].copy().shift(displacement))))
    topGhostAnim = ghosts[2].animate.shift(displacement).set(fill_color=active_color(get_activation(ghosts[2].copy().shift(displacement))))

    anims = [leftNeuronAnim, leftSpringAnim, botSpringAnim, topSpringAnim, topWeightAnim, botWeightAnim, botGhostAnim, topGhostAnim]
    self.play(AnimationGroup(*anims), run_time=0.35)

    self.inference_fast(neurons, weights, ghosts, springs)
    # self.update_weights_fast(POLE_HEIGHT / 12, neurons, weights, ghosts, springs)
    self.update_weights(weights, POLE_HEIGHT / 12, ghosts, springs)

  def inference_fast(self, neurons, weights, ghosts, springs):
    """
    Similar to the other inference method, but skips a lot of animations to make it faster
    """

    displacement = pole2.get_center() - neurons[1].get_center()
    leftNeuronAnim = neurons[1].animate(rate_func=spring_interp).shift(displacement).set(fill_color=active_color(0.5))

    topWeightAnim = weights[1].animate(rate_func=spring_interp).shift(displacement)
    botWeightAnim = weights[2].animate(rate_func=spring_interp).shift(displacement)

    botGhostAnim = ghosts[1].animate(rate_func=spring_interp).shift(displacement).set(fill_color=active_color(get_activation(ghosts[1].copy().shift(displacement))))
    topGhostAnim = ghosts[2].animate(rate_func=spring_interp).shift(displacement).set(fill_color=active_color(get_activation(ghosts[2].copy().shift(displacement))))

    leftSpringAnim = springs[0].animate(-displacement[1])
    botSpringAnim = springs[2].animate(-displacement[1])
    topSpringAnim = springs[1].animate(springs[1].get_height() + displacement[1])

    anims = [leftNeuronAnim, topWeightAnim, botWeightAnim, topGhostAnim, botGhostAnim, leftSpringAnim, botSpringAnim, topSpringAnim]
    self.play(AnimationGroup(*anims), run_time=1.5)

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

    global pole1
    pole1 = Rectangle(color=POLE_COLOR, width=0.1, height=POLE_HEIGHT, stroke_width=0).set_opacity(0.6).shift(rect1.get_center()).shift(UP * POLE_HEIGHT / 2)
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
    neuron0 = neuron.copy().shift(neuron_locations[0]).set(fill_color=active_color(activations[0]))
    neuron1 = neuron.copy().shift(neuron_locations[1]).set(fill_color=active_color(activations[1]))
    neuron2 = neuron.copy().shift(neuron_locations[2]).set(fill_color=active_color(activations[2]))
    neuron3 = neuron.copy().shift(neuron_locations[3]).set(fill_color=active_color(activations[3]))
    neurons = [neuron0, neuron1, neuron2, neuron3]

    self.play(AnimationGroup(*[Create(n) for n in neurons], lag_ratio=0.1))

    weight0 = Line(start=neuron0.get_center(), end=neuron1.get_center(), color=WEIGHT_COLOR, stroke_width=2)
    weight1 = Line(start=neuron1.get_center(), end=neuron2.get_center(), color=WEIGHT_COLOR, stroke_width=2)
    weight2 = Line(start=neuron1.get_center(), end=neuron3.get_center(), color=WEIGHT_COLOR, stroke_width=2)

    self.play(AnimationGroup(Create(weight0), Create(VGroup(weight1, weight2), lag_ratio=0), lag_ratio=0.3))

    return neurons, [weight0, weight1, weight2]

  def add_error(self, neuron: Circle, offset: float) -> tuple[Circle, SpringMaob]:
    """
    Adds the error by stretching the neuron to reveal the ghost.
    The height refers to how far down we are on the current training index

    Returns: the ghost neuron and the error spring
    """
    ghost = neuron.copy().set_opacity(0.5).set_z_index(19)
    self.add(ghost)

    height = POLE_HEIGHT - offset
    spring = SpringMaob(ghost.get_center(), 0.1, width=0.2)
    stretchAnim = spring.animate(height, rate_func=rate_functions.smooth)
    neuronAnim = neuron.animate().set_fill_color(INACTIVE_COLOR).shift(DOWN * height)
    self.play(AnimationGroup(stretchAnim, neuronAnim, run_time=self.speedup))

    return ghost, spring

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

  def inference(self, neurons: list[Circle], ghost1: Circle, weights: list[Line], spring: SpringMaob, offset: float) -> tuple[list[Circle], list[SpringMaob]]:
    ghost0 = neurons[1].copy().set_opacity(0.5).set_z_index(19)
    ghost2 = neurons[3].copy().set_opacity(0.5).set_z_index(19)
    self.add(ghost0)
    self.add(ghost2)

    displacement = DOWN * (POLE_HEIGHT / 2 - offset)
    targetColor = interpolate_color(WHITE, INACTIVE_COLOR, 0.5)
    leftNeuronAnim = neurons[1].animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    spring0 = SpringMaob(ghost0.get_center(), 0.1, width=0.2)
    stretchAnim0 = spring0.animate(POLE_HEIGHT / 2 - offset)

    spring1 = SpringMaob(ghost2.get_center(), 0.1, width=0.2)
    stretchAnim1 = spring1.animate(POLE_HEIGHT / 2 - offset)

    # Set this spring animation to be pinned at the bottom
    spring.set_pin('bot')
    stretchAnim2 = spring.animate(POLE_HEIGHT / 2 - offset)

    topWeightAnim = weights[1].animate(rate_func=spring_interp).shift(displacement)
    bottomWeightAnim = weights[2].animate(rate_func=spring_interp).shift(displacement)

    topGhostAnim = ghost1.animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)
    bottomGhostAnim = ghost2.animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    allAnims = [leftNeuronAnim, topWeightAnim, bottomWeightAnim, topGhostAnim, bottomGhostAnim, stretchAnim0, stretchAnim1, stretchAnim2]

    self.play(AnimationGroup(*allAnims), run_time=4 * self.speedup)

    return [ghost0, ghost1, ghost2], [spring0, spring, spring1]

  def update_weights(self, weights: list[Line], dw: float, ghost_neurons: list[Circle], springs: list[SpringMaob]) -> list[Line]:
    """
    Moves the weights to minimize the error
    """

    ghost_weights = [w.copy().set_opacity(0.4) for w in weights]
    self.add(*ghost_weights)

    dws = [dw, dw, -dw]
    move_weight_anims = [w.animate.put_start_and_end_on(w.get_start(), w.get_end() + DOWN * dws[i]) for (i, w) in enumerate(weights)]

    move_ghost_anims = [gn.animate.shift(DOWN * dws[i]).set(fill_color=active_color(get_activation(gn.copy().shift(DOWN * dws[i])))) for (i, gn) in enumerate(ghost_neurons)]

    springs[0].set_pin('bot')
    springs[1].set_pin('bot')
    springs[2].set_pin('top')
    spring_anims = [s.animate(s.get_height() - dw, rate_func=rate_functions.smooth) for s in springs]

    self.play(AnimationGroup(*move_weight_anims, *move_ghost_anims, *spring_anims), run_time=self.speedup)
    self.play(AnimationGroup(*[FadeOut(gw) for gw in ghost_weights]), run_time=self.speedup)

def active_color(alpha: float):
  """
  Get the color of a neuron based on how active it is
  """
  return interpolate_color(WHITE, INACTIVE_COLOR, alpha)

pole1 = None
pole2 = None
pole3 = None
pole4 = None

def get_activation(neuron: Circle) -> float:
  """
  Checks the activation of a neuron based on where it is on the pole
  """
  pole = None
  if (neuron.get_center()[0] == -3):
    pole = pole1
  elif (neuron.get_center()[0] == 0):
    pole = pole2
  elif (neuron.get_center()[0] == 3.75):
    pole = pole3
  elif (neuron.get_center()[0] == 2.25):
    pole = pole4
  else:
    raise Exception('Not valid neuron location')

  bot = pole.get_center()[1] - POLE_HEIGHT / 2
  top = pole.get_center()[1] + POLE_HEIGHT / 2
  point = neuron.get_center()[1]

  if point >= top:
    return 0
  elif point <= bot:
    return 1

  return 1 - (point - bot) / (top - bot)