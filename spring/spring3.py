"""
The third prototype - exactly the same as the second but orange
manim render -pqh spring3.py spring
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

    for i in range(6):
      self.speedup -= 0.1
      self.step = i

      neurons, weights = self.train(i, neuron_locations, neurons, weights)

    self.wait()

  def train(self, step: int, neuron_locs: list[np.ndarray], neurons, weights) -> tuple[list[Circle], list[Line]]:
    movement = POLE_HEIGHT / 12
    neuron_locations = [loc for loc in neuron_locs]
    first_offset = step * movement
    second_offset = step * movement * 2
    neuron_locations[1] += first_offset * DOWN
    neuron_locations[2] += second_offset * DOWN

    if neurons is None:
      self.change_label('PREDICTION')
      neurons, weights = self.create_neurons(neuron_locations, [0, first_offset / POLE_HEIGHT, second_offset / POLE_HEIGHT, 0])

    self.change_label('CALCULATE ERROR')
    ghost, spring = self.add_error(neurons[2], step * movement * 2)

    if self.step == 0:
      self.wait()

    pins = self.create_pins([n.get_center() for n in neurons])

    self.change_label('INFERENCE')
    if self.step == 0:
      self.wait()
    ghost_neurons, springs = self.inference(neurons, ghost, weights, spring, first_offset)
    
    if self.step == 0:
      self.wait()

    self.change_label('UPDATE WEIGHTS')
    self.update_weights(weights, movement, ghost_neurons, springs)

    # Fade out unnecessary things
    self.change_label('UPDATED PREDICTION')
    pins_fade = [FadeOut(p) for p in pins[1:]]

    self.play(AnimationGroup(*pins_fade, run_time=self.speedup))

    self.cleanup(neurons, ghost_neurons, springs, weights, (step + 1) * movement)

    return neurons, weights

  def change_label(self, new):
    new_label = Text(new, font=FONT, font_size=FONT_SIZE).shift(LEFT * 6.5 + UP * 3)
    new_label.shift(new_label.get_center() - new_label.get_left())

    anims = []
    if self.label is None:
      anims.append(Write(new_label))
    else:
      anims.append(ReplacementTransform(self.label, new_label))
      # self.remove(self.label)

    self.label = new_label

    if new == 'UPDATED PREDICTION':
      new_step_label = Text(str(self.step + 2), font=FONT, font_size=FONT_SIZE).shift(LEFT * 5.4 + UP * 2.5)
      anims.append(ReplacementTransform(self.step_label, new_step_label))
      self.step_label = new_step_label

    if new == 'PREDICTION':
      self.step_label = Text('1', font=FONT, font_size=FONT_SIZE).shift(LEFT * 5.4 + UP * 2.5)
      step = Text('STEP', font=FONT, font_size=FONT_SIZE).shift(LEFT * 6.05 + UP * 2.5)
      anims.append(Write(self.step_label))
      anims.append(Write(step))

    self.play(AnimationGroup(*anims), run_time=self.speedup)

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

    displacement = DOWN * (1.5 - offset)
    targetColor = interpolate_color(WHITE, INACTIVE_COLOR, 0.5)
    leftNeuronAnim = neurons[1].animate(rate_func=spring_interp).set(fill_color=targetColor).shift(displacement)

    spring0 = SpringMaob(ghost0.get_center(), 0.1, width=0.2)
    stretchAnim0 = spring0.animate(1.5 - offset)

    spring1 = SpringMaob(ghost2.get_center(), 0.1, width=0.2)
    stretchAnim1 = spring1.animate(1.5 - offset)

    # Set this spring animation to be pinned at the bottom
    spring.set_pin('bot')
    stretchAnim2 = spring.animate(1.5 - offset)

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

    # TODO: update colors as the ghosts move
    move_ghost_anims = [gn.animate.shift(DOWN * dws[i]) for (i, gn) in enumerate(ghost_neurons)]

    springs[0].set_pin('bot')
    springs[1].set_pin('bot')
    springs[2].set_pin('top')
    spring_anims = [s.animate(s.get_height() - dw, rate_func=rate_functions.smooth) for s in springs]

    self.play(AnimationGroup(*move_weight_anims, *move_ghost_anims, *spring_anims), run_time=self.speedup)
    self.play(AnimationGroup(*[FadeOut(gw) for gw in ghost_weights]), run_time=self.speedup)

  def cleanup(self, neurons: list[Circle], ghosts: list[Circle], springs: list[SpringMaob], weights: list[Line], dw: float):
    """
    Moves neurons to the position for the updated prediction and removed unneeded objects
    """

    displacement = (POLE_HEIGHT / 2 - dw) * UP
    color_interp = dw / POLE_HEIGHT
    mid_neuron_anim = neurons[1].animate.shift(displacement).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, color_interp))
    far_neuron_anim = neurons[2].animate.shift((POLE_HEIGHT - 2 * dw) * UP).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, 2 * color_interp))

    weight_anim1 = weights[1].animate.shift(displacement)
    weight_anim2 = weights[2].animate.shift(displacement)

    ghost_anim1 = ghosts[1].animate.shift(displacement).set(fill_color=interpolate_color(WHITE, INACTIVE_COLOR, 2 * color_interp))
    ghost_anim2 = ghosts[2].animate.shift(displacement).set(fill_color=WHITE)

    springs[0].set_pin('top')
    springs[1].set_pin('top')
    springs[2].set_pin('top')
    spring_anims = [s.animate(0.1, rate_func=rate_functions.smooth) for s in springs]

    # this spring is the only one that is "fixed" to a moving location, so we need an extra shift animation
    spring_anims[1] = spring_anims[1].shift(displacement)

    anims = [mid_neuron_anim, weight_anim1, weight_anim2, ghost_anim1, ghost_anim2, far_neuron_anim]

    self.play(AnimationGroup(*anims, *spring_anims), run_time=self.speedup)

    self.remove(*ghosts, *[s.get_spring() for s in springs])
