"""
A different scene that renders the PCN
11:30pt-1pt
manim render -pqh network.py network
"""

from typing import Callable, Literal
from manim import *
import math

INACTIVE_COLOR = '#ee7c31'
WEIGHT_COLOR = '#5b9bd5'
BASEPLATE_COLOR = '##0f0f0f'
BASEPLATE_OUTLINE = '#1F1F1F'
FONT = 'Lato'
FONT_SIZE = 26

class network(Scene):
  def construct(self):
    # Contruct the network
    network = Network([1, 1, 2], self.add, vert_space=4)
    network.set_activations([[1], [0.5], [0, 1]])
    network.disable_error()
    self.add(*network.get_moabs())

    # Create the base plates
    base1 = RoundedRectangle(corner_radius=0.1, sheen_factor=0.2, sheen_direction=[-1, -1, 0], color=BLACK, stroke_color=BASEPLATE_OUTLINE, stroke_width=2, width=1.5, height=2 * 2.6)
    base1.set_opacity(1)
    base2 = base1.copy().shift(LEFT * 3)
    base3 = base1.copy().shift(RIGHT * 3)
    self.add(base1, base2, base3)

    # Create the pins so we can remove them in the animation
    first_pin_locs = [
      (0, 0, 'on'),
      (2, 0, 'on'),
      (2, 1, 'on')
    ]
    self.play(network.pin_neurons(first_pin_locs))
    self.wait()

    # Remove the pins and reset the network
    second_pin_locs = [(x, y, 'off') for (x, y, _) in first_pin_locs]
    reset_activations = network.animate_activations([[1], [1], [1, 1]])
    self.play(AnimationGroup(network.pin_neurons(second_pin_locs), reset_activations))
    self.wait()

    # Forward propagation
    self.play(network.forward())
    self.wait()

    # Move the top right neuron to inactive
    network.set_error_positions([[1], [1], [1, 1]])
    self.play(network.animate_activations([[1], [1], [0, 1]]))
    self.wait()

    # Inference
    ground_activations = [[1], [0.5], [1, 1]]
    self.play(network.animate_inference(ground_activations, 4))
    self.wait()

class Network:
  def __init__(self, layer_dims: list[int], add, horz_space = 3, vert_space = 2):
    self.layer_dims = layer_dims
    self.horz_space = horz_space
    self.vert_space = vert_space
    self.add = add

    # Create neurons in the layer dimensions provided
    self.neurons = [[Neuron() for _ in range(layer_dim)] for layer_dim in layer_dims]

    for (i, layer) in enumerate(self.neurons):
      for (j, neuron) in enumerate(layer):
        # Vertically space neuron appropriately
        neuron.shift(DOWN * j * vert_space)
        # Center layer vertically
        neuron.shift(UP * (len(layer) - 1) * vert_space / 2)
        # Horizontally space neuron
        neuron.shift(RIGHT * i * horz_space)
        # Center layer horizontally
        neuron.shift(LEFT * (len(layer_dims) - 1) * horz_space / 2)

    weight: Callable[[Neuron, Neuron], Line] = lambda ln, rn: Line(start=ln.get_circle().get_center(), end=rn.get_circle().get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12)
    self.weights: list[list[list[Line]]] = [[[weight(ln, rn) for ln in self.neurons[i]] for rn in self.neurons[i + 1]] for i in range(len(self.neurons) - 1)]

  def get_neurons(self):
    return self.neurons

  def get_weights(self):
    return self.weights

  def forward(self):
    """
    Animates forward propagation of every layer sequentially
    """
    anims = []
    for layer in range(len(self.weights)):
      anims.append(self.forward_prop(layer))

    return AnimationGroup(*anims, lag_ratio=0.8)

  def forward_prop(self, layer):
    """
    Animates forward propagation of a single layer
    """
    weights = self.weights[layer]
    anims = []
    for weight in flat(weights):
      pulse = Dot(weight.get_start(), radius=0.03, z_index=13, color=WHITE, fill_opacity=1)
      trail = TracedPath(pulse.get_center, dissipating_time=0.5, z_index=13, stroke_color=WHITE, stroke_width=weight.get_stroke_width(), stroke_opacity=[1, 0])
      self.add(pulse, trail)
      anims.append(pulse.animate.shift(weight.get_end() - weight.get_start()))

    return AnimationGroup(*anims)

  def get_moabs(self):
    neuronMoabs = [n.get_moabs() for n in flat(self.neurons)]
    return flat([neuronMoabs, self.weights])

  def set_activations(self, activations: list[list[float]]):
    """
    Sets every neuron activations without animation
    """
    for (activation_layer, neuron_layer) in zip(activations, self.neurons):
      for (activation, neuron) in zip(activation_layer, neuron_layer):
        neuron.set_activation(activation)

  def animate_activations(self, activations: list[list[float]], rate_func=rate_functions.smooth):
    """
    Animate every neuron activation
    """
    anims = []
    for (activation_layer, neuron_layer) in zip(activations, self.neurons):
      for (activation, neuron) in zip(activation_layer, neuron_layer):
        anims.append(neuron.animate_activation(activation, rate_func))

    return AnimationGroup(*anims)

  def set_error_positions(self, errors: list[list[float]]):
    """
    Sets every error position without animation
    """
    for (error_layer, neuron_layer) in zip(errors, self.neurons):
      for (error, neuron) in zip(error_layer, neuron_layer):
        neuron.set_error_position(error)

  def animate_error_positions(self, errors: list[list[float]], rate_func=rate_functions.smooth):
    """
    Animate every neuron activation
    """
    anims = []
    for (error_layer, neuron_layer) in zip(errors, self.neurons):
      for (error, neuron) in zip(error_layer, neuron_layer):
        anims.append(neuron.animate_error_position(error, rate_func))

    return AnimationGroup(*anims)

  def animate_inference(self, ground_activations: list[list[float]], run_time=1):
    """
    Animates the process of inference where the neurons eventually end up at their ground state
    """

    anims = []

    last_layer_errors = [abs(n.active - a) for (n, a) in zip(self.neurons[-1], ground_activations[-1])]
    average_error = sum(last_layer_errors) / len(last_layer_errors)
    average_error = max(min(average_error, 1), 0)

    for (activation_layer, neuron_layer) in zip(ground_activations[1:-1], self.neurons[1:-1]):
      for (activation, neuron) in zip(activation_layer, neuron_layer):
        anims.append(neuron.animate_activation(activation, spring_interp))

    for neuron in self.neurons[-1]:
      anims.append(neuron.animate_error_position(neuron.errorPos - average_error, spring_interp))

    return AnimationGroup(*anims, run_time=run_time)

  def pin_neurons(self, locations: list[tuple[int, int, Literal['on', 'off']]]):
    """
    Pins multiple neurons, where the locations are listed as tuples.
    The first index of the tuple refers to the layer index and
    the second is the neuron index within that layer.
    """

    anims = []
    for (i, j, direction) in locations:
      neruon = self.neurons[i][j]
      anims.append(neruon.create_pin() if direction == 'on' else neruon.remove_pin())

    return AnimationGroup(*anims)

  def disable_error(self):
    """
    Disables the error glow from all neurons
    """
    for n in flat(self.neurons):
      n.disable_error()

class Neuron:
  def __init__(self):
    # For neuron activity, 1 is high on the pole, white. 0 is low on the pole, orange
    self.active = 1
    self.pin = None
    self.circle = Circle(radius=0.5, color=WHITE, z_index=20, stroke_width=2, stroke_color=INACTIVE_COLOR).set_opacity(1)
    self.errorCircle = ImageMobject('./error.png').set_z_index(8).scale(0.24)
    self.errorPos = 0
    self.errorScale = 1
    self.error_enabled = True

  def create_pin(self):
    """
    If pin does not already exist, create a new one
    """
    if self.pin is None:
      self.pin = SVGMobject('../spring/pin2.svg', height=0.6, width=0.6, z_index=26).scale(0.5)
      self.pin.move_to(self.circle.get_center())

      anim =  FadeIn(self.pin, shift=DOWN * 0.3)
      self.pin.set_z_index(26)
      return anim

  def remove_pin(self):
    """
    Fades out the pin if exists
    """
    if self.pin is not None:
      return FadeOut(self.pin, shift=UP * 0.3)

  def disable_error(self):
    """
    Removes the error glow from being shown at all
    """
    self.error_enabled = False
    self.errorPos = self.active
    self.__update_error(self.errorCircle)

  def set_error_position(self, position):
    """
    Sets the error position (ghost neuron) to the desired spot
    Does not use animation.
    """
    self.error_enabled = True
    self.errorPos = position
    self.__update_error(self.errorCircle)

  def animate_error_position(self, position, rate_func):
    """
    Animates the error position (ghost neuron) to the desired spot
    """
    self.error_enabled = True
    self.errorPos = position
    return self.__update_error(self.errorCircle.animate(rate_func=rate_func))
  
  def set_activation(self, active: float):
    """
    Sets the activation of the neuron without animating it
    active parameter should be between 0 and 1
    """
    self.active = active
    self.__update_activation(self.circle, self.errorCircle)

  def animate_activation(self, active: float, rate_func):
    """
    Sets the activation of the neuron by animating it
    active parameter should be between 0 and 1
    rate_func parameter should be a rate function
    """
    self.active = active
    anims = self.__update_activation(self.circle.animate(rate_func=rate_func), self.errorCircle.animate(rate_func=rate_func))
    return AnimationGroup(*anims)

  def shift(self, amount: np.ndarray, animate=False):
    """
    Shifts the entire neuron the amount provided
    If the animate flag is set, it returns an animation
    """
    anims = []
    anims.append((self.circle.animate if animate else self.circle).shift(amount))
    anims.append((self.errorCircle.animate if animate else self.errorCircle).shift(amount))
    if self.pin is not None:
      anims.append((self.pin.animate if animate else self.pin).shift(amount))

    if animate:
      return AnimationGroup(*anims)

  def __update_activation(self, neuron, errorCircle):
    """
    Updates the activation of the neuron.
    The neuron/errorCircle parameter should either be the circle or the circle's animation for updating
    """
    anims = []
    anims.append(neuron.set(fill_color=self.__active_color(self.active)))
    if self.error_enabled:
      anims.append(self.__update_error(errorCircle))
    return anims

  def __update_error(self, errorCricle):
    """
    Updates the error of a neuron.
    The errorCircle parameter should either be the circle or the circle's animation for updating
    """

    # Since we cannot scale the error to 0 because it would be impossble to scale back to 1,
    # We scale it just so it is invisible behind the neuron
    MIN_SCALE = 0.6

    error = abs(self.errorPos - self.active)
    width = interpolate(MIN_SCALE, 1, self.errorScale)
    scale = 1 + (interpolate(MIN_SCALE, 1, error) - width) / width

    anim = errorCricle.scale(scale)
    self.errorScale = (scale * width - MIN_SCALE) / MIN_SCALE
    return anim

  def __active_color(self, alpha: float) -> color.Color:
    """
    Get the color of a neuron based on how active it is
    """
    return interpolate_color(INACTIVE_COLOR, WHITE, alpha)

  def get_circle(self):
    return self.circle

  def get_moabs(self):
    moabs = [self.circle, self.errorCircle]
    if self.pin is not None:
      moabs.append(self.pin)

    return moabs

def spring_interp (x: float) -> float:
  """
  Interpolates a spring animation
  """
  nx = x - 0.16
  return (pow(2, -10 * nx) * math.sin(50 * nx) + 1.476) * 0.335 + 0.50645

def flat(list_of_lists):
  if len(list_of_lists) == 0:
    return list_of_lists
  if isinstance(list_of_lists[0], list):
    return flat(list_of_lists[0]) + flat(list_of_lists[1:])
  return list_of_lists[:1] + flat(list_of_lists[1:])