"""
A different scene that renders the PCN
manim render -pqh network2.py network
manim render -pqh network2.py classical
"""

from typing import Callable, Literal
from manim import *
import math
import random

random.seed(54356)

INACTIVE_COLOR = '#ee7c31'
WEIGHT_COLOR = '#5b9bd5'
BASEPLATE_COLOR = '##0f0f0f'
BASEPLATE_OUTLINE = '#1F1F1F'
FONT = 'Lato'
FONT_SIZE = 26
    
class Network:
  def __init__(self, layer_dims: list[int], add, horz_space = 3, vert_space = 2, ellipsis = False, max_weight_opacity=1):
    self.layer_dims = layer_dims
    self.horz_space = horz_space
    self.vert_space = vert_space
    self.add = add

    # Create neurons in the layer dimensions provided
    self.neurons = [[Neuron() for _ in range(layer_dim)] for layer_dim in layer_dims]
    self.pulses = []
    self.trails = []

    # Move neurons to their correct location
    for (i, layer) in enumerate(self.calc_neuron_pos(layer_dims, horz_space, vert_space)):
      for (j, pos) in enumerate(layer):
        self.neurons[i][j].shift(pos)

    # If we have an ellipsis, shift the first layer away
    self.ellipsis = []
    if ellipsis:
      layer_half = len(self.neurons[0]) // 2
      halfway = (self.neurons[0][layer_half].get_circle().get_center() + self.neurons[0][layer_half - 1].get_circle().get_center()) / 2
      self.ellipsis = [Dot(halfway + d) for d in [UP * vert_space / 2, ORIGIN, DOWN * vert_space / 2]]
      for (i, n) in enumerate(self.neurons[0]):
        if (i < layer_half):
          n.shift(UP * vert_space)
        else:
          n.shift(DOWN * vert_space)

    weight: Callable[[Neuron, Neuron], Line] = lambda ln, rn: Line(start=ln.get_circle().get_center(), end=rn.get_circle().get_center(), color=WEIGHT_COLOR, stroke_width=2).set_z_index(12).set_opacity(random.uniform(0, max_weight_opacity))
    self.weights: list[list[list[Line]]] = [[[weight(ln, rn) for ln in self.neurons[i]] for rn in self.neurons[i + 1]] for i in range(len(self.neurons) - 1)]

  def calc_neuron_pos(self, layer_dims: list[int], horz_space, vert_space):
    """
    Calculates the positions for each neuron
    """
    poses = [[ORIGIN for _ in range(layer_dim)] for layer_dim in layer_dims]

    for i in range(len(poses)):
      for j in range(len(poses[i])):
        # Vertically space neuron appropriately
        poses[i][j] = DOWN * j * vert_space
        # Center layer vertically
        poses[i][j] += UP * (len(poses[i]) - 1) * vert_space / 2
        # Horizontally space neuron
        poses[i][j] += RIGHT * i * horz_space
        # Center layer horizontally
        poses[i][j] += LEFT * (len(layer_dims) - 1) * horz_space / 2

    return poses

  def get_neurons(self):
    return self.neurons

  def get_weights(self):
    return self.weights
  
  def clear_trails(self, remover):
    # map(remover, self.pulses)
    # map(remover, self.trails)
    for p in self.pulses:
      remover(p)

    for t in self.trails:
      remover(t)

    self.pulses = []
    self.trails = []

  def forward(self):
    """
    Animates forward propagation of every layer sequentially
    """
    anims = []
    for layer in range(len(self.weights)):
      anims.append(self.forward_prop(layer))

    return AnimationGroup(*anims, lag_ratio=0.7)

  def forward_prop(self, layer):
    """
    Animates forward propagation of a single layer
    """
    weights = self.weights[layer]
    anims = []
    for weight in flat(weights):
      anims.append(self.__create_pulse(weight))

    return AnimationGroup(*anims)
  
  def __create_pulse(self, weight: Line, backwards = False):
    """
    Creates the pulse animation for a weight
    """
    width = self.neurons[0][0].get_circle().width
    pulse_width = width / 16
    width -= pulse_width
    trail = Line(LEFT * width / 2, RIGHT * width / 2, stroke_width=weight.get_stroke_width())
    trail.set_opacity(opacity=[weight.get_stroke_opacity(), 0])

    start = weight.get_end() if backwards else weight.get_start()
    end = weight.get_start() if backwards else weight.get_end()

    trail.rotate(math.atan2(end[1] - start[1], end[0] - start[0]))
    trail.shift(start)
    pulse = Dot(trail.get_end(), radius=pulse_width, fill_opacity=weight.get_stroke_opacity())

    self.pulses.append(pulse)
    self.trails.append(trail)

    return AnimationGroup(trail.animate.shift(end - start), pulse.animate.shift(end - start))

  def backward_prop(self, layer):
    """
    Animates backwards propagation of a single layer
    """
    weights = self.weights[layer]
    anims = []
    for weight in flat(weights):
      anims.append(self.__create_pulse(weight, backwards = True))

    return AnimationGroup(*anims)

  def forward_with_activation(self, activations: list[list[float]]):
    """
    Forward propagation, but the activations are also animated
    """
    anims = []
    for i in range(len(self.neurons) - 1):
      anims.append(self.animate_activations_layer(activations[i], i))
      anims.append(self.forward_prop(i))

    anims.append(self.animate_activations_layer(activations[-1], len(activations) - 1))

    return AnimationGroup(*anims, lag_ratio=0.7)

  def backward(self, activations: list[list[float]]):
    """
    Classical backwards propagation, but the activations are also animated
    """

    anims = []
    for i in reversed(range(1, len(self.neurons))):
      anims.append(self.animate_layer_error(activations[i], i))
      anims.append(self.backward_prop(i - 1))

    anims.append(self.animate_layer_error(activations[0], 0))

    return AnimationGroup(*anims, lag_ratio=0.7)
  
  def animate_layer_error(self, error_pos: list[float], index: int):
    """
    Animates the error changing for a single layer
    """
    anims = []
    for (error, neuron) in zip(error_pos, self.neurons[index]):
      anims.append(neuron.animate_error_position(error, rate_functions.smooth))

    return AnimationGroup(*anims)

  def get_moabs(self):
    neuronMoabs = [n.get_moabs() for n in flat(self.neurons)]
    return flat([neuronMoabs, self.weights, self.pulses, self.trails, self.ellipsis])

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
    for (i, activation_layer) in enumerate(activations):
      anims.append(self.animate_activations_layer(activation_layer, i))

    return AnimationGroup(*anims)

  def animate_activations_layer(self, activations: list[float], layer: int, rate_func=rate_functions.smooth):
    """
    Animate every neuron activation in a specified layer index
    """
    anims = []
    for (activation, neuron) in zip(activations, self.neurons[layer]):
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
      err = neuron.active - average_error if neuron.active - average_error > 0 else neuron.active + average_error
      anims.append(neuron.animate_error_position(err, spring_interp))

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
    self.errorCircle = ImageMobject('./error.png').set_z_index(8).scale(0.26)
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
      pin = self.pin
      self.pin = None
      return FadeOut(pin, shift=UP * 0.3)

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

class network(Scene):
  def construct(self):
    network3 = self.animate_221()
    network_big = self.animate_3x3(network3)
    self.animate_big(network_big)

  def animate_221(self):
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
    self.play(network.pin_neurons(first_pin_locs))
    self.wait()

    # Inference
    ground_activations = [[1], [0.5], [1, 1]]
    self.play(network.animate_inference(ground_activations, run_time=4))
    self.wait()

    # Fade out 2-2-1 network and fade in 3x3 network
    old_network_anims = [FadeOut(o) for o in network.get_moabs()]
    base_anims = [FadeOut(b) for b in [base1, base2, base3]]
    network = Network([3, 3, 3], self.add)
    network.disable_error()
    new_network_anims = [FadeIn(o) for o in network.get_moabs()]
    self.play(AnimationGroup(*old_network_anims, *base_anims, *new_network_anims))

    return network

  def animate_3x3(self, network: Network):
    activations = [[0.4, 0.9, 0.1], [0.6, 0.5, 0.9], [0.2, 0.05, 0.8]]
    self.play(network.forward_with_activation(activations))
    self.wait()

    # Pin neurons
    network.set_error_positions(activations)
    self.play(network.animate_activations(activations[:-1] + [[0.9, 0.1, 0.5]]))

    pin_locs = [(x, y, 'on') for x in [0, 2] for y in [0, 1, 2]]
    self.play(network.pin_neurons(pin_locs))

    # Inference
    ground = [activations[0]] + [[0.2, 0.4, 0.1]] + [activations[2]]
    self.play(network.animate_inference(ground, run_time=4))
    self.wait()

    # Fade into new network
    old_fades = [FadeOut(o) for o in network.get_moabs()]
    network_big = Network([7, 5, 5, 3], self.add, vert_space=1.25)
    network_big.disable_error()
    new_moabs = Group(*network_big.get_moabs())
    new_moabs.scale(0.8)
    self.play(AnimationGroup(*old_fades, FadeIn(new_moabs)))

    return network_big

  def animate_big(self, network: Network):
    activations = [[0.5, 0.1, 0.9, 0.2, 1, 0.8, 0.4], [0.7, 1, 0.2, 0.7, 0.3], [0, 0.8, 0.2, 1, 0.4, 0.6], [1, 0, 0.4]]
    self.play(network.forward_with_activation(activations))
    self.wait()

    # Pin neurons
    network.set_error_positions(activations)
    self.play(network.animate_activations(activations[:-1] + [[0, 1, 1]]))

    pin_locs = [(0, y, 'on') for y in range(7)] + [(3, y, 'on') for y in range(3)]
    self.play(network.pin_neurons(pin_locs))

    # Inference
    ground = [activations[0]] + [[0.3, 0, 0.1, 0.2, 0.9], [0.9, 0.6, 0.8, 0, 0.5, 0.2]] + [activations[3]]
    self.play(network.animate_inference(ground, run_time=4))
    self.wait()

    fades = [FadeOut(o) for o in network.get_moabs()]
    self.play(AnimationGroup(*fades))
    self.wait()

class classical(Scene):
  def construct(self):
    dims = [16, 16, 16, 10]
    network = Network(dims, self.add, vert_space=1.25, horz_space=6, ellipsis=True, max_weight_opacity=0.4)
    network.disable_error()
    new_moabs = Group(*network.get_moabs())
    new_moabs.scale(0.25)
    new_moabs.shift(DOWN)
    training_label = label('Training...', 'left').shift(LEFT * 6.5 + UP * 3)

    first_image = ImageMobject('./7.png')
    pixels = first_image.get_pixel_array()
    print(len(pixels))
    pixels = [[pixel[0] for pixel in col] for col in pixels]
    first_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])
    first_image.scale(11).move_to((network.neurons[0][0].get_circle().get_center() + network.neurons[0][-1].get_circle().get_center()) / 2 + LEFT * 2)
    pixel_radius = (first_image.get_left() - first_image.get_right())[0] / len(pixels) / 2
    create_pixel = lambda pixel, x, y: Circle(radius=pixel_radius, stroke_color=WHITE, color=interpolate_color(BLACK, WHITE, pixel / 255), stroke_width=0.5).shift(x * pixel_radius * 2 * LEFT + y * pixel_radius * 2 * UP).set_opacity(1)
    pixels = [[create_pixel(pixel, x, y) for (x, pixel) in enumerate(col)] for (y, col) in enumerate(pixels)]
    pixel_group = Group(*flat(pixels))
    pixel_group.move_to(first_image)

    self.play(FadeIn(new_moabs, first_image, pixel_group))
    self.wait()

    network_first_layer_distance = abs(network.neurons[0][-1].get_circle().get_center()[1] - network.neurons[0][0].get_circle().get_center()[1])
    move_pixel_location = lambda i, len: network.neurons[0][0].get_circle().get_center() + network_first_layer_distance * i / len * DOWN
    pixel_move_anims = [pixel.animate.move_to(move_pixel_location(i, len(flat(pixels)))).set_opacity(0) for (i, pixel) in enumerate(flat(pixels))]

    self.play(AnimationGroup(*pixel_move_anims, lag_ratio=0.01))

    self.wait()

    self.play(first_image.animate.move_to(UP * 3 + 1.5 * LEFT).scale(9 / 11))

    left_brace_moabs = Group(network.neurons[0][0].get_circle(), network.neurons[0][-1].get_circle())
    left_brace = Brace(left_brace_moabs, direction=LEFT, sharpness=1)
    left_label = label('256 Input Pixels', 'right').shift(left_brace.get_center() + 0.3 * LEFT)
    right_brace_moabs = Group(network.neurons[-1][0].get_circle(), network.neurons[-1][-1].get_circle())
    right_brace = Brace(right_brace_moabs, direction=RIGHT, sharpness=1)
    right_label = label('10 Output Categories', 'left').shift(right_brace.get_center() + 0.3 * RIGHT)
    top_brace_moabs = Group(network.neurons[1][0].get_circle(), network.neurons[-2][0].get_circle())
    top_brace = Brace(top_brace_moabs, direction=UP, sharpness=1)
    top_label = label('2 Hidden Layers', 'center').shift(top_brace.get_center() + 0.35 * UP)

    arrow = Arrow(UP * 3 + LEFT, UP * 3 + RIGHT)
    output = label('7', 'center').scale(2).shift(3 * UP + 1.5 * RIGHT)

    self.play(FadeIn(left_brace, right_brace, top_brace, left_label, right_label, top_label, arrow, output))
    self.wait()
    self.play(FadeOut(left_brace, right_brace, top_brace, left_label, right_label, top_label), FadeIn(training_label))

    activations = random_matrix(dims)
    output_vec = [0.13, 0.1, 0.4, 0.01, 0.2, 0.3, 0, 0.95, 0.1, 0.45]
    activations[-1] = [1 - i for i in output_vec]
    activations[0] = [1 for _ in activations[0]]
    self.play(network.forward_with_activation(activations))
    network.clear_trails(self.remove)
    self.wait()

    output_labels = [label(str(i), 'center').shift(n.get_circle().get_center() + RIGHT * 0.5).scale(0.7).set_opacity(0.5) for (i, n) in enumerate(network.neurons[-1])]
    output_labels[7].set_opacity(1)
    self.play(FadeIn(*output_labels))

    network.set_error_positions(activations)
    activations = [activations[0]] + random_matrix(dims[1:-1]) + [[1, 1, 1, 1, 1, 1, 1, 0, 1, 1]]
    self.play(network.backward(activations))
    network.clear_trails(self.remove)
    self.wait()

    # Begin second train
    activations = [[1 for _ in layer] for layer in activations]
    self.play(AnimationGroup(FadeOut(first_image, arrow, output, *output_labels), network.animate_activations(activations)))
    self.play(network.animate_error_positions(activations))

    second_image = ImageMobject('./2.png')
    second_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])
    second_image.scale(9).move_to(UP * 3 + 1.5 * LEFT)

    output = label('2', 'center').scale(2).shift(3 * UP + 1.5 * RIGHT)

    self.play(FadeIn(second_image, arrow, output))

    activations = random_matrix(dims)
    output_vec = [0, 0.95, 0.1, 0.11, 0.4, 0.22, 0.8, 0.12, 0.13, 0.3]
    activations[-1] = [1 - i for i in output_vec]
    activations[0] = [1 for _ in activations[0]]
    self.play(network.forward_with_activation(activations))
    network.clear_trails(self.remove)
    self.wait()

    output_labels = [label(str(i), 'center').shift(n.get_circle().get_center() + RIGHT * 0.5).scale(0.7).set_opacity(0.5) for (i, n) in enumerate(network.neurons[-1])]
    output_labels[2].set_opacity(1)
    self.play(FadeIn(*output_labels))

    self.play(FadeOut(new_moabs, training_label, *output_labels))
    self.wait()

def label (text: str, align: Literal['left', 'center', 'right']) -> Text:
  txt = Text(text, font='Lato', font_size=26)
  if align == 'left':
    txt.shift(txt.get_center() - txt.get_left())
  if align == 'right':
    txt.shift(txt.get_center() - txt.get_right())
  return txt

def random_matrix (lens: list[int]) -> list[list[float]]:
  return [random_array(i) for i in lens]

def random_array (len: int) -> list[float]:
  """
  Returns an array of random numbers between 0 and 1
  """
  return [random.uniform(0, 1) for _ in range(len)]

def spring_interp (x: float) -> float:
  """
  Interpolates a spring animation
  """
  nx = x - 0.16
  return (pow(2, -10 * nx) * math.sin(50 * nx) + 1.476) * 0.335 + 0.50645

def flat(list_of_lists):
  if not isinstance(list_of_lists, list):
    return list_of_lists
  ret = []
  for i in list_of_lists:
    if not isinstance(i, list):
      ret.append(i)
    else:
      ret += flat(i)
  return ret
