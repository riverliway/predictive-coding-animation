from typing import TypeVar
import random
from manim import *

class pnet(Scene):
  def construct(self):
    _, predicted_nodes, all_errors = self.create_neurons()
    weights = self.create_weights(predicted_nodes, all_errors)
    updated_nodes = self.override_label(predicted_nodes)
    self.inference(predicted_nodes, updated_nodes, all_errors, weights)

  def create_neurons(self):
    # Create background circles representing the entire neuron
    neuron_pattern = Circle(radius=0.8, color=DARKER_GRAY, fill_opacity=1, z_index=1)
    neuron_dist = 3 # The distance between layers on the screen
    neurons_per_layer = 3
    neuron_layer_1 = [neuron_pattern.copy().shift(neuron_dist * 1.5 * LEFT + (i - 1) * 2.5 * DOWN) for i in range(neurons_per_layer)]
    neuron_layer_2 = [neuron_pattern.copy().shift(neuron_dist * 0.5 * LEFT + (i - 1) * 2.5 * DOWN) for i in range(neurons_per_layer)]
    neuron_layer_3 = [neuron_pattern.copy().shift(neuron_dist * 0.5 * RIGHT + (i - 1) * 2.5 * DOWN) for i in range(neurons_per_layer)]
    neuron_layer_4 = [neuron_pattern.copy().shift(neuron_dist * 1.5 * RIGHT + (i - 1) * 2.5 * DOWN) for i in range(neurons_per_layer)]
    all_neurons = [neuron_layer_1, neuron_layer_2, neuron_layer_3, neuron_layer_4]

    # Create the proper nodes (circles) for each neuron
    node_pattern = Circle(radius=0.3, color=WHITE, fill_opacity=1, fill_color=neuron_pattern.get_color(), z_index=11)
    node_offset = 0.35 * UP + 0.35 * LEFT
    all_nodes = [[node_pattern.copy().shift(neuron.get_center() + node_offset) for neuron in layer] for layer in all_neurons]

    # Create the error nodes (triangles) for each neuron
    error_pattern = Triangle(color=WHITE, fill_opacity=1, fill_color=neuron_pattern.get_color(), z_index=10).scale(0.35)
    node_offset = 0.35 * DOWN + 0.35 * RIGHT
    all_errors = [[error_pattern.copy().shift(neuron.get_center() + node_offset) for neuron in layer] for layer in all_neurons]

    # Animate the network getting created
    animate_neuron = [FadeIn(neuron) for neuron in flat(all_neurons)]
    animate_node = [Create(node) for node in flat(all_nodes)]
    animate_error = [Create(node) for node in flat(all_errors)]
    self.play(AnimationGroup(*animate_neuron, lag_ratio=0.1), AnimationGroup(*animate_node, lag_ratio=0.1), AnimationGroup(*animate_error, lag_ratio=0.1))

    # If we don't want to animate the creation of the network, just add them
    # self.add(*flat(all_neurons), *flat(all_nodes), *flat(all_errors))

    return all_neurons, all_nodes, all_errors

  def create_weights(self, all_nodes: list[list[Circle]], all_errors: list[list[Triangle]]):
    weight_pattern = lambda: Line(start=LEFT, end=RIGHT, color=YELLOW, z_index=5, stroke_width=random.uniform(1, 10))

    # Produce every connection of weights
    # Even indexes are inner weights between the node and the error
    # Odd indexes are outer weights between the error and the next neuron's node 
    # len(weights) == len(all_nodes) - 1
    weights: list[list[Line]] = []

    for i in range(len(all_nodes)):
      weights.append([weight_pattern().put_start_and_end_on(node.get_center(), error.get_center()) for (node, error) in zip(all_nodes[i], all_errors[i])])
      if i != len(all_nodes) - 1:
        weights.append([weight_pattern().put_start_and_end_on(error.get_center(), node.get_center()) for error in all_errors[i] for node in all_nodes[i + 1]])

    for i in range(len(all_nodes)):
      self.play(AnimationGroup(*[n.animate(run_time=0.5).set_fill_color(rand_gray()) for n in all_nodes[i]]))
      if i == len(all_nodes) - 1:
        self.play(AnimationGroup(*[Create(w, run_time=0.5) for w in weights[-1]], lag_ratio=0.1))
      else:
        self.play(AnimationGroup(*[Create(w, run_time=0.5) for w in flat([weights[i * 2], weights[i * 2 + 1]])], lag_ratio=0.1))

    return weights
  
  def override_label(self, predicted_nodes: list[list[Circle]]) -> list[list[Circle]]:
    # Create label nodes for last layer
    label_nodes = [n.copy().shift(RIGHT).set_fill(color=random.choice([WHITE, BLACK])) for n in predicted_nodes[-1]]

    self.play(AnimationGroup(*[FadeIn(n) for n in label_nodes]))
    self.play(AnimationGroup(*[n.animate.shift(LEFT) for n in label_nodes]))

    # Replace with labels in nodes array for future use in error calculations
    return predicted_nodes[0:-1] + [label_nodes]

  def inference(self, predicted_nodes: list[list[Circle]], updated_nodes: list[list[Circle]], errors: list[list[Triangle]], weights: list[list[Line]]):
    # Create the dots and trails representing the pulse to the error node
    error_pulses = [[Dot(n.get_center(), z_index=7, color=RED, fill_opacity=1) for n in layer] for layer in updated_nodes]
    create_trail = lambda p, w: TracedPath(p.get_center, dissipating_time=0.5, z_index=6, stroke_color=RED, stroke_width=w.get_stroke_width(), stroke_opacity=[1, 0])
    error_trails = [create_trail(p, w) for i in range(len(error_pulses)) for (p, w) in zip(error_pulses[i], weights[i * 2])]
    for (pulse, trail) in zip(flat(error_pulses), error_trails):
      self.add(pulse, trail)

    self.play(AnimationGroup(*[p.animate.shift(e.get_center() - p.get_center()) for (p, e) in zip(flat(error_pulses), flat(errors))]))

    update_error = lambda p, u, e: e.animate.set_fill_color(interpolate_color(BLACK, RED, calc_error(p, u)))
    error_anims = [update_error(p, u, e) for (p, u, e) in zip(flat(predicted_nodes), flat(updated_nodes), flat(errors))]
    self.play(AnimationGroup(*error_anims))


T = TypeVar('T')
def flat(arrays: list[list[T]]) -> list[T]:
  return [item for sublist in arrays for item in sublist]

def rand_gray(original: color.Color | None = None, range: float | None = None):
  """
  Returns a random grayscale color

  If original and range are provided, then the new color will be generated by
  sticking to a range by the original color
  """
  if original is None or range is None:
    return interpolate_color(BLACK, WHITE, random.uniform(0, 1))

  direction = random.randint(0, 1)
  if direction == 0:
    return interpolate_color(BLACK, original, random.uniform(0, 1) * range)

  return interpolate_color(WHITE, original, random.uniform(0, 1) * range)

def calc_error(predicted: Circle, actual: Circle) -> float:
  return abs(gray_value(predicted.get_fill_color()) - gray_value(actual.get_fill_color()))

def gray_value(color: color.Color) -> float:
  return color_to_rgb(color)[0]