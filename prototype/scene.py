from typing import TypeVar
from manim import *

class pnet(Scene):
  def construct(self):
    all_neurons, all_nodes, all_errors = self.create_neurons()
    self.create_weights(all_nodes, all_errors)

  def create_neurons(self):
    # Create background circles representing the entire neuron
    neuron_pattern = Circle(radius=0.8, color=DARKER_GRAY, fill_opacity=1, z_index=-10)
    left_neurons = [neuron_pattern.copy().shift(4 * LEFT + (i - 1) * 2.5 * UP) for i in range(3)]
    center_neurons = [neuron_pattern.copy().shift((i - 1) * 2.5 * UP) for i in range(3)]
    right_neurons = [neuron_pattern.copy().shift(4 * RIGHT + (i - 1) * 2.5 * UP) for i in range(3)]
    all_neurons = [left_neurons, center_neurons, right_neurons]

    # Create the proper nodes (circles) for each neuron
    node_pattern = Circle(radius=0.3, color=WHITE, fill_opacity=1, fill_color=neuron_pattern.get_color(), z_index=10)
    left_nodes = [node_pattern.copy().shift(neuron.get_center() + 0.35 * UP + 0.35 * LEFT) for neuron in left_neurons]
    center_nodes = [node_pattern.copy().shift(neuron.get_center() + 0.35 * UP + 0.35 * LEFT) for neuron in center_neurons]
    right_nodes = [node_pattern.copy().shift(neuron.get_center() + 0.35 * UP + 0.35 * LEFT) for neuron in right_neurons]
    all_nodes = [left_nodes, center_nodes, right_nodes]

    # Create the error nodes (triangles) for each neuron
    error_pattern = Triangle(color=WHITE, fill_opacity=1, fill_color=neuron_pattern.get_color(), z_index=11).scale(0.35)
    left_errors = [error_pattern.copy().shift(neuron.get_center() + 0.35 * DOWN + 0.35 * RIGHT) for neuron in left_neurons]
    center_errors = [error_pattern.copy().shift(neuron.get_center() + 0.35 * DOWN + 0.35 * RIGHT) for neuron in center_neurons]
    right_errors = [error_pattern.copy().shift(neuron.get_center() + 0.35 * DOWN + 0.35 * RIGHT) for neuron in right_neurons]
    all_errors = [left_errors, center_errors, right_errors]

    # Animate the network getting created
    # animate_neuron = [FadeIn(neuron) for neuron in flat(all_neurons)]
    # animate_node = [Create(node) for node in flat(all_nodes)]
    # animate_error = [Create(node) for node in flat(all_errors)]
    # self.play(AnimationGroup(*animate_neuron, lag_ratio=0.1), AnimationGroup(*animate_node, lag_ratio=0.1), AnimationGroup(*animate_error, lag_ratio=0.1))

    # If we don't want to animate the creation of the network, just add them
    self.add(*flat(all_neurons), *flat(all_nodes), *flat(all_errors))

    return all_neurons, all_nodes, all_errors

  def create_weights(self, all_nodes: list[list[Circle]], all_errors: list[list[Triangle]]):
    # Create the lines between the left nodes and left errors
    leftInnerWeights = [Line(start=node.get_center(), end=error.get_center(), color=YELLOW, z_index=5) for (node, error) in zip(all_nodes[0], all_errors[0])]

    # Create the lines between the left errors and the center nodes
    leftOuterWeights = [Line(start=error.get_center(), end=node.get_center(), color=YELLOW, z_index=5) for error in all_errors[0] for node in all_nodes[1]]

    # Create the lines between the center nodes and center errors
    centerInnerWeights = [Line(start=node.get_center(), end=error.get_center(), color=YELLOW, z_index=5) for (node, error) in zip(all_nodes[1], all_errors[1])]

    # Create the lines between the center errors and the right nodes
    rightOuterWeights = [Line(start=error.get_center(), end=node.get_center(), color=YELLOW, z_index=5) for error in all_errors[1] for node in all_nodes[2]]

    # Create the lines between the center nodes and center errors
    rightInnerWeights = [Line(start=node.get_center(), end=error.get_center(), color=YELLOW, z_index=5) for (node, error) in zip(all_nodes[2], all_errors[2])]

    self.play(AnimationGroup(*[Create(w) for w in leftInnerWeights]))
    self.play(AnimationGroup(*[Create(w) for w in leftOuterWeights]))
    self.play(AnimationGroup(*[Create(w) for w in centerInnerWeights]))
    self.play(AnimationGroup(*[Create(w) for w in rightOuterWeights]))
    self.play(AnimationGroup(*[Create(w) for w in rightInnerWeights]))
    
T = TypeVar('T')
def flat(arrays: list[list[T]]) -> list[T]:
  return [item for sublist in arrays for item in sublist]