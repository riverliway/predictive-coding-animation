from typing import TypeVar
import random
from manim import *

class spring3d(ThreeDScene):
  def construct(self):
    random.seed(8936)

    self.set_camera_orientation(phi=65 * DEGREES, theta=15 * DEGREES)
    # [1, 0, 0] = towards camera
    # [0, 1, 0] = right
    # [0, 0, 1] = up
    self.construct_baseplates()
    cube = Cube(side_length=0.5, fill_opacity=1, fill_color=DARK_BLUE)
    axes = ThreeDAxes()

    ball = Sphere([0, 1, 0], resolution=20)
    ball.set_sheen(0, UP)
    ball.set_color(DARK_BLUE)

    self.add(cube, axes)

    self.create_neurons()
    
    self.wait()

  def construct_baseplates(self):
        # Add lines for each neuron to be skewered with
    line = Line3D(start=[0, -2, -2], end=[0, -2, 2], thickness=0.02, color=GRAY)
    self.add(line)
    self.add(line.copy().shift([0, 2, 0]))
    self.add(line.copy().shift([1, 4, 0]))
    self.add(line.copy().shift([-1, 4, 0]))

    line = Prism(dimensions=[0.05, 0.05, 4], fill_opacity=1, fill_color=GRAY).shift([0, -2, 0])
    self.add(line)

    # Add plates at the bottom of the scene
    rect = RoundedRectangle(0.5, color=DARK_GRAY, height=1.5, width=6).shift([0, -2, -2]).set_opacity(1).set_z_index(0)
    self.add(rect)
    self.add(rect.copy().shift([0, 2, 0]))
    self.add(rect.copy().shift([0, 4, 0]))

    self.bring_to_back(line)

  def create_neurons(self):
    neuron1 = Cube(side_length=0.5, fill_opacity=1, fill_color=DARK_BLUE).shift([0, -2, 0])
    neuron2 = neuron1.copy().shift([0, 2, 0])
    neuron3 = neuron1.copy().shift([1, 4, 0])
    neuron4 = neuron1.copy().shift([-1, 4, 0])

    self.play(Create(neuron1), Create(neuron2), Create(neuron3), Create(neuron4))

def neuron_color(activity: float) -> color.Color:
  return interpolate_color(BLUE_E, WHITE, activity)