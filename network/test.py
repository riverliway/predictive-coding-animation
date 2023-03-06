from manim import *
import math

class classical(Scene):
  def construct(self):
    self.line()

  def line(self):
    self.play(fading_line(LEFT, RIGHT, RIGHT * 3))

  def lines(self):
    cases = 16
    lines = [Line(ORIGIN, 2 * UP * math.sin(2 * PI / cases * i) + 2 * RIGHT * math.cos(2 * PI / cases * i)) for i in range(cases)]
    for l in lines:
      l.set_color(color=[PINK, YELLOW])

    self.add(*lines)

  def traced_path(self):
    cases = 16

    pulses = [Dot(ORIGIN) for _ in range(cases)]
    trails = [TracedPath(pulse.get_center, dissipating_time=0.25, stroke_color=WHITE, stroke_opacity=[1, 0]) for pulse in pulses]
    self.add(*pulses, *trails)

    anims = [p.animate.shift(2 * UP * math.sin(2 * PI / cases * i) + 2 * RIGHT * math.cos(2 * PI / cases * i)) for (i, p) in enumerate(pulses)]
    self.play(AnimationGroup(*anims))

    self.wait()

def fading_line(x1, y1, delta):
  line = Line(LEFT, RIGHT)
  line.set_opacity(opacity=[1, 0])
  line.set_angle(45 * DEGREES)
  # line.put_start_and_end_on(x1, y1)
  dot = Dot(y1)

  return AnimationGroup(line.animate.shift(delta), dot.animate.shift(delta))
