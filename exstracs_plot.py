import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from exstracs_classifier import *
from exstracs_tree import *
import math


def plotPopulation(population, exploreIter):

  """
  N = 50
  x = np.random.rand(N)
  y = np.random.rand(N)
  colors = np.random.rand(N)
  area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

  plt.scatter(x, y, s=area, c=colors, alpha=0.5)
  plt.show()

  for i in range(0, 5):
    plt.clear()
    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses

    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.plot()
    plt.show()

  """
  accuracy = []
  coverage = []
  colors = []
  area = []
  for cl in population.popSet:
    if cl.isTree:
      accuracy.append(cl.accuracyComponent)
      coverage.append(cl.coverDiff)
      colors.append('red')
      value = np.pi * (math.log1p((exploreIter - cl.initTimeStamp)/ exploreIter) * 5)**2
      area.append(value)

  max_cover = 0
  for cov in coverage:
    if cov > max_cover:
      max_cover = cov

  xlist = [cov/max_cover for cov in coverage]



  x = np.array(xlist)
  y = np.array(accuracy)
  color_array = np.array(colors)
  area_array = np.array(area)

  plt.scatter(x, y, s=area_array, c=color_array, alpha = 0.5)
  plt.plot()
  plt.show()






"""
Rain simulation

Simulates rain drops on a surface by animating the scale and opacity
of 50 scatter points.

Author: Nicolas P. Rougier


# Create new Figure and an Axes which fills it.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_axes([0, 0, 1, 1], frameon=False)
ax.set_xlim(0, 1), ax.set_xticks([0.1])
ax.set_ylim(0, 1), ax.set_yticks([0.1])

# Create rain data
n_drops = 50
rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
                                      ('size',     float, 1),
                                      ('growth',   float, 1),
                                      ('color',    float, 4)])

# Initialize the raindrops in random positions and with
# random growth rates.
rain_drops['position'] = np.random.uniform(0, 1, (n_drops, 2))
rain_drops['growth'] = np.random.uniform(50, 200, n_drops)

# Construct the scatter which we will update during animation
# as the raindrops develop.
scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
                  s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
                  facecolors='none')


def update(frame_number):

    # Make all colors more transparent as time progresses.
    rain_drops['color'][:, 3] -= 1.0/len(rain_drops)
    rain_drops['color'][:, 3] = np.clip(rain_drops['color'][:, 3], 0, 1)

    # Make all circles bigger.
    rain_drops['size'] += rain_drops['growth']

    # Pick a new position for oldest rain drop, resetting its size,
    # color and growth factor.
    rain_drops['position'][current_index] = np.random.uniform(0, 1, 2)
    rain_drops['size'][current_index] = 5
    rain_drops['color'][current_index] = (0, 0, 0, 1)
    rain_drops['growth'][current_index] = np.random.uniform(50, 200)

    # Update the scatter collection, with the new colors, sizes and positions.
    scat.set_edgecolors(rain_drops['color'])
    scat.set_sizes(rain_drops['size'])
    scat.set_offsets(rain_drops['position'])


# Construct the animation, using the update function as the animation
# director.

animation = FuncAnimation(fig, update, interval=10)
plt.show()

"""

