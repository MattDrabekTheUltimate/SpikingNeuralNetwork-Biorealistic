import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from vispy import scene
from vispy.scene import visuals

# Load the output spikes
output_sink_data = np.load('data/output_sink_data.npy')

# 2D Visualization using matplotlib
fig, ax = plt.subplots()
ax.set_xlim(0, output_sink_data.shape[1])
ax.set_ylim(0, output_sink_data.shape[0])
line, = ax.plot([], [], 'r-')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    spike_times = np.where(output_sink_data[:, frame])[0]
    line.set_ydata(spike_times)
    return line,

ani = animation.FuncAnimation(fig, update, frames=output_sink_data.shape[1], init_func=init, blit=True, interval=20)
plt.show()

# 3D Visualization using vispy
canvas = scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()

scatter = visuals.Markers()
view.add(scatter)

view.camera = scene.cameras.TurntableCamera(fov=45)

# Generate random positions for neurons
pos = np.random.normal(size=(100, 3), scale=0.2)
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, 0.5), size=5)

def update_vispy(frame):
    spike_times = np.where(output_sink_data[:, frame])[0]
    scatter.set_data(pos[spike_times], edge_color=None, face_color=(1, 0, 0, 0.5), size=10)

timer = canvas.events.timer.connect(lambda event: update_vispy(event))
timer.start(0.1)
canvas.app.run()

