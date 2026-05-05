import threading
import time
import dearpygui.dearpygui as dpg

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection


# ----------------------------
# Shared State
# ----------------------------
class SharedState:
    def __init__(self):
        self.data = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
        }
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get_all(self):
        with self.lock:
            return dict(self.data)


# ----------------------------
# UI (Main Thread)
# ----------------------------
def run_ui(state: SharedState):
    def slider_callback(sender, app_data, user_data):
        state.set(user_data, app_data)

    dpg.create_context()

    with dpg.window(label="3D Control Panel"):
        dpg.add_slider_float(label="X", min_value=-10, max_value=10,
                             default_value=0, callback=slider_callback, user_data="x")

        dpg.add_slider_float(label="Y", min_value=-10, max_value=10,
                             default_value=0, callback=slider_callback, user_data="y")

        dpg.add_slider_float(label="Z", min_value=-10, max_value=10,
                             default_value=0, callback=slider_callback, user_data="z")

    dpg.create_viewport(title="UI", width=300, height=200)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# ----------------------------
# Matplotlib Worker Thread
# ----------------------------
def plot_loop(state: SharedState):
    plt.ion()  # interactive mode

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # initial point
    point, = ax.plot([0], [0], [0], marker='o')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    while True:
        data = state.get_all()

        x, y, z = data["x"], data["y"], data["z"]

        # update point data
        point.set_data([x], [y])
        point.set_3d_properties([z])

        plt.draw()
        plt.pause(0.01)  # allows GUI to update

        time.sleep(0.02)


# ----------------------------
# Main
# ----------------------------
def main():
    state = SharedState()

    # Start plotting thread
    threading.Thread(target=plot_loop, args=(state,), daemon=True).start()

    # Run UI in main thread
    run_ui(state)


if __name__ == "__main__":
    main()