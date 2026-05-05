import threading
import time
import random
import dearpygui.dearpygui as dpg


# ----------------------------
# Shared State
# ----------------------------
class SharedState:
    def __init__(self):
        self.data = {
            "rand_bool": False,
            "my_val": 0.0,
        }
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get(self, key):
        with self.lock:
            return self.data[key]


# ----------------------------
# Controlled Object
# ----------------------------
class RandomVar:
    def __init__(self, state: SharedState):
        self.state = state
        self.rand_bool = False
        self.my_val = 0.0

    def randomize_my_val(self, rand_bool):
        if rand_bool:
            # generate new value
            self.my_val = random.uniform(0, 10)

            # write back to shared state
            self.state.set("my_val", self.my_val)

            # reset trigger
            self.state.set("rand_bool", False)


# ----------------------------
# Worker Loop
# ----------------------------
def worker_loop(state: SharedState, obj: RandomVar):
    while True:
        # read trigger flag
        rand_flag = state.get("rand_bool")

        # run object logic
        obj.randomize_my_val(rand_flag)

        # push value back to UI slider
        val = state.get("my_val")
        dpg.set_value("my_slider", val)

        print(f"[WORKER] my_val = {val:.3f}")

        time.sleep(0.05)


# ----------------------------
# UI
# ----------------------------
def run_ui(state: SharedState):

    def slider_callback(sender, app_data):
        state.set("my_val", app_data)

    def button_callback():
        state.set("rand_bool", True)

    dpg.create_context()

    with dpg.window(label="RandomVar Test"):
        dpg.add_slider_float(
            label="My Value",
            tag="my_slider",
            min_value=0,
            max_value=10,
            default_value=0,
            callback=slider_callback,
        )

        dpg.add_button(label="Randomize", callback=button_callback)

    dpg.create_viewport(title="UI", width=300, height=150)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    # start worker thread AFTER UI exists (important for dpg.set_value)
    obj = RandomVar(state)
    threading.Thread(target=worker_loop, args=(state, obj), daemon=True).start()

    dpg.start_dearpygui()
    dpg.destroy_context()


# ----------------------------
# Main
# ----------------------------
def main():
    state = SharedState()
    run_ui(state)


if __name__ == "__main__":
    main()