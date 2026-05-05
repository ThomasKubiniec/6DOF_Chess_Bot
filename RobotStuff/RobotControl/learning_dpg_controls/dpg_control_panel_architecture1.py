import time
import threading
import dearpygui.dearpygui as dpg


# ----------------------------
# Shared State
# ----------------------------
class State:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get_all(self):
        with self.lock:
            return dict(self.data)


# ----------------------------
# UI Generator with Groups
# ----------------------------
class LiveUI:
    def __init__(self, state, config):
        self.state = state
        self.config = config

    def build(self):
        with dpg.window(label="Control Panel", width=400, height=600):
            for group in self.config:
                self._create_group(group)

    def _create_group(self, group):
        label = group["group"]
        items = group["items"]

        with dpg.collapsing_header(label=label, default_open=True):
            for item in items:
                self._create_widget(item)

    def _create_widget(self, item):
        name = item["name"]
        default = item.get("default", 0)
        self.state.set(name, default)

        def callback(sender, app_data, user_data):
            self.state.set(user_data, app_data)

        if item["type"] == "float":
            dpg.add_slider_float(
                label=name,
                default_value=default,
                min_value=item.get("min", 0.0),
                max_value=item.get("max", 1.0),
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "int":
            dpg.add_slider_int(
                label=name,
                default_value=default,
                min_value=item.get("min", 0),
                max_value=item.get("max", 100),
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "bool":
            dpg.add_checkbox(
                label=name,
                default_value=default,
                callback=callback,
                user_data=name,
            )

        elif item["type"] == "button":
            def button_cb():
                print(f"[BUTTON PRESSED] {name}")

            dpg.add_button(label=name, callback=button_cb)


# ----------------------------
# Background Print Loop
# ----------------------------
def print_loop(state):
    while True:
        data = state.get_all()
        print("\n--- STATE ---")
        for k, v in data.items():
            print(f"{k}: {v}")
        time.sleep(1)


# ----------------------------
# Example Config (Grouped)
# ----------------------------
config = [
    {
        "group": "Loss Weights",
        "items": [
            {"name": "position_weight", "type": "float", "default": 1.0, "min": 0, "max": 10},
            {"name": "orientation_weight", "type": "float", "default": 0.5, "min": 0, "max": 10},
            {"name": "collision_weight", "type": "float", "default": 2.0, "min": 0, "max": 20},
        ],
    },
    {
        "group": "Solver Settings",
        "items": [
            {"name": "iterations", "type": "int", "default": 100, "min": 1, "max": 1000},
            {"name": "learning_rate", "type": "float", "default": 0.01, "min": 0.0001, "max": 0.1},
        ],
    },
    {
        "group": "Controls",
        "items": [
            {"name": "solve", "type": "button"},
            {"name": "randomize_target", "type": "button"},
        ],
    },
]


# ----------------------------
# Main
# ----------------------------
def main():
    state = State()

    dpg.create_context()

    ui = LiveUI(state, config)
    ui.build()

    dpg.create_viewport(title="IK Tuning UI", width=420, height=640)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    threading.Thread(target=print_loop, args=(state,), daemon=True).start()

    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()