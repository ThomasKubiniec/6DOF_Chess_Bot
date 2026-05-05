import threading
import time
import dearpygui.dearpygui as dpg


# ----------------------------
# Shared State
# ----------------------------
class SharedState:
    def __init__(self):
        self.data = {
            "var1": 0.0,
            "var2": 0.0,
        }
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.data[key] = value

    def get_all(self):
        with self.lock:
            return dict(self.data)


# ----------------------------
# UI Thread
# ----------------------------
def run_ui(state: SharedState):
    def slider_callback(sender, app_data, user_data):
        state.set(user_data, app_data)

    dpg.create_context()

    with dpg.window(label="Control Panel"):
        dpg.add_slider_float(
            label="Variable 1",
            default_value=0.0,
            min_value=0.0,
            max_value=10.0,
            callback=slider_callback,
            user_data="var1",
        )

        dpg.add_slider_float(
            label="Variable 2",
            default_value=0.0,
            min_value=0.0,
            max_value=10.0,
            callback=slider_callback,
            user_data="var2",
        )

    dpg.create_viewport(title="UI Thread", width=300, height=200)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()


# ----------------------------
# Worker Thread
# ----------------------------
def worker_loop(state: SharedState):
    while True:
        values = state.get_all()
        print(f"[WORKER] var1={values['var1']:.2f}, var2={values['var2']:.2f}")
        time.sleep(1)


# ----------------------------
# Main
# ----------------------------
def main():
    state = SharedState()

    # Start worker thread
    threading.Thread(target=worker_loop, args=(state,), daemon=True).start()

    # Run UI in main thread (important!)
    run_ui(state)


if __name__ == "__main__":
    main()