import dearpygui.dearpygui as dpg
import numpy as np

_settings = {
    "signal_type": "Sine",
    "frequency": 5.0, "amplitude": 2.5, "offset": 0.0, "noise": 0.1,
    "trigger_level": 1.0, "trigger_edge": "Rising",
    "time_per_div": 0.02, "volts_per_div": 0.5,
    "cursor_v1": 1.0, "cursor_v2": -1.0,
    "cursor_t1_pos": 0.25, "cursor_t2_pos": 0.75,
}

def generate_signal(settings):
    timebase = settings["time_per_div"] * 10
    sample_rate = 50000
    num_points = int(timebase * sample_rate)
    buffer_factor = 3
    t = np.linspace(0, timebase * buffer_factor, num_points * buffer_factor, endpoint=False)
    v = np.zeros_like(t)
    freq, amp, off = settings["frequency"], settings["amplitude"], settings["offset"]
    if settings["signal_type"] == "Sine":
        v = amp * np.sin(2 * np.pi * freq * t)
    elif settings["signal_type"] == "Square":
        v = amp * np.sign(np.sin(2 * np.pi * freq * t))
    elif settings["signal_type"] == "Triangle":
        v = amp * 2 / np.pi * np.arcsin(np.sin(2 * np.pi * freq * t))
    elif settings["signal_type"] == "Sawtooth":
        v = amp * 2 * (t * freq - np.floor(0.5 + t * freq))
    v += off
    if settings["noise"] > 0:
        v += (np.random.rand(len(v)) - 0.5) * 2 * settings["noise"]
    return t, v

def apply_trigger(t, v, settings):
    level = settings["trigger_level"]
    timebase = settings["time_per_div"] * 10
    num_points_in_window = int(timebase * (len(t) / (t[-1] - t[0])))
    search_start_idx, search_end_idx = len(t) // 4, len(t) * 3 // 4
    v_search = v[search_start_idx:search_end_idx]
    if settings["trigger_edge"] == "Rising":
        indices = np.where((v_search[:-1] < level) & (v_search[1:] >= level))[0]
    else:
        indices = np.where((v_search[:-1] > level) & (v_search[1:] <= level))[0]
    trigger_idx = indices[0] + search_start_idx if len(indices) > 0 else len(t) // 2
    start_idx = max(0, trigger_idx - num_points_in_window // 2)
    end_idx = start_idx + num_points_in_window
    if end_idx > len(t):
        end_idx = len(t)
        start_idx = max(0, end_idx - num_points_in_window)
    return t[start_idx:end_idx], v[start_idx:end_idx]

def calculate_measurements(t, v, settings):
    if len(v) < 2:
        return ("N/A",) * 6
    v_pp, v_rms, v_avg = np.ptp(v), np.sqrt(np.mean(v**2)), np.mean(v)
    zero_crossings = np.where(np.diff(np.sign(v - v_avg)))[0]
    duration = t[-1] - t[0]
    measured_freq = (len(zero_crossings) / 2) / duration if duration > 1e-9 else 0
    delta_v = abs(settings["cursor_v1"] - settings["cursor_v2"])
    t_range = t[-1] - t[0]
    delta_t = abs(settings["cursor_t1_pos"] - settings["cursor_t2_pos"]) * t_range
    return (
        f"{v_pp:.3f} V", f"{v_rms:.3f} V", f"{v_avg:.3f} V",
        f"{measured_freq:.2f} Hz", f"{delta_v:.3f} V", f"{delta_t:.2e} s"
    )

def update_simulation(sender=None, app_data=None):
    for key in _settings:
        if dpg.does_item_exist(key):
            value = dpg.get_value(key)
            if key in ["time_per_div", "volts_per_div"]:
                _settings[key] = float(value)
            else:
                _settings[key] = value
    raw_t, raw_v = generate_signal(_settings)
    display_t, display_v = apply_trigger(raw_t, raw_v, _settings)
    if len(display_t) > 0:
        dpg.set_value("signal_series", [list(display_t), list(display_v)])
    else:
        dpg.set_value("signal_series", [[], []])
    time_total = _settings["time_per_div"] * 10
    volt_total = _settings["volts_per_div"] * 10
    t_center = display_t.mean() if len(display_t) > 0 else 0
    t_start, t_end = t_center - time_total / 2, t_center + time_total / 2
    dpg.set_axis_limits("x_axis", t_start, t_end)
    dpg.set_axis_limits("y_axis", -volt_total / 2, volt_total / 2)
    dpg.set_value("trigger_line", [[t_start, t_end], [_settings["trigger_level"]] * 2])
    dpg.set_value("v_cursor_1", [[t_start, t_end], [_settings["cursor_v1"]] * 2])
    dpg.set_value("v_cursor_2", [[t_start, t_end], [_settings["cursor_v2"]] * 2])
    t1_abs = t_start + _settings["cursor_t1_pos"] * time_total
    t2_abs = t_start + _settings["cursor_t2_pos"] * time_total
    dpg.set_value("t_cursor_1", [[t1_abs] * 2, [-volt_total, volt_total]])
    dpg.set_value("t_cursor_2", [[t2_abs] * 2, [-volt_total, volt_total]])
    vpp, vrms, vavg, freq, dv, dt = calculate_measurements(display_t, display_v, _settings)
    dpg.set_value("vpp_text", vpp)
    dpg.set_value("vrms_text", vrms)
    dpg.set_value("vavg_text", vavg)
    dpg.set_value("freq_text", freq)
    dpg.set_value("delta_v_text", dv)
    dpg.set_value("delta_t_text", dt)

def create_gui():
    dpg.create_context()
    with dpg.window(label="Virtual Oscilloscope", tag="main_window"):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=-350, border=False):
                with dpg.plot(label="Scope Plot", height=-1, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag="x_axis")
                    with dpg.plot_axis(dpg.mvYAxis, label="Voltage (V)", tag="y_axis"):
                        dpg.add_line_series([], [], label="Signal", tag="signal_series")
                        dpg.add_line_series([], [], label="Trigger", tag="trigger_line")
                        dpg.add_line_series([], [], label="V_Cursor1", tag="v_cursor_1")
                        dpg.add_line_series([], [], label="V_Cursor2", tag="v_cursor_2")
                        dpg.add_line_series([], [], label="T_Cursor1", tag="t_cursor_1")
                        dpg.add_line_series([], [], label="T_Cursor2", tag="t_cursor_2")
            with dpg.child_window(width=340):
                def add_control_group(label, controls):
                    with dpg.collapsing_header(label=label, default_open=True):
                        controls()
                def signal_controls():
                    dpg.add_combo(["Sine", "Square", "Triangle", "Sawtooth"], label="Waveform",
                                  tag="signal_type", default_value=_settings["signal_type"], callback=update_simulation)
                    dpg.add_slider_float(label="Frequency", tag="frequency", min_value=0.1, max_value=1000,
                                         default_value=_settings["frequency"], callback=update_simulation, format="%.1f Hz")
                    dpg.add_slider_float(label="Amplitude", tag="amplitude", min_value=0.1, max_value=10.0,
                                         default_value=_settings["amplitude"], callback=update_simulation, format="%.2f V")
                    dpg.add_slider_float(label="DC Offset", tag="offset", min_value=-5.0, max_value=5.0,
                                         default_value=_settings["offset"], callback=update_simulation, format="%.2f V")
                    dpg.add_slider_float(label="Noise", tag="noise", min_value=0.0, max_value=1.0,
                                         default_value=_settings["noise"], callback=update_simulation, format="%.2f V")
                def scale_controls():
                    dpg.add_combo(["0.0001", "0.001", "0.01", "0.02", "0.05", "0.1"], label="Time/Div",
                                  tag="time_per_div", default_value=str(_settings["time_per_div"]), callback=update_simulation)
                    dpg.add_combo(["0.01", "0.1", "0.2", "0.5", "1.0", "2.0"], label="Volts/Div",
                                  tag="volts_per_div", default_value=str(_settings["volts_per_div"]), callback=update_simulation)
                def trigger_controls():
                    dpg.add_slider_float(label="Level", tag="trigger_level", min_value=-10.0, max_value=10.0,
                                         default_value=_settings["trigger_level"], callback=update_simulation, format="%.2f V")
                    dpg.add_combo(["Rising", "Falling"], label="Edge", tag="trigger_edge",
                                  default_value=_settings["trigger_edge"], callback=update_simulation)
                def cursor_controls():
                    dpg.add_slider_float(label="V1", tag="cursor_v1", min_value=-10.0, max_value=10.0,
                                         default_value=_settings["cursor_v1"], callback=update_simulation, format="%.2f V")
                    dpg.add_slider_float(label="V2", tag="cursor_v2", min_value=-10.0, max_value=10.0,
                                         default_value=_settings["cursor_v2"], callback=update_simulation, format="%.2f V")
                    dpg.add_slider_float(label="T1", tag="cursor_t1_pos", min_value=0.0, max_value=1.0,
                                         default_value=_settings["cursor_t1_pos"], callback=update_simulation)
                    dpg.add_slider_float(label="T2", tag="cursor_t2_pos", min_value=0.0, max_value=1.0,
                                         default_value=_settings["cursor_t2_pos"], callback=update_simulation)
                def measurement_display():
                    for label, tag in [("Vpp", "vpp_text"), ("Vrms", "vrms_text"),
                                       ("Vavg", "vavg_text"), ("Freq", "freq_text"),
                                       ("ΔV", "delta_v_text"), ("ΔT", "delta_t_text")]:
                        with dpg.group(horizontal=True):
                            dpg.add_text(f"{label}:")
                            dpg.add_text("N/A", tag=tag)
                add_control_group("Signal Generator", signal_controls)
                add_control_group("Vertical & Horizontal", scale_controls)
                add_control_group("Trigger", trigger_controls)
                add_control_group("Cursors", cursor_controls)
                add_control_group("Measurements", measurement_display)
    dpg.create_viewport(title='Virtual Oscilloscope (Python)', width=1280, height=720)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main_window", True)

if __name__ == "__main__":
    create_gui()
    update_simulation()
    dpg.start_dearpygui()
    dpg.destroy_context()
