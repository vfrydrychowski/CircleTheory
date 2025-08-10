import umidiparser as midi
from typing import List, Dict
import numpy as np
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def event_to_dict(me: midi.MidiEvent) -> Dict[str, int]:
    if (me.status == midi.NOTE_OFF) or (me.status == midi.NOTE_ON) and (me.velocity == 0):
        status = 0
    elif me.status == midi.NOTE_ON:
        status = 1
    else:
        status = None
    return {
        "status": status,
        "note": me.note,
        "delta": me.delta_us,
    }


def delta_to_time(midi_list: List[Dict[str,float]]) -> List[Dict[str,float]]:
    prev_delta = 0
    for e in midi_list:
        time = prev_delta + e["delta"]
        prev_delta = time
        e["time"] = time
    return midi_list


def midi_to_list(mf: midi.MidiFile) -> List[Dict[str,int]]:
    midi_list: List[Dict[str, int]] = [
        event_to_dict(ev) for ev in mf if (ev.status == midi.NOTE_OFF) or (ev.status == midi.NOTE_ON)
    ]
    return midi_list


def note_to_rad(n: int):
    return (np.pi/6)*(5-n)%(2*np.pi)


def list_to_point(l: List[Dict[str,float]]):
    point_list = []
    for i_e in range(len(l)):
        event = l[i_e]
        if i_e == 0:
            point_list.append({
                "status": event["status"],
                "note": event["note"],
                "x": 0,
                "y": 0,
                "time": 0,
            })
        else :
            if event["status"] == 1:
                prev_point = point_list[-1]
                prev_rad = note_to_rad(prev_point["note"])
                prev_x = prev_point["x"]
                prev_y = prev_point["y"]
                point_list.append({
                    "status": event["status"],
                    "note": event["note"],
                    "x": np.cos(prev_rad)*event["delta"] + prev_x,
                    "y": np.sin(prev_rad)*event["delta"] + prev_y,
                    "time": event["time"],
                })
            else:
                for inv in range(-1, -len(point_list)-1, -1):
                    if point_list[inv]["note"] == event["note"]:
                        start_event = point_list[inv]
                        prev_x = start_event["x"]
                        prev_y = start_event["y"]
                        rad = note_to_rad(event["note"])
                        delta = event["time"] - start_event["time"]
                        point_list.append({
                            "status": event["status"],
                            "note": event["note"],
                            "x": np.cos(rad)*delta + prev_x,
                            "y": np.sin(rad)*delta + prev_y,
                            "time": event["time"],
                        })
                        break
    return point_list


def point_to_segment(l):
    l_segments = []
    for i in range(len(l)):
        point = l[i]
        if point["status"] == 1:
            for k in range(i+1,len(l)):
                end_point=l[k]
                if (point["note"] == end_point["note"]) and end_point["status"] == 0:
                    l_segments.append({
                        "note": point["note"],
                        "x1": point["x"],
                        "y1": point["y"],
                        "x2": end_point["x"],
                        "y2": end_point["y"],
                        "lenght": end_point["time"] - point["time"],
                    })
                    break
    return l_segments


# Define color map function
def midi_note_to_color(note):
    """Map a MIDI note to a color using HSL with octave-based brightness."""
    semitone = note % 12
    octave = note // 12 - 1
    hue = semitone * (360 / 12) / 360  # Normalized hue
    base_lightness = 0.4
    lightness_step = 0.05
    lightness = min(base_lightness + octave * lightness_step, 0.9)
    saturation = 0.8
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    # Music notation for the 12 pitch classes
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # Return RGB tuple (values 0-1) and the note name
    return (r, g, b), note_names[semitone]


def plot_segments(df: pd.DataFrame):
    """Plots the MIDI note segments using Matplotlib."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 10))

    # Track which notes have been added to the legend
    legend_added = set()

    # Plot each segment as a single line segment
    for _, row in df.iterrows():
        note_label = f'{row["note_name"]}{row["note"] // 12 - 1}'

        # Only add a label for the legend for the first occurrence of a note
        if note_label not in legend_added:
            ax.plot(
                [row["x1"], row["x2"]],
                [row["y1"], row["y2"]],
                color=row["color"],
                linewidth=2,
                label=note_label,
            )
            legend_added.add(note_label)
        else:
            ax.plot(
                [row["x1"], row["x2"]], [row["y1"], row["y2"]], color=row["color"], linewidth=2
            )

    # Layout adjustments
    ax.set_title("Line Segments Colored by MIDI Note")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.show()


def main():
    # Set up the root Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window, we only want the file dialog

    # Open file dialog to select a MIDI file
    midi_path = filedialog.askopenfilename(
        title="Select a MIDI file to visualize",
        filetypes=(("MIDI files", "*.mid *.midi"), ("All files", "*.*")),
    )

    # If the user cancels the dialog, midi_path will be empty
    if not midi_path:
        print("No file selected. Exiting.")
        return
    midi_file = midi.MidiFile(midi_path, buffer_size=0)
    list_dic = midi_to_list(midi_file)
    l_segment = point_to_segment(list_to_point(delta_to_time(list_dic)))
    df = pd.DataFrame(l_segment)

    # Assign colors and note names
    df[["color", "note_name"]] = df["note"].apply(
        lambda n: pd.Series(midi_note_to_color(n))
    )

    plot_segments(df)


if __name__ == "__main__":
    main()
