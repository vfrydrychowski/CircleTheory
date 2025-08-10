import umidiparser as midi
from typing import List, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import colorsys


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
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})', note_names[semitone]


def plot_segments(df: pd.DataFrame):
    """Plots the MIDI note segments using Plotly."""
    # Initialize Plotly figure
    fig = go.Figure()

    # Track which notes have been added to the legend
    legend_added = set()

    # Plot each segment as a single line segment with hover info
    for _, row in df.iterrows():
        note_label = f'{row["note_name"]}{row["note"] // 12 - 1}'
        fig.add_trace(
            go.Scatter(
                x=[row["x1"], row["x2"], None],  # Add None to prevent connecting lines between segments
                y=[row["y1"], row["y2"], None],
                mode="lines",
                line=dict(color=row["color"], width=4),
                name=note_label if note_label not in legend_added else None,  # Only show legend once per note
                hoverinfo="text",
                text=f'Note: {note_label} ({row["note"]})',
                legendgroup=note_label,  # Group all segments of the same note
                showlegend=note_label not in legend_added,  # Show legend only for the first occurrence
            )
        )
        legend_added.add(note_label)  # Mark this note as added to the legend

    # Layout adjustments
    fig.update_layout(
        title="Line Segments Colored by MIDI Note (Unique Legend Entries)",
        xaxis_title="X",
        yaxis_title="Y",
        template="plotly_dark",
        width=1200,
        height=1200,
        showlegend=True,
        legend_traceorder="grouped",
        legend_itemclick=False,
        legend_itemdoubleclick=False,
    )

    fig.update_traces(selector=dict(type="scatter"), visible=True)

    fig.show(renderer="browser")


def main():
    midi_path = "/home/fryghost/Win/code$/CircleTheory/one_note_quantize.mid"
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
