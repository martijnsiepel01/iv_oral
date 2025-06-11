import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple


# Updated graph builder: include all available data (prescriptions, cultures, vitals: bpm, o2, temp, crp)
def build_detailed_structured_graph(treatment, measurements):
    G = nx.DiGraph()

    # Define day range
    start_date = pd.to_datetime(treatment["treatment_start"]).normalize()
    end_date = pd.to_datetime(treatment["treatment_end"]).normalize()
    days = pd.date_range(start=start_date, end=end_date)

    # Create Day nodes and sequential links
    for i, day in enumerate(days):
        day_node = f"Day_{day.date()}"
        G.add_node(day_node, type="day", label=day.strftime("Day %Y-%m-%d"))
        if i > 0:
            prev_day_node = f"Day_{days[i - 1].date()}"
            G.add_edge(prev_day_node, day_node, type="temporal")

    # Add prescriptions with full info
    for i, p in enumerate(treatment.get("prescriptions", [])):
        p_node = f"Prescription_{i+1}"
        label = (
            f"{p['MedicatieStofnaam']} ({p['ToedieningsRoute']})\n"
            f"Voorschrijf: {p['VoorschrijfDatumTijd']}\n"
            f"Doel: {p['doel']}\n"
            f"Locatie: {p['locatie']}\n"
            f"Informatie: {p.get('aanvullende_informatie', '-')}\n"
            f"Specialisme: {p['Specialisme']}\n"
            f"Groep: {p['Specialisme_groep']}\n"
            f"Setting: {p['community_or_hospital']}"
        )
        G.add_node(p_node, type="prescription", label=label)

        p_start = pd.to_datetime(p["StartDatumTijd"]).normalize()
        p_end = pd.to_datetime(p["StopDatumTijd"]).normalize()
        for d in pd.date_range(p_start, p_end):
            d_node = f"Day_{d.date()}"
            if d_node in G:
                G.add_edge(d_node, p_node, type="prescribed")

    # Add cultures with full info
    for i, c in enumerate(treatment.get("treatment_cultures", [])):
        c_node = f"Culture_{i+1}"
        label = (
            f"Materiaal: {c['materiaal_catCustom']}\n"
            f"Resultaat: {c['kweek_uitslagDef']}\n"
            f"Genus: {c.get('microbe_genus', '-')}\n"
            f"Microbe type: {c['microbe_catCustom']}\n"
        )
        G.add_node(c_node, type="culture", label=label)

        c_day = pd.to_datetime(c["afnamedatumtijd"]).normalize()
        d_node = f"Day_{c_day.date()}"
        if d_node in G:
            G.add_edge(d_node, c_node, type="culture")

    # Summarize vitals/labs into one measurement node per day
    vitals_by_day = {}
    for key, entries in measurements.items():
        for e in entries:
            date = pd.to_datetime(e["MeetMoment"]).date()
            vitals_by_day.setdefault(date, {}).setdefault(key, []).append(e)

    for i, day in enumerate(days):
        date = day.date()
        if date not in vitals_by_day:
            continue
        v = vitals_by_day[date]
        label_parts = []
        if "bpm" in v:
            hr_vals = [e["Hartfrequentie"] for e in v["bpm"] if "Hartfrequentie" in e]
            if hr_vals:
                label_parts.append(f"HR {int(pd.Series(hr_vals).mean())}")
        if "o2" in v:
            o2_vals = [e["o2Saturatie"] for e in v["o2"] if "o2Saturatie" in e]
            if o2_vals:
                label_parts.append(f"O2 {int(pd.Series(o2_vals).mean())}%")
        if "temp" in v:
            t_vals = [e["Temperatuur"] for e in v["temp"] if "Temperatuur" in e]
            if t_vals:
                label_parts.append(f"T {pd.Series(t_vals).mean():.1f}°C")
        if "crp" in v:
            crp_vals = [e["UitslagNumeriek"] for e in v["crp"] if "UitslagNumeriek" in e]
            if crp_vals:
                label_parts.append(f"CRP {pd.Series(crp_vals).mean():.1f}")

        m_node = f"Measurements_{i+1}"
        label = ", ".join(label_parts)
        G.add_node(m_node, type="measurement", label=label)
        d_node = f"Day_{date}"
        if d_node in G:
            G.add_edge(d_node, m_node, type="measurement")

    return G

# Visualization function
def visualize_graph(G, title="Treatment Graph"):
    pos = nx.spring_layout(G, seed=42)
    node_colors = {
        "day": "skyblue",
        "prescription": "lightgreen",
        "culture": "orange",
        "measurement": "violet"
    }
    node_color_list = [node_colors.get(G.nodes[n]["type"], "gray") for n in G.nodes]
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels,
            node_color=node_color_list, node_size=1200, font_size=8, arrows=True)
    plt.title(title)
    plt.show()

def get_treatment_by_ids(data: dict, patient_id: str, treatment_id: int) -> Tuple[dict, str]:
    """Find a treatment from the JSON data by patient and treatment ID."""
    patient = data.get(patient_id)
    if not patient:
        raise ValueError(f"No patient with ID {patient_id}")
    for admission in patient.get("admissions", []):
        for treatment in admission.get("treatments", []):
            if treatment["treatment_id"] == treatment_id:
                return treatment, f"Patient {patient_id} - Treatment {treatment_id}"
    raise ValueError(f"No treatment with ID {treatment_id} found for patient {patient_id}")

# Load JSON and visualize nth treatment
json_path = r"C:\Users\Martijn\OneDrive\PhD\iv_oral_v0.1\data_raw\prescriptions\prescriptions_with_measurements_agg_1h.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Load treatment and measurements
selected_treatment, title = get_treatment_by_ids(data, "5", 1)
selected_measurements = next(
    adm["measurements"]
    for adm in data["44"]["admissions"]
    if any(t["treatment_id"] == 1 for t in adm["treatments"])
)

# Build and visualize graph
G = build_detailed_structured_graph(selected_treatment, selected_measurements)
visualize_graph(G, title=f"{title} (Structured View)")