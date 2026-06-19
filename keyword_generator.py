#!/usr/bin/env python3
"""
generate_queries.py — Build long-tail YouTube search queries by combining
vehicle/format terms with locations and modifiers, and save them to a file.
"""

import itertools
import sys

VEHICLE_FORMAT = [
    "bike ride", "car drive", "motovlog", "road trip", "dashcam",
    "scooter ride", "truck drive", "bike vlog", "car vlog",
]

LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Chennai", "Kolkata",
    "Goa", "Ladakh", "Kashmir", "Himachal", "Rajasthan", "Kerala",
    "Northeast India", "NH48", "Mumbai-Pune Expressway", "Yamuna Expressway",
    "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh", "Coimbatore", "Surat",
]

MODIFIERS = [
    "vlog", "POV", "GoPro", "4K", "dashcam", "highway", "city traffic",
    "night", "monsoon", "daily commute",
]

def generate(max_queries: int = None) -> list[str]:
    queries = []
    for vf, loc, mod in itertools.product(VEHICLE_FORMAT, LOCATIONS, MODIFIERS):
        # Avoid redundant phrasing like "dashcam ... dashcam"
        if vf == "dashcam" and mod == "dashcam":
            continue
        queries.append(f"{vf} {loc} {mod}")

    if max_queries:
        queries = queries[:max_queries]
        
    return queries

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    queries = generate(limit)
    
    # Save the generated queries directly to a text file
    output_filename = "combinatorial_keywords.txt"
    
    with open(output_filename, "w", encoding="utf-8") as file:
        for q in queries:
            file.write(q + "\n")
            
    print(f"Success! Saved {len(queries)} keywords to '{output_filename}'.")