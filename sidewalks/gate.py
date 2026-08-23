#!/usr/bin/env python3
"""
Step 1 gate: is Farmington's sidewalk coverage in OSM dense enough to map?

Counts three things over the town boundary:
  - total road kilometres
  - kilometres of road carrying an explicit sidewalk=* tag
  - kilometres of sidewalk drawn as separate footway geometry

Prints the tagged fraction and a PASS/FAIL against the threshold. Nothing
downstream runs unless this passes.

Usage:
    python3 gate.py                      # Farmington, Connecticut
    python3 gate.py --state US-MI --threshold 0.2
"""

import argparse
import json
import math
import sys
import urllib.error
import urllib.parse
import urllib.request

# Mirrors are tried in order; the first that answers wins.
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# The public street network a pedestrian could plausibly walk along.
# service ways (driveways, parking aisles) and motorways are excluded from the
# denominator by default -- they are never sidewalk-tagged and would deflate the
# fraction into meaninglessness. Both are still measured and reported.
ROAD_CORE = [
    "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "pedestrian",
    "trunk_link", "primary_link", "secondary_link", "tertiary_link",
]
ROAD_EXCLUDED = ["motorway", "motorway_link", "service"]

SIDEWALK_YES = {"both", "left", "right", "yes", "separate"}
SIDEWALK_NO = {"no", "none"}


def build_query(town, state, timeout=180):
    """Overpass QL for every road and footway inside the named town."""
    roads = "|".join(ROAD_CORE + ROAD_EXCLUDED)
    return f"""
[out:json][timeout:{timeout}];
area["ISO3166-2"="{state}"]["admin_level"="4"]->.state;
rel(area.state)["boundary"="administrative"]["admin_level"="8"]["name"="{town}"];
map_to_area->.town;
(
  way["highway"~"^({roads})$"](area.town);
  way["highway"~"^(footway|path|cycleway|steps)$"](area.town);
);
out geom;
""".strip()


def overpass(query):
    """POST to each mirror in turn. Raises with a plain-language reason."""
    reasons = []
    payload = urllib.parse.urlencode({"data": query}).encode()
    for url in OVERPASS_ENDPOINTS:
        req = urllib.request.Request(
            url, data=payload,
            headers={"User-Agent": "farmington-sidewalk-gate/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 407):
                reasons.append(f"{url}: {exc.code} from egress proxy (host not permitted)")
            else:
                reasons.append(f"{url}: HTTP {exc.code}")
        except Exception as exc:  # URLError, timeout, JSON garbage
            reasons.append(f"{url}: {exc}")
    raise SystemExit(
        "Could not reach any Overpass mirror:\n  "
        + "\n  ".join(reasons)
        + "\n\nIf these are 403s, the hosts are blocked by network policy rather "
          "than down. Allowlist an Overpass mirror and re-run."
    )


def way_km(geometry):
    """Haversine length of an OSM way's geometry, in kilometres."""
    total = 0.0
    for a, b in zip(geometry, geometry[1:]):
        lat1, lon1 = math.radians(a["lat"]), math.radians(a["lon"])
        lat2, lon2 = math.radians(b["lat"]), math.radians(b["lon"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        total += 2 * 6371.0088 * math.asin(math.sqrt(h))
    return total


def sidewalk_state(tags):
    """Classify a road's own sidewalk tagging: 'yes', 'no', or 'unknown'.

    Reads sidewalk=* plus the sidewalk:left/right/both side keys. A road is
    'no' only when every side present says no -- one tagged side that says
    yes makes the road a yes.
    """
    values = []
    for key in ("sidewalk", "sidewalk:both", "sidewalk:left", "sidewalk:right"):
        if key in tags:
            values.append(tags[key].strip().lower())
    if not values:
        return "unknown"
    if any(v in SIDEWALK_YES for v in values):
        return "yes"
    if all(v in SIDEWALK_NO for v in values):
        return "no"
    return "unknown"  # e.g. sidewalk=unknown, or a value we don't recognise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Farmington")
    ap.add_argument("--state", default="US-CT", help="ISO3166-2 code, e.g. US-CT")
    ap.add_argument("--threshold", type=float, default=0.20)
    ap.add_argument("--save", default="raw.json", help="cache the Overpass response here")
    args = ap.parse_args()

    print(f"Querying Overpass for {args.town} ({args.state})...", file=sys.stderr)
    data = overpass(build_query(args.town, args.state))
    elements = data.get("elements", [])
    if not elements:
        raise SystemExit(
            f"Overpass returned nothing for {args.town} in {args.state}. "
            "Check the town name and state code -- there are many Farmingtons."
        )
    with open(args.save, "w") as fh:
        json.dump(data, fh)
    print(f"  {len(elements)} ways -> {args.save}", file=sys.stderr)

    km = {
        "core": 0.0, "excluded": 0.0,
        "tag_yes": 0.0, "tag_no": 0.0, "tag_unknown": 0.0,
        "sidewalk_geom": 0.0, "footway_other": 0.0,
    }
    for el in elements:
        if el.get("type") != "way" or not el.get("geometry"):
            continue
        tags = el.get("tags", {})
        hw = tags.get("highway")
        length = way_km(el["geometry"])

        if hw in ("footway", "path", "cycleway", "steps"):
            if tags.get("footway") == "sidewalk" or tags.get("path") == "sidewalk":
                km["sidewalk_geom"] += length
            else:
                km["footway_other"] += length
            continue

        if hw in ROAD_EXCLUDED:
            km["excluded"] += length
            continue

        km["core"] += length
        km["tag_" + sidewalk_state(tags)] += length

    tagged = km["tag_yes"] + km["tag_no"]
    fraction = tagged / km["core"] if km["core"] else 0.0

    print()
    print(f"  Farmington, {args.state} -- OSM sidewalk coverage gate")
    print("  " + "-" * 52)
    print(f"  Total road km (walkable public network) {km['core']:9.1f}")
    print(f"    of which explicitly tagged            {tagged:9.1f}")
    print(f"      sidewalk present                    {km['tag_yes']:9.1f}")
    print(f"      sidewalk explicitly none            {km['tag_no']:9.1f}")
    print(f"    untagged (unsurveyed)                 {km['tag_unknown']:9.1f}")
    print()
    print(f"  Separate sidewalk geometry km           {km['sidewalk_geom']:9.1f}")
    print(f"  Other footway/path km                   {km['footway_other']:9.1f}")
    print(f"  Excluded (motorway, service) km         {km['excluded']:9.1f}")
    print("  " + "-" * 52)
    print(f"  TAGGED FRACTION                         {fraction:9.1%}")
    print()

    if fraction < args.threshold:
        print(f"  GATE FAILED -- {fraction:.1%} < {args.threshold:.0%} threshold.")
        print("  OSM does not know enough about Farmington's sidewalks.")
        print("  The map is not worth building from this data. Stopping here.")
        return 1

    print(f"  GATE PASSED -- {fraction:.1%} >= {args.threshold:.0%} threshold.")
    print(f"  Proceed to the join and render steps using {args.save}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
