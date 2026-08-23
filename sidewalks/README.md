# Farmington sidewalk map

## Status: blocked at step one

The gate has not run. Every OSM data host is refused by this environment's
network egress policy (HTTP 403 at the proxy, not a timeout or an outage):

    overpass-api.de            403
    overpass.kumi.systems      403
    overpass.private.coffee    403
    z.overpass-api.de          403
    lz4.overpass-api.de        403
    nominatim.openstreetmap.org 403
    api.openstreetmap.org      403
    download.geofabrik.de      403

So there are no real numbers yet, and no fraction to report. The join and
render steps stay unbuilt until the gate actually clears on real data.

## Running the gate

Anywhere with Overpass access:

    python3 gate.py

Defaults to Farmington, Connecticut (`--state US-CT`), chosen because the
repo README places its author in Connecticut. There are many Farmingtons --
pass `--state US-MI`, `--state US-NM` and so on for a different one.

Exit code 0 means the tagged fraction cleared the 20% threshold and step two
is worth starting; exit 1 means it did not and the plan ends there.

## What it measures

- **Total road km** -- the walkable public street network: trunk through
  residential, plus living_street and pedestrian and the link roads.
  Motorways and `highway=service` (driveways, parking aisles) are measured
  but held out of the denominator; they are never sidewalk-tagged and would
  deflate the fraction into meaninglessness. Both are printed so the choice
  is auditable.
- **Explicitly tagged km** -- roads carrying `sidewalk=*` or the
  `sidewalk:left` / `:right` / `:both` side keys. A road counts as having a
  sidewalk if any tagged side says so, and as explicitly having none only
  when every tagged side says no. `sidewalk=unknown` is not tagging.
- **Separate sidewalk geometry km** -- ways tagged `footway=sidewalk`.
  Other footways and paths are counted separately; they are trails and
  crossings, not sidewalks.

Lengths are haversine over the way geometry.

## Verifying

    python3 test_gate.py

22 offline checks: haversine distances against known values, the sidewalk
tag classification table including the mixed-sides and case cases, and two
end-to-end runs over fixtures with hand-computed totals -- one that clears
the threshold and one that does not.
