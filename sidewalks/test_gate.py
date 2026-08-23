"""Offline checks for the gate's measurement logic."""
import json, math, sys, importlib.util

spec = importlib.util.spec_from_file_location("gate", "gate.py")
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)

fails = []
def check(name, got, want, tol=None):
    ok = abs(got - want) <= tol if tol is not None else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok: fails.append(name)

print("way_km -- one degree of latitude at the equator is ~111.19 km")
check("1 deg lat", gate.way_km([{"lat":0,"lon":0},{"lat":1,"lon":0}]), 111.19, tol=0.05)

print("way_km -- a known short segment in Farmington CT (~0.5 km)")
# 41.7200,-72.8320 -> 41.7245,-72.8320 is 0.0045 deg lat
check("0.0045 deg lat", gate.way_km([{"lat":41.72,"lon":-72.832},{"lat":41.7245,"lon":-72.832}]), 0.5004, tol=0.005)

print("way_km -- multi-segment sums")
three = gate.way_km([{"lat":0,"lon":0},{"lat":1,"lon":0},{"lat":2,"lon":0}])
check("2 deg via 3 nodes", three, 222.39, tol=0.1)
check("single node is zero", gate.way_km([{"lat":5,"lon":5}]), 0.0)

print("sidewalk_state classification")
for tags, want in [
    ({}, "unknown"),
    ({"sidewalk":"both"}, "yes"),
    ({"sidewalk":"no"}, "no"),
    ({"sidewalk":"none"}, "no"),
    ({"sidewalk":"separate"}, "yes"),
    ({"sidewalk":"BOTH"}, "yes"),              # case-insensitive
    ({"sidewalk:left":"yes","sidewalk:right":"no"}, "yes"),   # one side counts
    ({"sidewalk:left":"no","sidewalk:right":"no"}, "no"),     # both sides none
    ({"sidewalk":"unknown"}, "unknown"),       # explicit unknown != tagged
    ({"foot":"no"}, "unknown"),                # foot= is not sidewalk=
]:
    check(f"sidewalk_state({tags})", gate.sidewalk_state(tags), want)

print("end-to-end over a fixture with hand-computed totals")
# Each way is 0.0045 deg of latitude = 0.5004 km, so totals are countable by hand.
def seg(i): return [{"lat":41.72+i*0.01,"lon":-72.832},{"lat":41.7245+i*0.01,"lon":-72.832}]
fixture = {"elements":[
    {"type":"way","geometry":seg(0),"tags":{"highway":"residential","sidewalk":"both"}},
    {"type":"way","geometry":seg(1),"tags":{"highway":"residential","sidewalk":"no"}},
    {"type":"way","geometry":seg(2),"tags":{"highway":"residential"}},            # unknown
    {"type":"way","geometry":seg(3),"tags":{"highway":"tertiary","sidewalk:left":"yes"}},
    {"type":"way","geometry":seg(4),"tags":{"highway":"service"}},                # excluded
    {"type":"way","geometry":seg(5),"tags":{"highway":"motorway"}},               # excluded
    {"type":"way","geometry":seg(6),"tags":{"highway":"footway","footway":"sidewalk"}},
    {"type":"way","geometry":seg(7),"tags":{"highway":"footway"}},                # other footway
    {"type":"node","lat":41.7,"lon":-72.8},                                       # ignored
]}
gate.overpass = lambda q: fixture
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["gate.py","--save","/tmp/fixture.json"]
    rc = gate.main()
out = buf.getvalue()
print(out)
U = 0.5004
# core = 4 ways (2 res tagged, 1 res unknown, 1 tertiary) = 4U ; tagged = 3U -> 75%
check("core km", float(out.split("public network)")[1].split()[0]), 4*U, tol=0.01)
check("tagged yes km", float(out.split("sidewalk present")[1].split()[0]), 2*U, tol=0.01)
check("tagged none km", float(out.split("explicitly none")[1].split()[0]), 1*U, tol=0.01)
check("unknown km", float(out.split("(unsurveyed)")[1].split()[0]), 1*U, tol=0.01)
check("separate sidewalk km", float(out.split("geometry km")[1].split()[0]), 1*U, tol=0.01)
check("excluded km", float(out.split("service) km")[1].split()[0]), 2*U, tol=0.01)
check("fraction is 75%", out.split("TAGGED FRACTION")[1].split()[0], "75.0%")
check("gate passes at 75%", rc, 0)

print("gate fails below threshold")
gate.overpass = lambda q: {"elements":[
    {"type":"way","geometry":seg(0),"tags":{"highway":"residential","sidewalk":"both"}},
] + [{"type":"way","geometry":seg(i),"tags":{"highway":"residential"}} for i in range(1,10)]}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["gate.py","--save","/tmp/fixture2.json"]
    rc2 = gate.main()
check("10% fraction fails gate", rc2, 1)
check("prints FAILED", "GATE FAILED" in buf.getvalue(), True)

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
