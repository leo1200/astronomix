import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from astronomix.option_classes.simulation_config import FOURTH_ORDER_CONSERVATIVE
from _collapse_lib import run_collapse

snaps, helper, rv = run_collapse(32, FOURTH_ORDER_CONSERVATIVE, t_end=0.4,
                                 want_states=True)
print("snapshot fields:", [f for f in snaps._fields] if hasattr(snaps, "_fields") else dir(snaps))
st = np.asarray(snaps.states)
tp = np.asarray(snaps.time_points)
print("states shape:", st.shape, "dtype:", st.dtype)
print("time_points:", tp)
print("density_index:", rv.density_index, "pressure_index:", rv.pressure_index)
print("snap0 rho min/max:", float(np.min(st[0, rv.density_index])), float(np.max(st[0, rv.density_index])))
print("snap-1 rho min/max:", float(np.min(st[-1, rv.density_index])), float(np.max(st[-1, rv.density_index])))
print("helper.r shape:", np.asarray(helper.r).shape, "state field shape:", st[0, rv.density_index].shape)
