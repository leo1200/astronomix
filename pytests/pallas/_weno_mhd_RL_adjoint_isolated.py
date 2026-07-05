"""Isolated bit-exact check of the hand-derived transposes of the MHD WENO
R_col / L_row coefficient maps w.r.t. the differentiable eigenstructure scalar
vector (``proj_keys``), versus ``jax.vjp``.

These are vjps #2 (``_Rcol_apply``) and #3 (``_proj_functional``) from the task.
Both maps are explicit polynomial/rational algebra in the 16 ``proj_keys``
scalars, with piecewise-constant (sgn/branch) factors held fixed.  We replicate
the forward closures standalone and the hand transpose, and compare.
"""
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

gamma = 5.0 / 3.0
gm1 = gamma - 1.0
gam0 = 1.0 - gamma
gam1 = 0.5 * (gamma - 1.0)
gam2 = (gamma - 2.0) / (gamma - 1.0)
zero_typed = 0.0
one_typed = 1.0
ncomp = 8

proj_keys = ('vn', 'vt1', 'vt2', 'Bt1', 'Bt2', 'v2', 'csq', 'cface',
             'lf', 'ls', 'amf', 'ams', 'bn1', 'bn2', 'srho', 'invc')


def make_fp(rng):
    # generic non-degenerate eigenstructure scalars
    fp = dict(
        vn=rng.normal(), vt1=rng.normal(), vt2=rng.normal(),
        Bt1=rng.normal(), Bt2=rng.normal(),
        v2=abs(rng.normal()) + 0.5,
        csq=abs(rng.normal()) + 0.5,
        cface=abs(rng.normal()) + 0.5,
        lf=abs(rng.normal()) + 1.0,
        ls=abs(rng.normal()) + 0.2,
        amf=abs(rng.normal()) + 0.3,
        ams=abs(rng.normal()) + 0.3,
        bn1=rng.normal(), bn2=rng.normal(),
        srho=abs(rng.normal()) + 0.5,
        invc=abs(rng.normal()) + 0.5,
        # constants (held fixed):
        sbn=1.0 if rng.random() > 0.5 else -1.0,
        sbt=1.0 if rng.random() > 0.5 else -1.0,
        geq=bool(rng.random() > 0.5),
    )
    return {k: jnp.asarray(v) if not isinstance(v, bool) else v for k, v in fp.items()}


# -------- forward R_cols / L_rows (replica of the module closures) --------
def _Rcols(mode, fp):
    vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
    v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
    lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
    bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
    srho = fp['srho']; geq = fp['geq']
    if mode == 0:
        R = [amf, amf * (vn - lf),
             amf * vt1 + ams * ls * bn1 * sbn,
             amf * vt2 + ams * ls * bn2 * sbn, zero_typed,
             cface * ams * bn1 / srho, cface * ams * bn2 / srho,
             amf * (lf * lf - lf * vn + 0.5 * v2 - gam2 * csq)
             + ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn]
        scale = jnp.where(~geq, sbt, one_typed)
    elif mode == 1:
        R = [zero_typed, zero_typed, -bn2, bn1, zero_typed,
             -bn2 * sbn / srho, bn1 * sbn / srho, bn1 * vt2 - bn2 * vt1]
        scale = one_typed
    elif mode == 2:
        R = [ams, ams * (vn - ls),
             ams * vt1 - amf * lf * bn1 * sbn,
             ams * vt2 - amf * lf * bn2 * sbn, zero_typed,
             -cface * amf * bn1 / srho, -cface * amf * bn2 / srho,
             ams * (ls * ls - ls * vn + 0.5 * v2 - gam2 * csq)
             - amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn]
        scale = jnp.where(geq, sbt, one_typed)
    elif mode == 3:
        R = [one_typed, vn, vt1, vt2, zero_typed, zero_typed, zero_typed, 0.5 * v2]
        scale = one_typed
    elif mode == 4:
        R = [ams, ams * (vn + ls),
             ams * vt1 + amf * lf * bn1 * sbn,
             ams * vt2 + amf * lf * bn2 * sbn, zero_typed,
             -cface * amf * bn1 / srho, -cface * amf * bn2 / srho,
             ams * (ls * ls + ls * vn + 0.5 * v2 - gam2 * csq)
             + amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn]
        scale = jnp.where(geq, sbt, one_typed)
    elif mode == 5:
        R = [zero_typed, zero_typed, -bn2, bn1, zero_typed,
             bn2 * sbn / srho, -bn1 * sbn / srho, bn1 * vt2 - bn2 * vt1]
        scale = one_typed
    else:
        R = [amf, amf * (vn + lf),
             amf * vt1 - ams * ls * bn1 * sbn,
             amf * vt2 - ams * ls * bn2 * sbn, zero_typed,
             cface * ams * bn1 / srho, cface * ams * bn2 / srho,
             amf * (lf * lf + lf * vn + 0.5 * v2 - gam2 * csq)
             - ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn]
        scale = jnp.where(~geq, sbt, one_typed)
    return R, scale


def _Lrows(mode, fp):
    vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
    Bt1 = fp['Bt1']; Bt2 = fp['Bt2']
    v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
    lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
    bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
    srho = fp['srho']; geq = fp['geq']; invc = fp['invc']
    if mode == 0:
        L = [(amf * (gam1 * v2 + lf * vn) - ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
             (amf * (gam0 * vn - lf), 1),
             (gam0 * amf * vt1 + ams * ls * bn1 * sbn, 2),
             (gam0 * amf * vt2 + ams * ls * bn2 * sbn, 3),
             (gam0 * amf * Bt1 + cface * ams * bn1 * srho, 5),
             (gam0 * amf * Bt2 + cface * ams * bn2 * srho, 6),
             (-gam0 * amf, 7)]
        scale = 0.5 * invc * jnp.where(~geq, sbt, one_typed)
        return L, scale
    if mode == 1:
        L = [(bn2 * vt1 - bn1 * vt2, 0), (-bn2, 2), (bn1, 3),
             (-bn2 * sbn * srho, 5), (bn1 * sbn * srho, 6)]
        return L, 0.5 + zero_typed
    if mode == 2:
        L = [(ams * (gam1 * v2 + ls * vn) + amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
             (ams * (gam0 * vn) - ams * ls, 1),
             (gam0 * ams * vt1 - amf * lf * bn1 * sbn, 2),
             (gam0 * ams * vt2 - amf * lf * bn2 * sbn, 3),
             (gam0 * ams * Bt1 - cface * amf * bn1 * srho, 5),
             (gam0 * ams * Bt2 - cface * amf * bn2 * srho, 6),
             (-gam0 * ams, 7)]
        scale = 0.5 * invc * jnp.where(geq, sbt, one_typed)
        return L, scale
    if mode == 3:
        L = [(-csq / gam0 - 0.5 * v2, 0), (vn, 1), (vt1, 2), (vt2, 3),
             (Bt1, 5), (Bt2, 6), (-1.0 + zero_typed, 7)]
        return L, -gam0 * invc
    if mode == 4:
        L = [(ams * (gam1 * v2 - ls * vn) - amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
             (ams * (gam0 * vn + ls), 1),
             (gam0 * ams * vt1 + amf * lf * bn1 * sbn, 2),
             (gam0 * ams * vt2 + amf * lf * bn2 * sbn, 3),
             (gam0 * ams * Bt1 - cface * amf * bn1 * srho, 5),
             (gam0 * ams * Bt2 - cface * amf * bn2 * srho, 6),
             (-gam0 * ams, 7)]
        scale = 0.5 * invc * jnp.where(geq, sbt, one_typed)
        return L, scale
    if mode == 5:
        L = [(bn2 * vt1 - bn1 * vt2, 0), (-bn2, 2), (bn1, 3),
             (bn2 * sbn * srho, 5), (-bn1 * sbn * srho, 6)]
        return L, 0.5 + zero_typed
    L = [(amf * (gam1 * v2 - lf * vn) + ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
         (amf * (gam0 * vn + lf), 1),
         (gam0 * amf * vt1 - ams * ls * bn1 * sbn, 2),
         (gam0 * amf * vt2 - ams * ls * bn2 * sbn, 3),
         (gam0 * amf * Bt1 + cface * ams * bn1 * srho, 5),
         (gam0 * amf * Bt2 + cface * ams * bn2 * srho, 6),
         (-gam0 * amf, 7)]
    scale = 0.5 * invc * jnp.where(~geq, sbt, one_typed)
    return L, scale


# -------- forward functionals (match the module) --------
def _Rcol_apply(mode, fp_base, scal_tuple):
    fpl = dict(fp_base)
    for key, val in zip(proj_keys, scal_tuple):
        fpl[key] = val
    R, scale = _Rcols(mode, fpl)
    ref = scal_tuple[0]
    return tuple(jnp.broadcast_to(R[slot] * scale, jnp.shape(ref)) for slot in range(ncomp))


def left_project_fwd(mode, values, fp):
    L, scale = _Lrows(mode, fp)
    acc = zero_typed
    for coeff, idx in L:
        acc = acc + coeff * values[idx]
    return acc * scale


# ============== HAND-DERIVED TRANSPOSES ==============
# We accumulate cotangents on the 16 proj_keys scalars.  The constant factors
# (sbn, sbt, geq) are held fixed (zero cotangent).  We exploit that each entry
# of R / each coefficient of L is an explicit elementwise algebraic expression
# in the scalars; we transpose term by term.

def _idx(key):
    return proj_keys.index(key)


def _rcol_apply_adj(mode, fp, rsbar):
    """Transpose of _Rcol_apply: given cotangents rsbar[slot] on out[slot]
    (= R[slot]*scale), return length-16 cotangent over proj_keys."""
    sbn = fp['sbn']; sbt = fp['sbt']; geq = fp['geq']
    vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
    v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
    lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
    bn1 = fp['bn1']; bn2 = fp['bn2']; srho = fp['srho']
    g = [zero_typed] * 16
    # scale (piecewise constant)
    if mode in (0, 6):
        scale = jnp.where(~geq, sbt, one_typed)
    elif mode in (2, 4):
        scale = jnp.where(geq, sbt, one_typed)
    else:
        scale = one_typed
    # cotangent on R[slot] = rsbar[slot]*scale
    rb = [rsbar[slot] * scale for slot in range(ncomp)]

    def add(key, val):
        g[_idx(key)] = g[_idx(key)] + val

    if mode == 0:
        # R0 = amf
        add('amf', rb[0])
        # R1 = amf*(vn-lf)
        add('amf', rb[1] * (vn - lf)); add('vn', rb[1] * amf); add('lf', rb[1] * (-amf))
        # R2 = amf*vt1 + ams*ls*bn1*sbn
        add('amf', rb[2] * vt1); add('vt1', rb[2] * amf)
        add('ams', rb[2] * ls * bn1 * sbn); add('ls', rb[2] * ams * bn1 * sbn); add('bn1', rb[2] * ams * ls * sbn)
        # R3 = amf*vt2 + ams*ls*bn2*sbn
        add('amf', rb[3] * vt2); add('vt2', rb[3] * amf)
        add('ams', rb[3] * ls * bn2 * sbn); add('ls', rb[3] * ams * bn2 * sbn); add('bn2', rb[3] * ams * ls * sbn)
        # R5 = cface*ams*bn1/srho
        add('cface', rb[5] * ams * bn1 / srho); add('ams', rb[5] * cface * bn1 / srho)
        add('bn1', rb[5] * cface * ams / srho); add('srho', rb[5] * (-cface * ams * bn1 / srho ** 2))
        # R6 = cface*ams*bn2/srho
        add('cface', rb[6] * ams * bn2 / srho); add('ams', rb[6] * cface * bn2 / srho)
        add('bn2', rb[6] * cface * ams / srho); add('srho', rb[6] * (-cface * ams * bn2 / srho ** 2))
        # R7 = amf*(lf^2 - lf*vn + 0.5*v2 - gam2*csq) + ams*ls*(bn1*vt1+bn2*vt2)*sbn
        E = (lf * lf - lf * vn + 0.5 * v2 - gam2 * csq)
        add('amf', rb[7] * E)
        add('lf', rb[7] * amf * (2.0 * lf - vn)); add('vn', rb[7] * amf * (-lf))
        add('v2', rb[7] * amf * 0.5); add('csq', rb[7] * amf * (-gam2))
        bsum = (bn1 * vt1 + bn2 * vt2)
        add('ams', rb[7] * ls * bsum * sbn); add('ls', rb[7] * ams * bsum * sbn)
        add('bn1', rb[7] * ams * ls * vt1 * sbn); add('vt1', rb[7] * ams * ls * bn1 * sbn)
        add('bn2', rb[7] * ams * ls * vt2 * sbn); add('vt2', rb[7] * ams * ls * bn2 * sbn)
    elif mode == 1:
        # R2=-bn2, R3=bn1, R5=-bn2*sbn/srho, R6=bn1*sbn/srho, R7=bn1*vt2-bn2*vt1
        add('bn2', rb[2] * (-1.0)); add('bn1', rb[3] * 1.0)
        add('bn2', rb[5] * (-sbn / srho)); add('srho', rb[5] * (bn2 * sbn / srho ** 2))
        add('bn1', rb[6] * (sbn / srho)); add('srho', rb[6] * (-bn1 * sbn / srho ** 2))
        add('bn1', rb[7] * vt2); add('vt2', rb[7] * bn1); add('bn2', rb[7] * (-vt1)); add('vt1', rb[7] * (-bn2))
    elif mode == 2:
        add('ams', rb[0])
        add('ams', rb[1] * (vn - ls)); add('vn', rb[1] * ams); add('ls', rb[1] * (-ams))
        add('ams', rb[2] * vt1); add('vt1', rb[2] * ams)
        add('amf', rb[2] * (-lf * bn1 * sbn)); add('lf', rb[2] * (-amf * bn1 * sbn)); add('bn1', rb[2] * (-amf * lf * sbn))
        add('ams', rb[3] * vt2); add('vt2', rb[3] * ams)
        add('amf', rb[3] * (-lf * bn2 * sbn)); add('lf', rb[3] * (-amf * bn2 * sbn)); add('bn2', rb[3] * (-amf * lf * sbn))
        # R5 = -cface*amf*bn1/srho
        add('cface', rb[5] * (-amf * bn1 / srho)); add('amf', rb[5] * (-cface * bn1 / srho))
        add('bn1', rb[5] * (-cface * amf / srho)); add('srho', rb[5] * (cface * amf * bn1 / srho ** 2))
        add('cface', rb[6] * (-amf * bn2 / srho)); add('amf', rb[6] * (-cface * bn2 / srho))
        add('bn2', rb[6] * (-cface * amf / srho)); add('srho', rb[6] * (cface * amf * bn2 / srho ** 2))
        E = (ls * ls - ls * vn + 0.5 * v2 - gam2 * csq)
        add('ams', rb[7] * E)
        add('ls', rb[7] * ams * (2.0 * ls - vn)); add('vn', rb[7] * ams * (-ls))
        add('v2', rb[7] * ams * 0.5); add('csq', rb[7] * ams * (-gam2))
        bsum = (bn1 * vt1 + bn2 * vt2)
        add('amf', rb[7] * (-lf * bsum * sbn)); add('lf', rb[7] * (-amf * bsum * sbn))
        add('bn1', rb[7] * (-amf * lf * vt1 * sbn)); add('vt1', rb[7] * (-amf * lf * bn1 * sbn))
        add('bn2', rb[7] * (-amf * lf * vt2 * sbn)); add('vt2', rb[7] * (-amf * lf * bn2 * sbn))
    elif mode == 3:
        # R = [1, vn, vt1, vt2, 0,0,0, 0.5*v2]
        add('vn', rb[1]); add('vt1', rb[2]); add('vt2', rb[3]); add('v2', rb[7] * 0.5)
    elif mode == 4:
        add('ams', rb[0])
        add('ams', rb[1] * (vn + ls)); add('vn', rb[1] * ams); add('ls', rb[1] * ams)
        add('ams', rb[2] * vt1); add('vt1', rb[2] * ams)
        add('amf', rb[2] * (lf * bn1 * sbn)); add('lf', rb[2] * (amf * bn1 * sbn)); add('bn1', rb[2] * (amf * lf * sbn))
        add('ams', rb[3] * vt2); add('vt2', rb[3] * ams)
        add('amf', rb[3] * (lf * bn2 * sbn)); add('lf', rb[3] * (amf * bn2 * sbn)); add('bn2', rb[3] * (amf * lf * sbn))
        add('cface', rb[5] * (-amf * bn1 / srho)); add('amf', rb[5] * (-cface * bn1 / srho))
        add('bn1', rb[5] * (-cface * amf / srho)); add('srho', rb[5] * (cface * amf * bn1 / srho ** 2))
        add('cface', rb[6] * (-amf * bn2 / srho)); add('amf', rb[6] * (-cface * bn2 / srho))
        add('bn2', rb[6] * (-cface * amf / srho)); add('srho', rb[6] * (cface * amf * bn2 / srho ** 2))
        E = (ls * ls + ls * vn + 0.5 * v2 - gam2 * csq)
        add('ams', rb[7] * E)
        add('ls', rb[7] * ams * (2.0 * ls + vn)); add('vn', rb[7] * ams * ls)
        add('v2', rb[7] * ams * 0.5); add('csq', rb[7] * ams * (-gam2))
        bsum = (bn1 * vt1 + bn2 * vt2)
        add('amf', rb[7] * (lf * bsum * sbn)); add('lf', rb[7] * (amf * bsum * sbn))
        add('bn1', rb[7] * (amf * lf * vt1 * sbn)); add('vt1', rb[7] * (amf * lf * bn1 * sbn))
        add('bn2', rb[7] * (amf * lf * vt2 * sbn)); add('vt2', rb[7] * (amf * lf * bn2 * sbn))
    elif mode == 5:
        add('bn2', rb[2] * (-1.0)); add('bn1', rb[3] * 1.0)
        add('bn2', rb[5] * (sbn / srho)); add('srho', rb[5] * (-bn2 * sbn / srho ** 2))
        add('bn1', rb[6] * (-sbn / srho)); add('srho', rb[6] * (bn1 * sbn / srho ** 2))
        add('bn1', rb[7] * vt2); add('vt2', rb[7] * bn1); add('bn2', rb[7] * (-vt1)); add('vt1', rb[7] * (-bn2))
    else:  # mode 6
        add('amf', rb[0])
        add('amf', rb[1] * (vn + lf)); add('vn', rb[1] * amf); add('lf', rb[1] * amf)
        add('amf', rb[2] * vt1); add('vt1', rb[2] * amf)
        add('ams', rb[2] * (-ls * bn1 * sbn)); add('ls', rb[2] * (-ams * bn1 * sbn)); add('bn1', rb[2] * (-ams * ls * sbn))
        add('amf', rb[3] * vt2); add('vt2', rb[3] * amf)
        add('ams', rb[3] * (-ls * bn2 * sbn)); add('ls', rb[3] * (-ams * bn2 * sbn)); add('bn2', rb[3] * (-ams * ls * sbn))
        add('cface', rb[5] * ams * bn1 / srho); add('ams', rb[5] * cface * bn1 / srho)
        add('bn1', rb[5] * cface * ams / srho); add('srho', rb[5] * (-cface * ams * bn1 / srho ** 2))
        add('cface', rb[6] * ams * bn2 / srho); add('ams', rb[6] * cface * bn2 / srho)
        add('bn2', rb[6] * cface * ams / srho); add('srho', rb[6] * (-cface * ams * bn2 / srho ** 2))
        E = (lf * lf + lf * vn + 0.5 * v2 - gam2 * csq)
        add('amf', rb[7] * E)
        add('lf', rb[7] * amf * (2.0 * lf + vn)); add('vn', rb[7] * amf * lf)
        add('v2', rb[7] * amf * 0.5); add('csq', rb[7] * amf * (-gam2))
        bsum = (bn1 * vt1 + bn2 * vt2)
        add('ams', rb[7] * (-ls * bsum * sbn)); add('ls', rb[7] * (-ams * bsum * sbn))
        add('bn1', rb[7] * (-ams * ls * vt1 * sbn)); add('vt1', rb[7] * (-ams * ls * bn1 * sbn))
        add('bn2', rb[7] * (-ams * ls * vt2 * sbn)); add('vt2', rb[7] * (-ams * ls * bn2 * sbn))
    return g


def _lrow_apply_adj(mode, fp, values, out_bar):
    """Transpose of left_project_fwd's dependence on the fp scalars only.
    Given scalar cotangent out_bar on the scalar output left_project_fwd(...),
    accumulate cotangents over the 16 proj_keys.  values is the (fixed) 8-vector.
    out = scale * sum_i coeff_i * values[idx_i].
    d out/d s = out_bar*[ dscale*S + scale*sum_i dcoeff_i*values[idx_i] ],
    where dscale/ds = 0 (piecewise const factor; invc IS differentiable though).
    """
    vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
    Bt1 = fp['Bt1']; Bt2 = fp['Bt2']
    v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
    lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
    bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
    srho = fp['srho']; geq = fp['geq']; invc = fp['invc']
    g = [zero_typed] * 16
    L, scale = _Lrows(mode, fp)
    S = zero_typed
    for coeff, idx in L:
        S = S + coeff * values[idx]

    def add(key, val):
        g[_idx(key)] = g[_idx(key)] + val

    # scale depends on invc for modes 0,2,3,4,6; alfven (1,5) scale const.
    # scale = 0.5*invc*const  (modes 0,2,4,6) ; -gam0*invc (mode 3); const (1,5)
    if mode in (0, 6):
        # scale = 0.5*invc*where(~geq,sbt,1)
        sc_const = 0.5 * jnp.where(~geq, sbt, one_typed)
        add('invc', out_bar * sc_const * S)
    elif mode in (2, 4):
        sc_const = 0.5 * jnp.where(geq, sbt, one_typed)
        add('invc', out_bar * sc_const * S)
    elif mode == 3:
        add('invc', out_bar * (-gam0) * S)
    # else mode 1,5: scale constant, no invc dep

    # cotangent on S: ob_S = out_bar*scale
    ob_S = out_bar * scale
    # dS/d(coeff_i) = values[idx_i]; we need dcoeff_i/d(scalar) per coeff.
    # cb_i := ob_S * values[idx_i]   is cotangent on coeff_i.
    cb = {idx: ob_S * values[idx] for (_c, idx) in L}

    if mode == 0:
        # coeff0(idx0): amf*(gam1*v2+lf*vn) - ams*ls*(bn1*vt1+bn2*vt2)*sbn
        c = cb[0]
        add('amf', c * (gam1 * v2 + lf * vn)); add('v2', c * amf * gam1); add('lf', c * amf * vn); add('vn', c * amf * lf)
        bsum = bn1 * vt1 + bn2 * vt2
        add('ams', c * (-ls * bsum * sbn)); add('ls', c * (-ams * bsum * sbn))
        add('bn1', c * (-ams * ls * vt1 * sbn)); add('vt1', c * (-ams * ls * bn1 * sbn))
        add('bn2', c * (-ams * ls * vt2 * sbn)); add('vt2', c * (-ams * ls * bn2 * sbn))
        # coeff1(idx1): amf*(gam0*vn - lf)
        c = cb[1]; add('amf', c * (gam0 * vn - lf)); add('vn', c * amf * gam0); add('lf', c * (-amf))
        # coeff2(idx2): gam0*amf*vt1 + ams*ls*bn1*sbn
        c = cb[2]; add('amf', c * gam0 * vt1); add('vt1', c * gam0 * amf)
        add('ams', c * ls * bn1 * sbn); add('ls', c * ams * bn1 * sbn); add('bn1', c * ams * ls * sbn)
        # coeff3(idx3): gam0*amf*vt2 + ams*ls*bn2*sbn
        c = cb[3]; add('amf', c * gam0 * vt2); add('vt2', c * gam0 * amf)
        add('ams', c * ls * bn2 * sbn); add('ls', c * ams * bn2 * sbn); add('bn2', c * ams * ls * sbn)
        # coeff(idx5): gam0*amf*Bt1 + cface*ams*bn1*srho   (Bt1 differentiable!)
        c = cb[5]; add('amf', c * gam0 * Bt1); add('Bt1', c * gam0 * amf)
        add('cface', c * ams * bn1 * srho); add('ams', c * cface * bn1 * srho)
        add('bn1', c * cface * ams * srho); add('srho', c * cface * ams * bn1)
        # coeff(idx6): gam0*amf*Bt2 + cface*ams*bn2*srho
        c = cb[6]; add('amf', c * gam0 * Bt2); add('Bt2', c * gam0 * amf)
        add('cface', c * ams * bn2 * srho); add('ams', c * cface * bn2 * srho)
        add('bn2', c * cface * ams * srho); add('srho', c * cface * ams * bn2)
        # coeff(idx7): -gam0*amf
        c = cb[7]; add('amf', c * (-gam0))
    elif mode == 1:
        c = cb[0]; add('bn2', c * vt1); add('vt1', c * bn2); add('bn1', c * (-vt2)); add('vt2', c * (-bn1))
        c = cb[2]; add('bn2', c * (-1.0))
        c = cb[3]; add('bn1', c * 1.0)
        c = cb[5]; add('bn2', c * (-sbn * srho)); add('srho', c * (-bn2 * sbn))
        c = cb[6]; add('bn1', c * (sbn * srho)); add('srho', c * (bn1 * sbn))
    elif mode == 2:
        c = cb[0]
        add('ams', c * (gam1 * v2 + ls * vn)); add('v2', c * ams * gam1); add('ls', c * ams * vn); add('vn', c * ams * ls)
        bsum = bn1 * vt1 + bn2 * vt2
        add('amf', c * lf * bsum * sbn); add('lf', c * amf * bsum * sbn)
        add('bn1', c * amf * lf * vt1 * sbn); add('vt1', c * amf * lf * bn1 * sbn)
        add('bn2', c * amf * lf * vt2 * sbn); add('vt2', c * amf * lf * bn2 * sbn)
        c = cb[1]; add('ams', c * (gam0 * vn - ls)); add('vn', c * ams * gam0); add('ls', c * (-ams))
        c = cb[2]; add('ams', c * gam0 * vt1); add('vt1', c * gam0 * ams)
        add('amf', c * (-lf * bn1 * sbn)); add('lf', c * (-amf * bn1 * sbn)); add('bn1', c * (-amf * lf * sbn))
        c = cb[3]; add('ams', c * gam0 * vt2); add('vt2', c * gam0 * ams)
        add('amf', c * (-lf * bn2 * sbn)); add('lf', c * (-amf * bn2 * sbn)); add('bn2', c * (-amf * lf * sbn))
        c = cb[5]; add('ams', c * gam0 * Bt1); add('Bt1', c * gam0 * ams)
        add('cface', c * (-amf * bn1 * srho)); add('amf', c * (-cface * bn1 * srho))
        add('bn1', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn1))
        c = cb[6]; add('ams', c * gam0 * Bt2); add('Bt2', c * gam0 * ams)
        add('cface', c * (-amf * bn2 * srho)); add('amf', c * (-cface * bn2 * srho))
        add('bn2', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn2))
        c = cb[7]; add('ams', c * (-gam0))
    elif mode == 3:
        c = cb[0]; add('csq', c * (-1.0 / gam0)); add('v2', c * (-0.5))
        c = cb[1]; add('vn', c)
        c = cb[2]; add('vt1', c)
        c = cb[3]; add('vt2', c)
        c = cb[5]; add('Bt1', c)
        c = cb[6]; add('Bt2', c)
        # cb[7] coeff is constant -1 -> no scalar dep
    elif mode == 4:
        c = cb[0]
        add('ams', c * (gam1 * v2 - ls * vn)); add('v2', c * ams * gam1); add('ls', c * (-ams * vn)); add('vn', c * (-ams * ls))
        bsum = bn1 * vt1 + bn2 * vt2
        add('amf', c * (-lf * bsum * sbn)); add('lf', c * (-amf * bsum * sbn))
        add('bn1', c * (-amf * lf * vt1 * sbn)); add('vt1', c * (-amf * lf * bn1 * sbn))
        add('bn2', c * (-amf * lf * vt2 * sbn)); add('vt2', c * (-amf * lf * bn2 * sbn))
        c = cb[1]; add('ams', c * (gam0 * vn + ls)); add('vn', c * ams * gam0); add('ls', c * ams)
        c = cb[2]; add('ams', c * gam0 * vt1); add('vt1', c * gam0 * ams)
        add('amf', c * lf * bn1 * sbn); add('lf', c * amf * bn1 * sbn); add('bn1', c * amf * lf * sbn)
        c = cb[3]; add('ams', c * gam0 * vt2); add('vt2', c * gam0 * ams)
        add('amf', c * lf * bn2 * sbn); add('lf', c * amf * bn2 * sbn); add('bn2', c * amf * lf * sbn)
        c = cb[5]; add('ams', c * gam0 * Bt1); add('Bt1', c * gam0 * ams)
        add('cface', c * (-amf * bn1 * srho)); add('amf', c * (-cface * bn1 * srho))
        add('bn1', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn1))
        c = cb[6]; add('ams', c * gam0 * Bt2); add('Bt2', c * gam0 * ams)
        add('cface', c * (-amf * bn2 * srho)); add('amf', c * (-cface * bn2 * srho))
        add('bn2', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn2))
        c = cb[7]; add('ams', c * (-gam0))
    elif mode == 5:
        c = cb[0]; add('bn2', c * vt1); add('vt1', c * bn2); add('bn1', c * (-vt2)); add('vt2', c * (-bn1))
        c = cb[2]; add('bn2', c * (-1.0))
        c = cb[3]; add('bn1', c * 1.0)
        c = cb[5]; add('bn2', c * (sbn * srho)); add('srho', c * (bn2 * sbn))
        c = cb[6]; add('bn1', c * (-sbn * srho)); add('srho', c * (-bn1 * sbn))
    else:  # mode 6
        c = cb[0]
        add('amf', c * (gam1 * v2 - lf * vn)); add('v2', c * amf * gam1); add('lf', c * (-amf * vn)); add('vn', c * (-amf * lf))
        bsum = bn1 * vt1 + bn2 * vt2
        add('ams', c * ls * bsum * sbn); add('ls', c * ams * bsum * sbn)
        add('bn1', c * ams * ls * vt1 * sbn); add('vt1', c * ams * ls * bn1 * sbn)
        add('bn2', c * ams * ls * vt2 * sbn); add('vt2', c * ams * ls * bn2 * sbn)
        c = cb[1]; add('amf', c * (gam0 * vn + lf)); add('vn', c * amf * gam0); add('lf', c * amf)
        c = cb[2]; add('amf', c * gam0 * vt1); add('vt1', c * gam0 * amf)
        add('ams', c * (-ls * bn1 * sbn)); add('ls', c * (-ams * bn1 * sbn)); add('bn1', c * (-ams * ls * sbn))
        c = cb[3]; add('amf', c * gam0 * vt2); add('vt2', c * gam0 * amf)
        add('ams', c * (-ls * bn2 * sbn)); add('ls', c * (-ams * bn2 * sbn)); add('bn2', c * (-ams * ls * sbn))
        c = cb[5]; add('amf', c * gam0 * Bt1); add('Bt1', c * gam0 * amf)
        add('cface', c * ams * bn1 * srho); add('ams', c * cface * bn1 * srho)
        add('bn1', c * cface * ams * srho); add('srho', c * cface * ams * bn1)
        c = cb[6]; add('amf', c * gam0 * Bt2); add('Bt2', c * gam0 * amf)
        add('cface', c * ams * bn2 * srho); add('ams', c * cface * bn2 * srho)
        add('bn2', c * cface * ams * srho); add('srho', c * cface * ams * bn2)
        c = cb[7]; add('amf', c * (-gam0))
    return g


def relerr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-30)


def main():
    rng = np.random.default_rng(1)
    ok = True
    worstR = 0.0
    worstL = 0.0
    for trial in range(20):
        fp = make_fp(rng)
        scal_vec = tuple(fp[k] for k in proj_keys)
        values = jnp.asarray(rng.normal(size=8))
        for mode in range(7):
            # --- R_col ---
            f_r = lambda sv: _Rcol_apply(mode, fp, sv)
            out, vjp = jax.vjp(f_r, scal_vec)
            rsbar = tuple(jnp.asarray(rng.normal()) for _ in range(8))
            (gref,) = vjp(rsbar)
            ghand = _rcol_apply_adj(mode, fp, list(rsbar))
            worstR = max(worstR, relerr(jnp.asarray(gref), jnp.asarray(ghand)))
            # --- L_row ---
            f_l = lambda sv: left_project_fwd(mode, values, {**fp, **dict(zip(proj_keys, sv))})
            out2, vjp2 = jax.vjp(f_l, scal_vec)
            ob = jnp.asarray(rng.normal())
            (gref2,) = vjp2(ob)
            ghand2 = _lrow_apply_adj(mode, fp, values, ob)
            worstL = max(worstL, relerr(jnp.asarray(gref2), jnp.asarray(ghand2)))
    print(f"R_col adj  worst rel = {worstR:.2e}")
    print(f"L_row adj  worst rel = {worstL:.2e}")
    ok = worstR < 1e-11 and worstL < 1e-11
    print("SUMMARY:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
