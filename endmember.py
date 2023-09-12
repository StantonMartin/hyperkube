import numpy as np
import os
import os.path as osp
import cProfile, pstats

import pysptools.util as util
import pysptools.eea as eea
import pysptools.abundance_maps as amp

import numpy as np


_doProfile = False


def profile():
    if _doProfile == True:
        pr = cProfile.Profile()
        pr.enable()
        return pr


def stat(pr):
    if _doProfile == True:
        pr.disable()
        ps = pstats.Stats(pr)
        ps.strip_dirs()
        ps.sort_stats("time")
        ps.print_stats()


def ROI(data, path):
    r = util.ROIs(data.shape[0], data.shape[1])

    r.add("Area", {"rec": (30, 30, 100, 100)})
    r.plot(path, colorMap="Accent")
    return r


def makeaxes(wvl):
    axes = {}
    for c in wvl:
        axes["wavelength"] = wvl
        axes["x"] = "Wavelength"
        axes["y"] = "Reflectance"

    return axes


def test_amap(data, U, umix_source, path, mask, amaps=None):
    if amaps == None:
        test_UCLS(data, U, umix_source, mask, path)
        test_NNLS(data, U, umix_source, mask, path)
        test_FCLS(data, U, umix_source, mask, path)
    else:
        if "UCLS" in amaps:
            test_UCLS(data, U, umix_source, mask, path)
        if "NNLS" in amaps:
            test_NNLS(data, U, umix_source, mask, path)
        if "FCLS" in amaps:
            test_FCLS(data, U, umix_source, mask, path)


def test_UCLS(data, U, umix_source, mask, path):
    print("  Testing UCLS")
    ucls = amp.UCLS()
    pr = profile()
    if mask is None:
        amap = ucls.map(data, U, normalize=True)
    else:
        amap = ucls.map(data, U, normalize=True, mask=mask)
    stat(pr)
    print(str(ucls))
    ucls.plot(path, suffix=umix_source)
    if mask is None:
        ucls.plot(path, colorMap="jet", suffix=umix_source + "_mask")
    else:
        ucls.plot(path, mask=mask, colorMap="jet", suffix=umix_source + "_mask")
    ucls.plot(path, interpolation="spline36", suffix=umix_source + "_spline36")

    if mask is None:
        ucls.plot(
            path, interpolation="spline36", columns=2, suffix=umix_source + "_spline36"
        )
        ucls.plot(path, interpolation="spline36", suffix=umix_source + "_spline36")
    else:
        ucls.plot(
            path,
            mask=mask,
            interpolation="spline36",
            columns=2,
            suffix=umix_source + "_spline36",
        )
        ucls.plot(
            path, mask=mask, interpolation="spline36", suffix=umix_source + "_spline36"
        )


def test_NNLS(data, U, umix_source, mask, path):
    print("  Testing NNLS")
    nnls = amp.NNLS()
    pr = profile()
    amap = nnls.map(data, U, normalize=True)
    stat(pr)
    print(str(nnls))
    nnls.plot(path, colorMap="jet", suffix=umix_source)
    if mask is None:
        nnls.plot(path, colorMap="jet", suffix=umix_source + "_mask")
    else:
        nnls.plot(path, mask=mask, colorMap="jet", suffix=umix_source + "_mask")
    nnls.plot(path, interpolation="spline36", suffix=umix_source + "_spline36")


def test_FCLS(data, U, umix_source, mask, path):
    print("  Testing FCLS")
    fcls = amp.FCLS()
    pr = profile()
    if mask is None:
        amap = fcls.map(data, U, normalize=True)
    else:
        amap = fcls.map(data, U, normalize=True, mask=mask)
    stat(pr)
    print(str(fcls))
    fcls.plot(path, suffix=umix_source)


def test_PPI(data, wvl, path, mask=None):
    print("Testing PPI")
    ppi = eea.PPI()
    pr = profile()
    if mask is None:
        U = ppi.extract(data, 4, normalize=True)
    else:
        U = ppi.extract(data, 4, normalize=True, mask=mask)
    print(str(ppi))

    stat(pr)
    print("  End members indexes:", ppi.get_idx())
    ppi.plot(path, axes=wvl, suffix="test1")
    ppi.plot(path, suffix="test2")
    U = U[[0, 1], :]
    test_amap(data, U, "PPI", path, mask, amaps="UCLS")
    test_amap(data, U, "PPI", path, mask, amaps="NNLS")
    test_amap(data, U, "PPI", path, mask, amaps="FCLS")


def test_ATGP(data, wvl, path, mask=None):
    print("Testing ATGP")
    atgp = eea.ATGP()
    pr = profile()
    if mask is None:
        U = atgp.extract(data, 8, normalize=True)
    else:
        U = atgp.extract(data, 8, normalize=True, mask=mask)
    stat(pr)
    print(str(atgp))
    print("  End members indexes:", atgp.get_idx())
    atgp.plot(path, axes=wvl, suffix="test1")
    atgp.plot(path, suffix="test2")
    U = U[[0, 1], :]
    test_amap(data, U, "ATGP", path, mask, amaps="UCLS")


def test_FIPPI(data, wvl, path, mask=None):
    print("Testing FIPPI")
    fippi = eea.FIPPI()
    pr = profile()
    if mask is None:
        U = fippi.extract(data, 4, 1, normalize=True)
    else:
        U = fippi.extract(data, 4, 1, normalize=True, mask=mask)

    print(str(fippi))
    stat(pr)
    print("  End members indexes:", fippi.get_idx())
    fippi.plot(path, axes=wvl, suffix="test1")
    fippi.plot(path, suffix="test2")
    test_amap(data, U, "FIPPI", path, mask, amaps="NNLS")


def test_NFINDR(data, wvl, path, mask=None):
    print("Testing NFINDR")
    nfindr = eea.NFINDR()
    pr = profile()
    if mask is None:
        U = nfindr.extract(data, 8, maxit=5, normalize=False, ATGP_init=False)
    else:
        U = nfindr.extract(
            data, 8, maxit=5, normalize=False, ATGP_init=False, mask=mask
        )

    stat(pr)
    print(str(nfindr))
    print("  Iterations:", nfindr.get_iterations())
    print("  End members indexes:", nfindr.get_idx())
    nfindr.plot(path, axes=wvl, suffix="test1")
    nfindr.plot(path, suffix="test2")
    U = U[[0, 1], :]
    test_amap(data, U, "NFINDR", path, mask, amaps="UCLS")
    test_amap(data, U, "NFINDR", path, mask, amaps="NNLS")
    test_amap(data, U, "NFINDR", path, mask, amaps="FCLS")
