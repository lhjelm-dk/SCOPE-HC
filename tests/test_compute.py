import numpy as np

from scopehc.compute import compute_results


def test_compute_results_units_and_finiteness():
    n = 1000
    res = compute_results(
        GRV_m3=np.full(n, 1_000_000.0),
        NtG=np.full(n, 0.7),
        Por=np.full(n, 0.2),
        RF_oil=np.full(n, 0.3),
        RF_gas=np.full(n, 0.6),
        Bg_rb_per_scf=np.full(n, 0.005),
        InvBo_STB_per_rb=np.full(n, 1 / 1.2),
        GOR_scf_per_STB=np.full(n, 100.0),
        gas_scf_per_boe=6000.0,
        f_oil=np.full(n, 0.6),
    )

    oil_stb = res["Oil_STB_rec"]
    gas_free = res["Gas_free_scf_rec"]
    gas_assoc = res["Gas_assoc_scf_rec"]
    thr = res["Total_surface_BOE"]

    assert np.all(oil_stb > 0)
    assert np.all(gas_free >= 0)
    assert np.allclose(gas_assoc, oil_stb * 100.0, rtol=1e-6)

    expected_thr = oil_stb + (gas_free + gas_assoc) / 6000.0
    assert np.allclose(thr, expected_thr, rtol=1e-6)

    for key, values in res.items():
        assert np.all(np.isfinite(values)), f"{key} contains non-finite values"

