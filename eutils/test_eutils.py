
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from eutils import C, Electron, Sampling


def test_electron_200keV():
    e = Electron(200.)

    d = {
        'energy': 200.,                # given
        'rest_energy': 510.99906,      # from constant
        'total_energy': 710.99906,     # TE = KE + RE
        'mass': 1.267471e-30,          # TE/c^2
        'rest_mass': 9.1093837e-31,    # from constant
        'momentum': 494.3679034888,    # sqrt(KE^2 + 2*KE*RE)
        'wavelength': 0.02507934741,   # h*c/momentum
        'gamma': 1.391390152459,       # long calculation
        'beta': 0.69531442627,         # sqrt(1 - gamma^-2)
        'velocity': 208450020.93,      # beta * c
        'interaction_param': 7.2883988338e-4, # 2 pi m e lambda / h^2
    }

    for (k, v) in d.items():
        assert pytest.approx(v, rel=1e-10) == getattr(e, k)


def test_electron_0keV():
    e = Electron(0.)

    d = {
        'energy': 0.,                  # given
        'rest_energy': 510.99906,      # from constant
        'total_energy': 510.99906,     # TE = KE + RE
        'mass': 9.1093837e-31,         # TE/c^2
        'rest_mass': 9.1093837e-31,    # from constant
        'momentum': 0.,                # KE = 0
        'gamma': 1.,
        'beta': 0.,                    # v = 0
        'velocity': 0.,                # v = 0
    }

    for (k, v) in d.items():
        assert pytest.approx(v, rel=1e-10) == getattr(e, k)

    with pytest.raises(ZeroDivisionError):
        e.wavelength

    with pytest.raises(ZeroDivisionError):
        e.interaction_param


def test_sampling():
    samp = Sampling((4, 5), extent=(4., 5.))

    assert_array_equal(samp.sampling, [1., 1.])
    assert_array_equal(samp.shape, [4, 5])
    assert_array_equal(samp.extent, [4., 5.])

    (yy, xx) = samp.real_grid()
    assert all(yy.shape == xx.shape == samp.shape)  # type: ignore
    assert_array_equal(yy, [[a] * 5 for a in [0., 1., 2., 3.]])
    assert_array_equal(xx, [[0., 1., 2., 3., 4.]] * 4)

    (ky, kx) = samp.recip_grid()
    assert all(ky.shape == kx.shape == samp.shape)  # type: ignore
    assert_array_equal(ky, [[a] * 5 for a in [0.00, 0.25, -0.50, -0.25]])
    assert_array_equal(kx, [[0., 0.2, 0.4, -0.4, -0.2]] * 4)

    (ky_centered, kx_centered) = samp.recip_grid(centered=True)
    assert_array_equal(ky_centered, [[a] * 5 for a in [-0.50, -0.25, 0.00, 0.25]])
    assert_array_equal(kx_centered, [[-0.4, -0.2, 0.0, 0.2, 0.4]] * 4)

    assert samp.bwlim(Electron(200.).wavelength) == pytest.approx(0.008359782470032896)
    assert samp.k_max == pytest.approx([1/2.]*2)

    assert_array_almost_equal(samp.mpl_real_extent(False), (0., 5., 4., 0.))
    assert_array_almost_equal(samp.mpl_real_extent(True), (-0.5, 4.5, 3.5, -0.5))
    # ky grid is shifted by half pixel
    assert_array_almost_equal(samp.mpl_recip_extent(False), (-0.4, 0.6, 0.5, -0.5))
    # shifts cancel out => kx grid is shifted
    assert_array_almost_equal(samp.mpl_recip_extent(True), (-0.5, 0.5, 0.375, -0.625))