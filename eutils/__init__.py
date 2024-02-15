
import math
from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike


class _Constants():
    e_rest_energy: float = 510.99906
    """Electron rest energy [keV] or mass [keV/c^2]."""
    e_rest_mass: float = 9.1093837e-31
    """Electron rest mass [kg]"""
    e_spin: float = 3.2910598e-19
    """Electron spin [keV-s]"""

    h: float = 4.135667696e-18
    """Planck's constant [keV-s]"""
    c = 299792458
    """Speed of light [m/s]"""
    hc: float = 12.3984244
    """Planck's constant * speed of light [keV-angstrom]"""

    e: float = 1.60217662e-19
    """Elementary charge [C]."""


C: _Constants = _Constants()


@dataclass(frozen=True)
class Electron:
    energy: float
    """Electron kinetic energy [keV]"""

    @property
    def rest_energy(self) -> float:
        """Electron rest energy (m_0c^2) [keV]."""
        return C.e_rest_energy

    @property
    def total_energy(self) -> float:
        """Total electron energy (mc^2) [keV]."""
        return self.energy + C.e_rest_energy

    @property
    def mass(self) -> float:
        """Electron mass [kg]."""
        return self.gamma * C.e_rest_mass

    @property
    def rest_mass(self) -> float:
        """Electron rest mass [kg]."""
        return C.e_rest_mass

    @property
    def momentum(self) -> float:
        """Electron momentum [keV-angstrom]."""
        return math.sqrt(self.energy * (2*C.e_rest_energy + self.energy))

    @property
    def wavelength(self) -> float:
        """Electron wavelength [angstrom]."""
        return C.hc / self.momentum

    @property
    def gamma(self) -> float:
        """Electron Lorentz factor (gamma) [unitless]."""
        return self.energy / C.e_rest_energy + 1.

    @property
    def beta(self) -> float:
        """Electron beta factor (v/c) [unitless]."""
        return math.sqrt(1 - self.gamma**-2)

    @property
    def velocity(self) -> float:
        """Electron velocity [m/s]."""
        return self.beta * C.c

    @property
    def interaction_param(self) -> float:
        """Electron interaction parameter (sigma) [radians/V-angstrom]"""
        m0_h2 = (C.e_rest_energy / C.hc**2)*1e-3  # RM/h^2 = RE/(hc)^2 [1/(eV angstrom^2)]
        return 2*math.pi * self.wavelength * (self.gamma * m0_h2)


@dataclass(frozen=True, init=False)
class Sampling:
    shape: NDArray[numpy.int_]
    """Sampling shape (n_y, n_x)"""
    extent: NDArray[numpy.float_]
    """Sampling extent (b, a)"""
    sampling: NDArray[numpy.float_]
    """Sample spacing (s_y, s_x)"""

    @property
    def k_max(self) -> NDArray[numpy.float_]:
        """
        Return maximum frequency (radius) of reciprocal space (1/(2s_y), 1/(2s_x))
        """
        return 1/(2 * self.sampling)

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: t.Tuple[float, float],
                 sampling: None = None):
        ...

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: None = None,
                 sampling: t.Tuple[float, float]):
        ...

    def __init__(self,
                 shape: ArrayLike, *,
                 extent: t.Optional[ArrayLike] = None,
                 sampling: t.Optional[ArrayLike] = None):
        try:
            object.__setattr__(self, 'shape', numpy.broadcast_to(shape, (2,)).astype(numpy.int_))
        except ValueError as e:
            raise ValueError(f"Expected a shape (n_y, n_x), instead got: {shape}") from e

        if extent is not None:
            try:
                object.__setattr__(self, 'extent', numpy.broadcast_to(extent, (2,)).astype(numpy.float_))
            except ValueError as e:
                raise ValueError(f"Expected an extent (b, a), instead got: {extent}") from e
            object.__setattr__(self, 'sampling', self.extent / self.shape)
        elif sampling is not None:
            try:
                object.__setattr__(self, 'sampling', numpy.broadcast_to(sampling, (2,)).astype(numpy.float_))
            except ValueError as e:
                raise ValueError(f"Expected a sampling (s_y, s_x), instead got: {sampling}") from e
            object.__setattr__(self, 'extent', self.sampling * self.shape)
        else:
            raise ValueError("Either 'extent' or 'sampling' must be specified")

    def real_grid(self) -> t.Tuple[NDArray[numpy.float_], NDArray[numpy.float_]]:
        """Return the realspace sampling grid `(yy, xx)`. Top left corner is `(0, 0)`"""
        ys = numpy.linspace(0., self.extent[0], self.shape[0], endpoint=False)
        xs = numpy.linspace(0., self.extent[1], self.shape[1], endpoint=False)
        return tuple(numpy.meshgrid(ys, xs, indexing='ij'))  # type: ignore

    def recip_grid(self, centered: bool = False) -> t.Tuple[NDArray[numpy.float_], NDArray[numpy.float_]]:
        """
        Return the reciprocal space sampling grid `(kyy, kxx)`.

        Unless `centered` is specified, the grid is fftshifted so the zero-frequency component is in the top left.
        """
        ky = numpy.fft.fftfreq(self.shape[0], self.sampling[0])
        kx = numpy.fft.fftfreq(self.shape[1], self.sampling[1])
        if centered:
            ky = numpy.fft.fftshift(ky)
            kx = numpy.fft.fftshift(kx)
        return tuple(numpy.meshgrid(ky, kx, indexing='ij'))  # type: ignore

    def bwlim(self, wavelength: float) -> float:
        """Return the bandwidth limit (in radians) for this sampling grid with the given wavelength."""
        return float(numpy.min(self.k_max) * 2./3. * wavelength)

    def mpl_real_extent(self, center: bool = False) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of real space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified, samples correspond to the center of pixels.
        Otherwise (the default), they correspond to the corners of pixels.
        """
        # shift pixel corners to centers
        shift = -self.sampling / 2. * int(center)
        return (shift[1], self.extent[1] + shift[1], self.extent[0] + shift[0], shift[0])

    def mpl_recip_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of reciprocal space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        kmax = self.k_max
        hp = 1/(2. * self.extent)
        # for odd sampling, grid is shifted by 1/2 pixel
        shift = hp * (self.shape % 2)  
        # shift pixel corners to centers
        if center:
            shift -= hp
        return (-kmax[1] + shift[1], kmax[1] + shift[1], kmax[0] + shift[0], -kmax[0] + shift[0])


__all__ = [
    'C', 'Electron', 'Sampling',
]