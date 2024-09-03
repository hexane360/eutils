
import sys
import typing as t

import click

import eutils


@click.group('eutils')
def main():
    """Electron utilities"""
    ...


@main.command("wavelength")
@click.argument("energy", type=float)
def wavelength(energy: float):
    """Get the wavelength for an electron of ENERGY keV"""
    electron = _make_electron(energy)
    _print_value("Wavelength", f"{electron.wavelength:.4f}", "Å")


@main.command("sigma")
@click.argument("energy", type=float)
def interaction_param(energy: float):
    """Get the interaction parameter for an electron of ENERGY keV"""
    electron = _make_electron(energy)
    _print_value("Interaction parameter (sigma)", f"{electron.interaction_param:.6f}", "rad/V-Å")


@main.command("bwlim")
@click.argument("energy", type=float)
@click.option("-n", type=int, required=True, help="# of samples")
@click.option("--extent", type=float, required=True, help="Cell extent [Å]")
def bandwidth_limit(energy: float, n: int, extent: float):
    """Calculate the bandwidth limit for a simulation"""
    electron = _make_electron(energy)
    wavelength = electron.wavelength

    sampling = extent / n
    nyquist = 1 / (2 * sampling)
    bwlim = nyquist * 2 /3

    print(f"Nyquist frequency: {nyquist * wavelength * 1e3:8.2f} mrad ({nyquist:.3f} 1/Å)")
    print(f"  Bandwidth limit: {bwlim * wavelength * 1e3:8.2f} mrad ({bwlim:.3f} 1/Å)")


def _print_value(name: str, val: str, unit: str, file: t.TextIO = sys.stdout):
    if file.isatty():
        print(f"{name}: {val} [{unit}]", file=file)
    else:
        print(val, file=file)


def _make_electron(energy: float) -> eutils.Electron:
    print(f"Electron energy: {energy:.1f} keV", file=sys.stderr)
    return eutils.Electron(energy)