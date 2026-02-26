import axon_velocity as av
from pathlib import Path
import numpy as np
import pytest

toy_data_folder = Path(__file__).parent / "data" / "toy"
FS = 32000


@pytest.fixture(scope="module")
def template_bifurcation():
    return np.load(toy_data_folder / "bifurcation" / "template.npy")


@pytest.fixture(scope="module")
def locations_bifurcation():
    return np.load(toy_data_folder / "bifurcation" / "locations.npy")


@pytest.fixture(scope="module")
def template_sinusoidal():
    return np.load(toy_data_folder / "sinusoidal" / "template.npy")


@pytest.fixture(scope="module")
def locations_sinusoidal():
    return np.load(toy_data_folder / "sinusoidal" / "locations.npy")


@pytest.fixture(scope="module")
def gtr_bif(template_bifurcation, locations_bifurcation):
    return av.compute_graph_propagation_velocity(
        template_bifurcation, locations_bifurcation, FS, verbose=True
    )


@pytest.fixture(scope="module")
def gtr_sin(template_sinusoidal, locations_sinusoidal):
    return av.compute_graph_propagation_velocity(
        template_sinusoidal, locations_sinusoidal, FS, verbose=True
    )


def _assert_branch_keys(branch):
    for key in ("channels", "r2", "velocity", "peak_times", "distances", "offset"):
        assert key in branch


def test_axon_velocity_bifurcation(gtr_bif, locations_bifurcation):
    assert len(gtr_bif.branches) > 0
    for branch in gtr_bif.branches:
        _assert_branch_keys(branch)
    assert len(locations_bifurcation) > len(gtr_bif.selected_channels)


def test_axon_velocity_sinusoidal(gtr_sin, locations_bifurcation):
    assert len(gtr_sin.branches) > 0
    for branch in gtr_sin.branches:
        _assert_branch_keys(branch)
    assert len(locations_bifurcation) > len(gtr_sin.selected_channels)


def test_plotting(template_bifurcation, locations_bifurcation, gtr_bif):
    av.plot_template(template_bifurcation, locations_bifurcation)
    av.plot_peak_latency_map(template_bifurcation, locations_bifurcation, FS)
    av.plot_peak_latency_map(template_bifurcation, locations_bifurcation, FS, plot_image=False)
    av.plot_amplitude_map(template_bifurcation, locations_bifurcation)
    av.plot_amplitude_map(template_bifurcation, locations_bifurcation, plot_image=False)
    av.plot_peak_std_map(template_bifurcation, locations_bifurcation, FS)
    av.plot_peak_std_map(template_bifurcation, locations_bifurcation, FS, plot_image=False)
    av.plot_branch_velocities(gtr_bif.branches)
    av.plot_axon_summary(gtr_bif)

    for br in gtr_bif.branches:
        av.plot_velocity(br["peak_times"], br["distances"], br["velocity"], br["offset"])
        av.plot_template_propagation(template_bifurcation, locations_bifurcation, br["channels"])
