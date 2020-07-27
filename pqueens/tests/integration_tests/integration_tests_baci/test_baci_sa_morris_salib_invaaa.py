"""
Test suite for integration tests for the Morris-Salib Iterator (Elementary Effects) for local
simulations with BACI using the INVAAA minimal model.
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pqueens.main import main
from pqueens.utils import injector


def test_baci_morris_salib(
    inputdir, tmpdir, third_party_inputs, config_dir, set_baci_links_for_gitlab_runner
):
    """Test a morris-salib run with a small BACI simulation model"""
    template = os.path.join(inputdir, "morris_baci_local_invaaa_template.json")
    input_file = os.path.join(tmpdir, "morris_baci_local_invaaa.json")
    third_party_input_file = os.path.join(third_party_inputs, "baci_input_files", "invaaa_ee.dat")

    baci_release = os.path.join(config_dir, "baci-release")
    post_drt_monitor = os.path.join(config_dir, "post_drt_monitor")

    # check if symbolic links are existent
    if (not os.path.islink(baci_release)) or (not os.path.islink(post_drt_monitor)):
        # set default baci location for testing machine
        dst_baci, dst_drt_monitor, src_baci, src_drt_monitor = set_baci_links_for_gitlab_runner
        try:
            os.symlink(src_baci, dst_baci)
            os.symlink(src_drt_monitor, dst_drt_monitor)
        except FileNotFoundError:
            raise FileNotFoundError(
                'No working baci-release or post_drt_monitor could be found! '
                'Make sure an appropriate symbolic link is made available '
                'under the config directory! \n'
                'You can create the symbolic links on Linux via:\n'
                '-------------------------------------------------------------------------\n'
                'ln -s <path/to/baci-release> <QUEENS_BaseDir>/config/baci-release\n'
                'ln -s <path/to/post_drt_monitor> <QUEENS_BaseDir>/config/post_drt_monitor\n'
                '-------------------------------------------------------------------------\n'
            )

    dir_dict = {
        'experiment_dir': str(tmpdir),
        'baci_input': third_party_input_file,
        'baci-release': baci_release,
        'post_drt_monitor': post_drt_monitor,
    }

    injector.inject(dir_dict, template, input_file)
    arguments = ['--input=' + input_file, '--output=' + str(tmpdir)]
    main(arguments)

    result_file = os.path.join(tmpdir, 'ee_invaaa_local.pickle')
    with open(result_file, 'rb') as handle:
        results = pickle.load(handle)

    # test results of SA analysis
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu"],
        np.array([-1.361395,  0.836351]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star"],
        np.array([1.361395, 0.836351]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["sigma"],
        np.array([0.198629, 0.198629]),
        rtol=1.0e-3,
    )
    np.testing.assert_allclose(
        results["sensitivity_indices"]["mu_star_conf"],
        np.array([0.11853, 0.146817]),
        rtol=1.0e-3,
    )

    # test if figure is shown
    # TODO: this should be tested more intensely after plotting capabilities are moved to
    #  visualization tool
    num_figure = plt.gcf().number
    assert num_figure == 1

    # test if figure was saved
    figure_path = os.path.join(tmpdir, 'ee_invaaa_local_result.png')
    assert os.path.isfile(figure_path)





