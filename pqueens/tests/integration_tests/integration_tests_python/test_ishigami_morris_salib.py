# import os
# import pickle
#
# import pytest
#
# from pqueens.main import main
#
## TODO fix these test, because as of now these test produce platform dependent
## resutls
# def test_ishigami_morris_salib(inputdir, tmpdir):
#    """ Test case for salib based morris iterator """
#    arguments = [
#        '--input=' + os.path.join(inputdir, 'ishigami_morris_salib.json'),
#        '--output=' + str(tmpdir),
#    ]
#
#    main(arguments)
#    result_file = str(tmpdir) + '/' + 'xxx.pickle'
#    with open(result_file, 'rb') as handle:
#        results = pickle.load(handle)
#
#    print(results)
#
#    assert results["sensitivity_indices"]['mu'][0] == pytest.approx(-2.41504207e01)
#    assert results["sensitivity_indices"]['mu'][1] == pytest.approx(1.26225897e01)
#    assert results["sensitivity_indices"]['mu'][2] == pytest.approx(-5.70363996e-12)
#
#    assert results["sensitivity_indices"]['mu_star'][0] == pytest.approx(2.41504207e01)
#    assert results["sensitivity_indices"]['mu_star'][1] == pytest.approx(2.78976691e01)
#    assert results["sensitivity_indices"]['mu_star'][2] == pytest.approx(5.70363996e-12)
#
#    assert results["sensitivity_indices"]['sigma'][0] == pytest.approx(1.65750124e01)
#    assert results["sensitivity_indices"]['sigma'][1] == pytest.approx(2.92131768e01)
#    assert results["sensitivity_indices"]['sigma'][2] == pytest.approx(2.88449896e-12)
#
#    assert results["sensitivity_indices"]['mu_star_conf'][0] == pytest.approx(13.26737536608863)
#    assert results["sensitivity_indices"]['mu_star_conf'][1] == pytest.approx(4.49548691231593)
#    assert results["sensitivity_indices"]['mu_star_conf'][2] == pytest.approx(
#        2.4516069312374997e-12
#    )