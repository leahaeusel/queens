def paraboloid(x1, x2):
    """ A paraboloid

    see  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Args:
        x1 (float):  Input one
        x2 (float):  Input two

    Returns:
        float: Value of function
    """

    a = 1.
    b = 2.5
    return (x1 - a)**2 + (x2 - b)**2

def main(job_id, params):
    """ Interface to Paraboloid function

    Args:
        job_id (int):   ID of job
        params (dict):  Dictionary with parameters

    Returns:
        float:          Value of paraboloid function at parameters
                        specified in input dict
    """
    return paraboloid(params['x1'], params['x2'])
