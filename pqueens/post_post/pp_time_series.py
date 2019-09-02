import abc
import numpy as np
import pandas as pd
import os.path
from pqueens.post_post.post_post import Post_post

class PP_time_series(Post_post):
    """ Base class for post_post routines """

    def __init__(self, base_settings):
        super(PP_time_series, self).__init__(base_settings)

    @classmethod
    def from_config_create_post_post(cls, config, base_settings):
        """ Create post_post routine from problem description

        Args:
            config: input json file with problem description

        Returns:
            post_post: post_post object
        """
        return cls(base_settings)

    def read_post_files(self, output_file): # output file given by driver
    # loop over several post files if list of post processors given
        output_dir = os.path.dirname(output_file)
        post_out = []

        for num in range(self.num_post):
            # different read methods depending on subfix
            if self.subfix=='mon':
                path = output_dir + r'/QoI_' + str(num+1) + r'.mon'
                post_data = np.loadtxt(path, usecols=self.usecols, skiprows=self.skiprows)
            elif self.subfix=='csv':
                path =output_dir + r'/QoI_' + str(num+1) + r'.csv'
                post_data = pd.read_csv(path, usecols=self.usecols, skiprows=self.skiprows)
            else:
                raise RuntimeError("Subfix of post processed file is unknown!")

            QoI_identifier = abs(post_data[:,0]-self.target_time) < self.time_tol
            QoI = post_data[QoI_identifier][0,1]
            post_out = np.append(post_out, QoI) # select only row with timestep equal to target time step
            if not post_out: # timestep reached? <=> variable is empty?
                self.error = True
        return post_out, self.error
