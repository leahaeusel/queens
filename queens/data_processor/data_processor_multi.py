"""Data processor class to combine the outputs of multiple other data
processors."""

import logging
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from queens.data_processor.data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorMulti(DataProcessor):
    def __init__(
        self,
        data_processors,
    ):
        super().__init__(
            file_name_identifier=" ",
            file_options_dict={},
        )
        self.data_processors = data_processors

    def get_data_from_file(self, base_dir_file):
        """Get data of interest from file.

        Args:
            base_dir_file (Path): Path of the base directory that contains the file of interest

        Returns:
            processed_data (np.array): Final data from data processor module
        """
        if not base_dir_file:
            raise ValueError(
                "The data processor requires a base_directory for the "
                "files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir_file, Path):
            raise TypeError(
                "The argument 'base_dir_file' must be of type 'Path' "
                f"but is of type {type(base_dir_file)}. Abort..."
            )

        processed_data = []
        for data_processor in self.data_processors:
            processed_data.append(data_processor.get_data_from_file(base_dir_file))

        return processed_data

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (obj): Raw data from file.
        """
        pass

    def filter_and_manipulate_raw_data(self, raw_data_set):
        """Filter or clean the raw data for given criteria.

        Args:
            raw_data (obj): Raw data from file.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        pass
