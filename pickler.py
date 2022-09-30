"""
This module reads datasets of parametric simulation data into
neutronics dataset reader classes and pickles the result for
faster reading when executing a ROM.

Command line arguments with problem names can be provided to
only pickle data related to those problem. The default behavior
pickles data from all available problems.
"""

import os
import sys
import time
import pickle

from readers import NeutronicsDatasetReader


if __name__ == "__main__":

    # Parse problem names
    problems = []
    for arg in sys.argv[1:]:
        if arg not in ["Sphere3g", "InfiniteSlab", "TWIGL", "LRA"]:
            raise ValueError(f"{arg} is not a valid problem name.")
        problems.append(arg)
    if len(problems) == 0:
        problems = valid_problems

    path = os.path.abspath(os.path.dirname(__file__))
    path = f"{path}/Problems"

    for problem in problems:
        data_path = f"{path}/{problem}/parametric"
        if not os.path.isdir(data_path):
            msg = f"{data_path} is not a valid directory."
            raise NotADirectoryError(msg)

        pickle_path = f"{path}/{problem}/pickles"
        if not os.path.isdir(pickle_path):
            os.makedirs(pickle_path)

        # Pickle each parameter study dataset
        for study in sorted(os.listdir(data_path)):
            study_path = f"{data_path}/{study}"
            if not os.path.isdir(study_path):
                continue

            tstart = time.time()
            reader = NeutronicsDatasetReader(study_path).read()
            tend = time.time()
            print(f"Reading {study_path}... "
                  f"{tend - tstart:.3g} s")

            filepath = f"{pickle_path}/{study}.obj"
            with open(filepath, 'wb') as file:
                pickle.dump(reader, file)
