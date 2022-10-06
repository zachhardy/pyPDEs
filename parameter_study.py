import os
import sys
import time
import copy
import pickle
import itertools
import numpy as np

from readers import NeutronicsDatasetReader


def setup_directory(path):
    """
    Prepare a directory for simulation outputs.

    Parameters
    ----------
    path : str, The path to the directory.
    """
    if os.path.isdir(path):
        os.system(f"rm -r {path}")
    os.makedirs(path)


def define_range(reference, variance, n, up=True, down=True):
    """
    Define sample points for a parameter given the nominal value,
    the variance, and the number of samples desired. This routine
    generates uniformly spaced samples within plus or minus the
    specified variance about the specified reference value.

    Parameters
    ----------
    reference : float, The nominal parameter value.
    variance : float, The variance of the parameter.
    n : int, The number of samples to generate.

    Returns
    -------
    numpy.ndarray : The samples to use in the parameter study.
    """
    assert up or down
    lower = -1.0 if down else 0.0
    upper = 1.0 if up else 0.0
    samples = np.linspace(lower, upper, n)
    return reference*(1.0 + variance * samples)


def parameter_study(
        problem: str,
        study: int,
        validation: bool = False,
        n_validation_runs: int = 10):
    """
    Define and run a parameter study.

    Parameters
    ----------
    problem : str {'Sphere3g', 'InfiniteSlab', 'TWIGL', 'LRA'}
        The problem to run a parameter study for.
    study : int, The pre-defined parameter study to run.
    lhs : bool, default False
        A flag for Latin Hypercube sampling.
    """

    path = os.path.dirname(os.path.abspath(__file__))
    path = f"{path}/Problems/{problem}"
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a valid directory.")
    run_filepath = f"{path}/run.py"

    study = int(study)

    ##################################################
    # Define the parameter study
    ##################################################

    # Three group sphere problem
    if problem == "Sphere3g":

        # Default value
        radius = 6.0 if study != 2 else 6.1612
        density = 0.05
        sig_s01 = 1.46

        # Define the parameter space
        parameters = {}
        if study == 0:
            parameters['radius'] = define_range(6.1612, 0.025, 21)
        elif study == 1:
            parameters['density'] = define_range(0.05134325, 0.025, 21)
        elif study == 2:
            parameters['scatter'] = define_range(sig_s01, 0.1, 21)
        elif study == 3:
            parameters['radius'] = define_range(radius, 0.01, 6)
            parameters['density'] = define_range(density, 0.01, 6)
        elif study == 4:
            parameters['radius'] = define_range(radius, 0.01, 6)
            parameters['scatter'] = define_range(sig_s01, 0.1, 6)
        elif study == 5:
            parameters['density'] = define_range(density, 0.01, 6)
            parameters['scatter'] = define_range(sig_s01, 0.1, 6)
        elif study == 6:
            parameters['radius'] = define_range(radius, 0.01, 4, up=False)
            parameters['density'] = define_range(density, 0.01, 4, up=False)
            parameters['scatter'] = define_range(sig_s01, 0.05, 4, up=False)
        else:
            raise ValueError(f"{study} is an invalid study number.")

    # Infinite slab problem
    elif problem == "InfiniteSlab":

        # Default values
        magnitude = -0.01
        duration = 1.0
        interface = 40.0

        # Define the parameter space
        parameters = {}
        if study == 0:
            parameters['magnitude'] = define_range(magnitude, 0.2, 21)
        elif study == 1:
            parameters['duration'] = define_range(duration, 0.2, 21)
        elif study == 2:
            parameters['interface'] = define_range(interface, 0.05, 21)
        elif study == 3:
            parameters['magnitude'] = define_range(magnitude, 0.2, 6)
            parameters['duration'] = define_range(duration, 0.2, 6)
        elif study == 4:
            parameters['magnitude'] = define_range(magnitude, 0.1, 6)
            parameters['interface'] = define_range(interface, 0.025, 6)
        elif study == 5:
            parameters['duration'] = define_range(duration, 0.2, 6)
            parameters['interface'] = define_range(interface, 0.025, 6)
        elif study == 6:
            parameters['magnitude'] = define_range(magnitude, 0.05, 4)
            parameters['duration'] = define_range(duration, 0.05, 4)
            parameters['interface'] = define_range(40.0, 0.025, 4)
        else:
            raise ValueError(f"{study} is an invalid study number.")

    # TWIGL problem
    elif problem == "TWIGL":

        # Default values
        magnitude = 0.97667 - 1.0
        duration = 0.2
        scatter = 0.01

        # Define the parameter space
        parameters = {}
        if study == 0:
            parameters['magnitude'] = define_range(magnitude, 0.2, 21)
        elif study == 1:
            parameters['duration'] = define_range(duration, 0.2, 21)
        elif study == 2:
            parameters['scatter'] = define_range(scatter, 0.25, 21)
        elif study == 3:
            parameters['magnitude'] = define_range(magnitude, 0.02, 6)
            parameters['duration'] = define_range(duration, 0.25, 6)
        elif study == 4:
            parameters['magnitude'] = define_range(magnitude, 0.2, 6)
            parameters['scatter'] = define_range(scatter, 0.25, 6)
        elif study == 5:
            parameters['duration'] = define_range(duration, 0.2, 6)
            parameters['scatter'] = define_range(scatter, 0.25, 6)
        elif study == 6:
            parameters['magnitude'] = define_range(magnitude, 0.2, 4)
            parameters['duration'] = define_range(duration, 0.2, 4)
            parameters['scatter'] = define_range(scatter, 0.25, 4)
        else:
            raise ValueError(f"{study} is an invalid study number.")

    # LRA benchmark problem
    elif problem == "LRA":

        # Default values
        magnitude = 0.8787631 - 1.0
        duration = 2.0
        feedback = 3.034e-3

        # Define parameter space
        parameters = {}
        if study == 0:
            parameters['magnitude'] = define_range(magnitude, 0.025, 21)
        elif study == 1:
            parameters['duration'] = define_range(duration, 0.05, 21)
        elif study == 2:
            parameters['feedback'] = define_range(feedback, 0.05, 21)
        elif study == 3:
            parameters['magnitude'] = define_range(magnitude, 0.025, 6)
            parameters['duration'] = define_range(duration, 0.05, 6)
        elif study == 4:
            parameters['magnitude'] = define_range(magnitude, 0.025, 6)
            parameters['feedback'] = define_range(feedback, 0.05, 6)
        elif study == 5:
            parameters['duration'] = define_range(duration, 0.05, 6)
            parameters['feedback'] = define_range(feedback, 0.05, 6)
        elif study == 6:
            parameters['magnitude'] = define_range(magnitude, 0.025, 4)
            parameters['duration'] = define_range(duration, 0.05, 4)
            parameters['feedback'] = define_range(feedback, 0.05, 4)
        else:
            raise ValueError(f"{study} is an invalid study number.")

    else:
        raise AssertionError(f"{problem} is an invalid problem name.")

    keys = list(parameters.keys())
    max_len = np.max([len(key) for key in keys])

    if validation:
        d = len(parameters.keys())
        rng = np.random.default_rng()
        values = np.zeros((n_validation_runs, d))
        for p, vals in enumerate(parameters.values()):
            low, high = min(vals), max(vals)
            values[:, p] = rng.uniform(low, high, n_validation_runs)
    else:
        values = np.array(list(itertools.product(*parameters.values())))
    values = np.round(values, 10)

    ##################################################
    # Setup the output paths
    ##################################################

    # Define the path to the output directory
    study_name = "_".join(keys)
    if validation:
        output_path = f"{path}/validation/{study_name}"
    else:
        output_path = f"{path}/parametric/{study_name}"
    setup_directory(output_path)

    # Save the parameters to a file
    param_path = f"{output_path}/params.txt"
    header = " ".join([f"{key:<13} " for key in keys])
    np.savetxt(param_path, values, fmt='%.8e', header=header)

    ##################################################
    # Run the reference problem
    ##################################################

    # sim_path = os.path.join(output_path, "reference")
    # setup_directory(sim_path)
    #
    # cmd = f"python {filepath} output_directory={sim_path} "
    # cmd += f"xs_directory={path}/xs >> {sim_path}/log.txt"
    # os.system(cmd)

    ##################################################
    # Run the parameter study
    ##################################################

    total_time = 0.0
    for n, params in enumerate(values):

        # Setup output path
        sim_path = os.path.join(output_path, str(n).zfill(3))
        setup_directory(sim_path)

        cmd = f"python {run_filepath} "
        for k, key in enumerate(keys):
            cmd += f"{key}={params[k]} "
        cmd += f"output_directory={sim_path} "
        cmd += f"xs_directory={path}/xs >> {sim_path}/log.txt"

        msg = f"="*50 + f"\nRunning Simulation {n}\n" + f"="*50
        for k, key in enumerate(keys):
            s = " ".join([w.capitalize() for w in key.split("_")])
            msg += f"\n{s:<{max_len}}:\t{params[k]:<5.3e}"
        print()
        print(msg)

        t_start = time.time()
        os.system(cmd)
        sim_time = time.time() - t_start
        total_time += sim_time

        print()
        print(f"Simulation Time = {sim_time:.3f} s")

    print()
    print(f"Average Simulation Time = {total_time/len(values):.3e} s")
    print(f"Total Parameter Study Time = {total_time:.3e} s")

    if validation:
        pickle_path = f"{path}/pickles/validation"
    else:
        pickle_path = f"{path}/pickles/training"
    if not os.path.isdir(pickle_path):
        os.makedirs(pickle_path)

    obj_filepath = f"{pickle_path}/{study_name}.obj"
    reader = NeutronicsDatasetReader(output_path).read()
    with open(obj_filepath, 'wb') as file:
        pickle.dump(reader, file)

    print(f"Dataset reader saved to {obj_filepath}.")


if __name__ == "__main__":

    ############################################################
    # Maximum Parameter Study Number:
    #   Sphere3g     - 6
    #   InfiniteSlab - 6
    #   TWIGL        - 6
    #   LRA          - 6
    ############################################################

    if len(sys.argv) < 2:
        msg = "Invalid command line arguments. "
        msg += "A problem name must be provided."
        raise AssertionError(msg)

    problem_name = sys.argv[1]
    study_num = -1

    # Validation parameters
    n_runs = 10
    is_validation = False

    # Parse other arguments
    if len(sys.argv) > 2:
        for i, arg in enumerate(sys.argv[2:]):
            if i == 0 and "=" not in arg:
                study_num = int(arg)
            else:
                argval = arg.split("=")[1]
                if "validation=" in arg:
                    is_validation = bool(int(argval))
                if "nruns=" in arg:
                    n_runs = int(argval)

    # If no study number, run all of them
    if study_num == -1:
        check = input("Are you sure you want to run every parameter "
                      f"study for {problem_name}? [y/n] ")\

        if "y" in check:
            for i in range(7):
                parameter_study(problem_name, i, is_validation, n_runs)

        if "n" in check:
            print("Terminating program.")
            exit(0)

    else:
        parameter_study(problem_name, study_num, is_validation, n_runs)
