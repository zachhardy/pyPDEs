import os
import sys
import time
import copy
import itertools
import numpy as np


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


def define_range(reference, variance, n):
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
    samples = np.linspace(-1.0, 1.0, n)
    return reference*(1.0 + variance * samples)


def parameter_study(
        problem: str,
        study: int,
        lhs: bool = False,
        n: int = 50):
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
        msg = f"{path} is not a valid directory."
        raise NotADirectoryError(msg)
    filepath = f"{path}/run.py"

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
            parameters['scatter'] = define_range(sig_s01, 0.2, 21)
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
            parameters['radius'] = define_range(radius, 0.02, 4)
            parameters['density'] = define_range(density, 0.005, 4)
            parameters['scatter'] = define_range(sig_s01, 0.1, 4)
        else:
            msg = f"Invalid study number for {problem}."
            raise AssertionError(msg)

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
            msg = f"Invalid study number for {problem}."
            raise AssertionError(msg)

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
            msg = f"Invalid study number for {problem}."
            raise AssertionError(msg)

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
            msg = f"Invalid study number for {problem}."
            raise AssertionError(msg)

    else:
        raise AssertionError("Invalid problem name.")

    keys = list(parameters.keys())
    max_len = np.max([len(key) for key in keys])

    if lhs:
        from scipy.stats import qmc
        d = len(parameters.keys())
        sampler = qmc.LatinHypercube(d=d)
        samples = sampler.random(n)

        l_bounds, u_bounds = [], []
        for vals in parameters.values():
            l_bounds.append(min(vals))
            u_bounds.append(max(vals))
        values = qmc.scale(samples, l_bounds, u_bounds)
    else:
        values = np.array(list(itertools.product(*parameters.values())))
    values = np.round(values, 10)

    ##################################################
    # Setup the output paths
    ##################################################

    # Define the path to the output directory
    output_path = f"{path}/parametric/{keys[0]}"
    for k, key in enumerate(keys[1:]):
        output_path += f"_{key}"
    setup_directory(output_path)

    # Save the parameters to a file
    param_path = f"{output_path}/params.txt"
    header = " ".join([f"{key:<13} " for key in keys])
    np.savetxt(param_path, values, fmt='%.8e', header=header)

    ##################################################
    # Run the reference problem
    ##################################################

    sim_path = os.path.join(output_path, "reference")
    setup_directory(sim_path)

    cmd = f"python {filepath} output_directory={sim_path} "
    cmd += f"xs_directory={path}/xs >> {sim_path}/log.txt"
    os.system(cmd)

    ##################################################
    # Run the parameter study
    ##################################################

    total_time = 0.0
    for n, params in enumerate(values):

        # Setup output path
        sim_path = os.path.join(output_path, str(n).zfill(3))
        setup_directory(sim_path)

        cmd = f"python {path}/run.py "
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


if __name__ == "__main__":

    ############################################################
    # Maximum Parameter Study Number:
    #   Sphere3g     - 6
    #   InfiniteSlab - 6
    #   TWIGL        - 6
    #   LRA          - 6
    ############################################################

    if len(sys.argv) == 2:
        problem_name = sys.argv[1]
        if problem_name == "Sphere3g":
            max_study = 6
        elif problem_name == "InfiniteSlab":
            max_study = 6
        elif problem_name == "TWIGL":
            max_study = 6
        elif problem_name == "LRA":
            max_study = 6
        else:
            raise NotImplementedError("Invalid problem name.")

        check = input("Are you sure you want to run every parameter "
                      f"study for {problem_name}? [y/n] ")
        if "y" in check:
            for i in range(max_study + 1):
                parameter_study(problem_name, i)
        if "n" in check:
            print("Terminating program.")
            exit(0)

    elif len(sys.argv) == 3:
        use_lhs = "y" in input("Latin Hypercubes? [y/n] ")
        n_samples = 0 if not use_lhs else int(input("How many samples? "))

        parameter_study(*sys.argv[1:], use_lhs, n_samples)

    else:
        raise AssertionError("Invalid command line inputs.")
