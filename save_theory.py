"""
Save theory points to a text file.

Note that adding this to the end of a pipeline will save *all* evaluated
points, not just accepted ones.  You will have to compare to the final
chain file to determine which points are accepted and which are not.
"""

from cosmosis.datablock import option_section
from cosmosis.runtime.config import Inifile
import numpy as np


def setup(options):
    # Name of where samples are put
    filename = options[option_section, 'filename']

    # If under MPI make lots of file names .0 .1 etc
    # Do not load MPI if asked not to (e.g. NERSC crash)
    if options.get_bool(option_section, 'mpi', default=True):
        import mpi4py.MPI
        rank = mpi4py.MPI.COMM_WORLD.Get_rank()
        size = mpi4py.MPI.COMM_WORLD.Get_size()
        if size > 1:
            filename = filename + '.' + str(rank)

    # Samples go here
    outfile = open(filename, 'w')

    # Read likelihoods requested in the param file,
    # or None 
    if options.has_value('pipeline', 'likelihoods'):
        likes = options['pipeline', 'likelihoods'].split()
    else:
        likes = None

    # Work out the varied parameters
    # Hackish to do this again, but it's not exposed here otherwise
    values_file = options['pipeline', 'values']
    values_ini = Inifile(values_file)

    # Work out which ones are varied and which are fixed
    varied_params = []
    for ((section, name), value) in values_ini:
        words = value.split()
        if len(words) == 3:
            varied_params.append((section,name))

    # Write file headers (just param names here,
    # data elements below)
    outfile.write('#')
    for (section,name) in varied_params:
        outfile.write('{}--{}  '.format(section,name))

    # For info    
    print(varied_params)

    return [outfile, likes, varied_params]

def execute(block, config):

    # We will modify config later by adding an element
    # to indicate that the header has been completed
    (outfile, likes, varied_params) = config[:3]

    # Build up output line, starting with
    # input param values
    data = []
    for (section, name) in varied_params:
        data.append(block[section, name])
    data = np.array(data)

    # If left blank (=automatic) then write determine
    # which likelihoods to use
    if likes is None:
        likes = []
        for _, key in block.keys('data_vector'):
            if key.endswith('_theory'):
                likes.append(key[:-7])

    # Get all the theory vectors
    for like in likes:
        theory = block['data_vector', like+"_theory"]
        data = np.append(data, theory)

    # If this is our first execution, write the rest of
    # the header line d_0  d_1  d_2 ...
    if len(config)==3:
        for i in range(len(data) - len(varied_params)):
            outfile.write('d_{}  '.format(i))
        outfile.write('\n')
        config.append(True)

    # Save output line - made 2D so it is saved as a single
    # line
    data = np.atleast_2d(data)
    np.savetxt(outfile, data)
    outfile.flush()

    return 0

