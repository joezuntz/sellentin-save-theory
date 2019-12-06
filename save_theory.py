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


    values_file = options['pipeline', 'values']
    values_ini = Inifile(values_file)

    varied_params = []
    for ((section, name), value) in values_ini:
        words = value.split()
        if len(words) == 3:
            varied_params.append((section,name))
    outfile.write('#')
    for (section,name) in varied_params:
        outfile.write('{}--{}  '.format(section,name))
    
    print(varied_params)

    return [outfile, likes, varied_params]

def execute(block, config):

    (outfile, likes, varied_params) = config[:3]

    data = []
    for (section, name) in varied_params:
        data.append(block[section, name])
    data = np.array(data)

    # 
    if likes is None:
        likes = []
        for _, key in block.keys('data_vector'):
            if key.endswith('_theory'):
                likes.append(key[:-7])

    for like in likes:
        theory = block['data_vector', like+"_theory"]
        data = np.append(data, theory)

    if len(config)==3:
        for i in range(len(data) - len(varied_params)):
            outfile.write('d_{}  '.format(i))
        outfile.write('\n')
        config.append(True)

    data = np.atleast_2d(data)
    np.savetxt(outfile, data)

    outfile.flush()

    return 0

