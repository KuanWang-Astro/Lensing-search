import argparse

parser = argparse.ArgumentParser(description='Compute delta sigma')
parser.add_argument('--td',type=int,default=0,dest='td',help='time delay in seconds') 
parser.add_argument('--Nparams',required=True,type=int,dest='Nparams')
parser.add_argument('--infile',required=True,dest='infile')
parser.add_argument('--outfile',required=True,dest='outfile')
parser.add_argument('--central',type=bool,default=False,dest='central')
#parser.add_argument('--color',choices=['red','blue'],default='all',dest='color')
args = parser.parse_args()

import time
time.sleep(args.td)

import collections
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor as Pool

from datetime import datetime

from halotools.sim_manager import CachedHaloCatalog

from HOD_models import decorated_hod_model
from HOD_models import standard_hod_model

from halotools.empirical_models import MockFactory

from halotools.mock_observables import return_xyz_formatted_array
from halotools.mock_observables import delta_sigma
from halotools.utils import randomly_downsample_data
from halotools.utils import crossmatch

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('ngals','deltasigma','rp','param')

##########################################################



halocat = CachedHaloCatalog(simname = 'bolplanck', version_name = 'halotools_v0p4', redshift = 0, \
                            halo_finder = 'rockstar')

Lbox = 250

rp_bins = np.logspace(-1.398, 1.176, 14)  ##to match the leauthaud paper
num_ptcls_to_use = int(1e6)
particle_masses = np.zeros(num_ptcls_to_use)+halocat.particle_mass
total_num_ptcls_in_snapshot = len(halocat.ptcl_table)
downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)

pos_part = return_xyz_formatted_array(*(halocat.ptcl_table[ax] for ax in 'xyz'), period=Lbox)

pos_part = randomly_downsample_data(pos_part, num_ptcls_to_use)

#########################################################


def calc_all_observables(param):

    model.param_dict.update(dict(zip(param_names, param)))    ##update model.param_dict with pairs (param_names:params)

    try:
        model.mock.populate()
    except:
        model.populate_mock(halocat)
    
    gc.collect()
    
    output = []


    pos_gals = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'),period=Lbox)
    pos_gals = np.array(pos_gals,dtype=float)
    if args.central:
        mask_cen = model.mock.galaxy_table['gal_type']=='centrals'
        pos_gals = pos_gals[mask_cen]

    """
    if args.color=='red':
        mask_red = (model.mock.galaxy_table['sfr_designation'] == 'quiescent')
        pos_gals = pos_gals[mask_red]
    elif args.color=='blue':
        mask_blue = (model.mock.galaxy_table['sfr_designation'] == 'active')
        pos_gals = pos_gals[mask_blue]
    """

    #ngals
    output.append(model.mock.galaxy_table['x'].size)
    
    #delta sigma
    
    deltasigma = delta_sigma(pos_gals, pos_part, particle_masses=particle_masses, downsampling_factor=downsampling_factor, rp_bins=rp_bins, period=Lbox)
    
    output.append(deltasigma[1])
    output.append(deltasigma[0])
    
    
    # parameter set
    output.append(param)
    
    return output


############################################################



def main(params_fname, output_fname):
    nparams = args.Nparams
    params5 = np.loadtxt(params_fname, usecols=range(5))
    params5 = params5[np.random.choice(len(params5), nparams)]
    
    params = np.zeros((1681*nparams,7))
    
    for i in range(nparams):
        params[i*1681:i*1681+1681,:5] = params5[i]
        for j in range(41):
            params[i*1681+j*41:i*1681+j*41+41,5] = -1+0.05*j
            for k in range(41):
                params[i*1681+j*41+k,6] = -1+0.05*k
    
    nproc = 55
    
    global model
    model = decorated_hod_model()

    output_dict = collections.defaultdict(list)

    with Pool(nproc) as pool:
        for i, output_data in enumerate(pool.map(calc_all_observables, params)):
            if i%55 == 54:
                print i
                print str(datetime.now())
            for name, data in zip(output_names, output_data):
                output_dict[name].append(data)

    for name in output_names:
        output_dict[name] = np.array(output_dict[name])

    np.savez(output_fname, **output_dict)


if __name__ == '__main__':
    main(args.infile, args.outfile)
    with open(args.outfile+'_log','w') as f:
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')
