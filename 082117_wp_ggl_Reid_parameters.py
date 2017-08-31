import argparse

parser = argparse.ArgumentParser(description='Compute delta sigma')
parser.add_argument('--td',type=int,default=0,dest='td',help='time delay in seconds') 
parser.add_argument('--Nparams',required=True,type=int,dest='Nparams')
parser.add_argument('--outfile',required=True,dest='outfile')
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
from halotools.mock_observables import wp
from halotools.utils import randomly_downsample_data
from halotools.utils import crossmatch

##########################################################

param_names = ('alpha','logM1','sigma_logM','logM0','logMmin','mean_occupation_centrals_assembias_param1','mean_occupation_satellites_assembias_param1')
output_names = ('ngals','deltasigma','rp','wprp','param')

##########################################################



halocat = CachedHaloCatalog(simname = 'diemerL0500', version_name = 'antonio', redshift = 0, \
                            halo_finder = 'rockstar',ptcl_version_name='antonioz0')

Lbox = 500

rp_bins = np.logspace(-1.398, 1.176, 14)  ##to match the leauthaud paper
num_ptcls_to_use = int(1e6)
particle_masses = np.zeros(num_ptcls_to_use)+halocat.particle_mass
total_num_ptcls_in_snapshot = len(halocat.ptcl_table)
downsampling_factor = total_num_ptcls_in_snapshot/float(num_ptcls_to_use)
##ggl

pi_max = 60
r_wp = np.logspace(-1, np.log10(Lbox)-1, 20)
##wp


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

    pos_gals_d = return_xyz_formatted_array(*(model.mock.galaxy_table[ax] for ax in 'xyz'), \
            velocity=model.mock.galaxy_table['vz'], velocity_distortion_dimension='z',\
                                          period=Lbox)             ##redshift space distorted
    pos_gals_d = np.array(pos_gals_d,dtype=float)

    #ngals
    output.append(model.mock.galaxy_table['x'].size)
    
    #delta sigma
    
    deltasigma = delta_sigma(pos_gals, pos_part, particle_masses=particle_masses, downsampling_factor=downsampling_factor, rp_bins=rp_bins, period=Lbox)
    
    output.append(deltasigma[1])
    output.append(deltasigma[0])
    
    # wprp
    output.append(wp(pos_gals_d, r_wp, pi_max, period=Lbox))
    
    
    # parameter set
    output.append(param)
    
    return output


############################################################



def main(output_fname):
    nparams = args.Nparams
    params = np.array((0.73,14.09,0.32,13.27,13.004))
    params = params*np.ones((nparams,5))
    
    nproc = 55
    
    global model
    model = standard_hod_model()

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
    main(args.outfile)
    with open(args.outfile+'_log','w') as f:
        for arg in vars(args):
            f.write(str(arg)+':'+str(getattr(args, arg))+'\n')
