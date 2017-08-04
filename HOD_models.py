from halotools.empirical_models import PrebuiltHodModelFactory
from halotools.empirical_models import AssembiasZheng07Cens
from halotools.empirical_models import TrivialPhaseSpace
from halotools.empirical_models import AssembiasZheng07Sats
from halotools.empirical_models import NFWPhaseSpace
from halotools.empirical_models import HodModelFactory
from halotools.empirical_models import PrebuiltHodModelFactory

def decorated_hod_model():
    cen_occ_model = AssembiasZheng07Cens(prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
    cen_prof_model = TrivialPhaseSpace()
    sat_occ_model = AssembiasZheng07Sats(prim_haloprop_key='halo_mvir', sec_haloprop_key='halo_nfw_conc')
    sat_prof_model = NFWPhaseSpace()
    return HodModelFactory(centrals_occupation=cen_occ_model, centrals_profile=cen_prof_model, satellites_occupation=sat_occ_model, satellites_profile=sat_prof_model)


def standard_hod_model():
    return PrebuiltHodModelFactory('zheng07', threshold=-20)