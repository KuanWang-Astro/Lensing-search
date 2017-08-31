import numpy as np
import math
import random

########

class Ngal_estimate(object):
    def __init__(self,halocat,param):
        self.halocat = halocat
        self.conc_split = np.median(halocat.halo_table['halo_nfw_conc'])
        self.alpha = param[0]
        self.logM1 = param[1]
        self.sigma_logM = param[2]
        self.logM0 = param[3]
        self.logMmin = param[4]
        self.Acen = param[5]
        self.Asat = param[6]

    def concentration_split(self,conc):
        if conc<self.conc_split:
            return -1
        else:
            return 1
    
    def decorated_hod_cen_moment1(self,Mvir,conc):
        standard = 0.5*(1+math.erf((np.log10(Mvir)-self.logMmin)/self.sigma_logM))
        if standard>0.5:
            return standard+self.concentration_split(conc)*self.Acen*(1.0-standard)
        else:
            return standard*(1.0+self.concentration_split(conc)*self.Acen)
    
    def decorated_hod_sat_moment1(self,Mvir,conc):
        if Mvir>np.power(10,self.logM0):
            return (1.0+self.Asat*self.concentration_split(conc))*np.power((Mvir-np.power(10,self.logM0))/np.power(10,self.logM1),self.alpha)
        else:
            return 0.0

    def ngal_estimate(self):
        ngal = 0
        for halo in self.halocat.halo_table:
            Mvir = halo['halo_mvir']
            conc = halo['halo_nfw_conc']
            ngal += self.decorated_hod_cen_moment1(Mvir,conc)+self.decorated_hod_sat_moment1(Mvir,conc)
            if ngal>1e5:
                break
        return ngal
