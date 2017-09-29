import numpy as np
import halotools
from halotools.sim_manager import CachedHaloCatalog
import math
import random


class HOD(object):
    def __init__(self):
        return None

            
    def standard_hod_cen_moment1(self,Mvir):
        return 0.5*(1+math.erf((np.log10(Mvir)-self.logMmin)/self.sigma_logM))

    def standard_hod_sat_moment1(self,Mvir):
        if Mvir>np.power(10,self.logM0):
            return np.power((Mvir-np.power(10,self.logM0))/np.power(10,self.logM1),self.alpha)
        else:
            return 0.0
    
    def decorated_hod_cen_moment1(self,Mvir):
        standard = 0.5*(1+math.erf((np.log10(Mvir)-self.logMmin)/self.sigma_logM))
        if standard>0.5:
            return standard+self.Acen*(1.0-standard),standard-self.Acen*(1.0-standard)
        else:
            return standard*(1.0+self.Acen),standard*(1.0-self.Acen)
    
    def decorated_hod_sat_moment1(self,Mvir):
        if Mvir>np.power(10,self.logM0):
            return (1.0+self.Asat)*np.power((Mvir-np.power(10,self.logM0))/np.power(10,self.logM1),self.alpha),(1.0-self.Asat)*np.power((Mvir-np.power(10,self.logM0))/np.power(10,self.logM1),self.alpha)
        else:
            return 0.0,0.0

    def set_param_model(self,param):
        self.model = 'standard'
        self.alpha = param[0]
        self.logM1 = param[1]
        self.sigma_logM = param[2]
        self.logM0 = param[3]
        self.logMmin = param[4]
        if param.size>5:
            self.Acen = param[5]
            self.Asat = param[6]
            self.model = 'decorated'

    def ngal_tot(self,Mvir,param):
        self.set_param_model(param)
        if self.model=='decorated':
            ngal = (self.decorated_hod_cen_moment1(Mvir)[0]*(1+self.decorated_hod_sat_moment1(Mvir)[0]),self.decorated_hod_cen_moment1(Mvir)[1]*(1+self.decorated_hod_sat_moment1(Mvir)[1]))
        else:
            ngal = self.standard_hod_cen_moment1(Mvir)*(1+self.standard_hod_sat_moment1(Mvir))
        return ngal

    def ngal_cen(self,Mvir,param):
        self.set_param_model(param)
        if self.model=='decorated':
            ngal = (self.decorated_hod_cen_moment1(Mvir)[0],self.decorated_hod_cen_moment1(Mvir)[1])
        else:
            ngal = self.standard_hod_cen_moment1(Mvir)
        return ngal




