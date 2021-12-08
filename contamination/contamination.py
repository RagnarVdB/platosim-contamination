from filters import Filter, filterB, filterR, filterP, fromVtoP
from platosim.simulation import Simulation
from platosim.simfile import SimFile
from sciencetools.fitting import fit

import glob
import os
from copy import deepcopy
from abc import abstractmethod, ABC
from functools import lru_cache
from typing import List

import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.stats import chisquare, poisson, norm
from PyAstronomy import funcFit, pyasl
from pytransit import QuadraticModel
from astropy.io import fits

home = os.environ["PLATO_WORKDIR"] + "/"

# Relation between magnitude and platosim flux
a = np.genfromtxt(home + "platocon/contamination/exponentialparameter.txt")
#mag, flux = np.genfromtxt(home + "platocon/contamination/spline.csv", delimiter=",")[1:,1:].T
mag, flux = np.genfromtxt(home + "platocon/contamination/spline.csv", delimiter=",").T

correction = interp1d(mag, flux)
def magToFlux(magnitude):
    return a*np.exp(-0.92103*magnitude) + correction(magnitude)


class Star:
    """
    Represents a star with blackbody model

    Parameters
    ----------
    Teff: float
        Effective temperature
    V_mag : float
        Magnitude in the reference passband, see ref_filter
    ref_filter : Filter
        Passband in which V_mag is calculated

    Attributes
    ----------
    Teff : float
        Effective temperature
    Vmag : float
        Magnitude in the reference passband, see ref_filter
    referenceFlux : float
    Ra : float
    Dec : float
    spectrumpath : str
        Path to a fits spectrum
    """
    def __init__(self, T_eff: float, V_mag: float, Ra: float = 86.73, Dec: float = -46.35, ref_filter: Filter = "visible", spectrumpath : str = None):
        self.Teff = T_eff
        # Special case: magnitudes are determined from plato magnitude
        if ref_filter == "visible":
            ref_filter = filterP
            self.Vmag = fromVtoP(V_mag, T_eff)
        else:
            self.Vmag = V_mag
        self.referenceFlux = self.getTotalFlux(ref_filter)
        self.Ra = Ra
        self.Dec = Dec

        if spectrumpath is None:
             #Hardcoded file
            spectrumpath = os.environ["PLATO_WORKDIR"] + '/inputfiles/SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011/Z-0.0/lte04200-4.50-0.0.PHOENIX-ACES-AGSS-COND-SPECINT-2011.fits'
            if not os.path.exists(spectrumpath):
                raise FileNotFoundError("The default spectrumfile '" + spectrumpath + "' does not exist.")
        else:
            if not os.path.exists(spectrumpath):
                raise FileNotFoundError("The spectrumfile '" + spectrumpath + "' does not exist.")
        self.spectrumpath = spectrumpath

    @classmethod
    def createStarFromCatalog(cls, catalogPath: str, ID: int):
        """
        Creates a Star object from a given path to a star catalog with star ID.
        """
        data = np.loadtxt(catalogPath)
        if len(data.shape) == 1:
            RA = data[0]
            Dec = data[1]
            Mag = data[2]
            Temp = data[3]
            return Star(Temp, Mag, RA, Dec)
        else:
            RAs = data[:,0]
            Decs = data[:,1]
            Mags = data[:,2]
            Temps = data[:,3]
            IDs = np.array(data[:,4], dtype=np.uint16)
       
            if len(np.where(IDs == ID)[0])==0:
                print("ID not in Catalog")

            index = np.where(IDs == ID)[0][0]

            return Star(Temps[index], Mags[index], RAs[index], Decs[index])

    def spectrum(self, wav: np.ndarray):
        """
        Given an array of desired wavelengths, this function returns the Planck Blackbody distribution
        """
        p = 3.741771852192757e+29
        e = 14387768.775039336
        return p/wav**5 * 1/(np.exp(e/(wav*self.Teff)))
    
    @lru_cache(maxsize=10)
    def getLdc(self, passband: Filter):
        """
        This function returns a tuple (u1,u2) of limb darkening coefficients, 
        corresponding to the self Star object and to a certain given Filter. 
        

        """
        grid_no = 1000 # ZELF

        wvl_int = np.linspace(passband.wv_min, passband.wv_max, grid_no)
        tran_int = passband(wvl_int)
        wvl_int *= 10 # Go to Angström
        

        # Get low resolution angle dependent SIS spectra: 
            # - Each row in data is a spectrum
            # - mu is then the corresponding values for each spectrum

    
        data, hdr = fits.getdata(self.spectrumpath, header=True)
        mu        = fits.getdata(self.spectrumpath, 'MU')
        wvl_data = np.arange(hdr["CRVAL1"], hdr["CRVAL1"]+hdr["NAXIS1"]*hdr["CDELT1"], hdr["CDELT1"])
        #wvl_int = np.linspace(wvl_data[0] , wvl_data[-1] , len(wvl_data))  #ZELF
        # Interpolate dataset

        data_int     = np.zeros(grid_no*len(mu)).reshape(grid_no, len(mu))
        for jj in range(len(mu)):
            data_int[:,jj] = np.interp(wvl_int, wvl_data, data[jj])

        # Colvolve data with transmission to find the intensity
        intensity = [np.nansum(data_int[:,i]*tran_int) for i in range(len(mu))]
        intensity = intensity/max(intensity)

        # Interpolate to a finer grid

        mu_int = np.linspace(0, 1, grid_no)
        intensity_int = np.interp(mu_int, mu, intensity)

        # Get rid of the lowest mu's to do 2nd order fitting
        # NOTE The stellar limb causes a jump for mu<0.15

        ind = np.where(mu_int >= 0.15)
        mu_trunc = mu_int[ind]
        intensity_trunc = intensity_int[ind]

        # Initialize the class that will do the fitting, and prepare for the fit
        model_ldc = LimbDarkening()
        model_ldc.thaw(['u1', 'u2'])
        model_ldc.assignValue({'u1':0.2, 'u2':0.2})
        model_ldc.fit(mu_trunc, intensity_trunc, xtol=1.e-7, ftol=1.e-7, disp=0)
        return (model_ldc["u1"], model_ldc["u2"])


    @lru_cache(maxsize=10)
    def getTotalFlux(self, passband: Filter):
        return integrate.quad(lambda x: self.spectrum(x)*passband.func(x)*10**(-9), passband.wv_min, passband.wv_max)[0]
    
    def getMagnitude(self, passband: Filter):
        return self.Vmag - 2.5*np.log10(self.getTotalFlux(passband) / self.referenceFlux)

    def getFlux(self, passband: Filter):
        mag = self.getMagnitude(passband)
        return magToFlux(mag)

    def getMagnitudeB(self): return self.getMagnitude(filterB)

    def getMagnitudeR(self): return self.getMagnitude(filterR)


class TransitObject(ABC):
    """
    Abstract class representing transitting object. Currently only subclass is Planet. Binary star may be added in the future
    Parameters
    ----------
    k : float
        Ratio between planet radius and host star radius
    t0 : float
        Time at the middle of the transit
    p : float
        Orbital period
    a: float
        Semi-long axis divided by radius host star
    i: float
        Inclination angle (radians)
    e : float
        Eccentricity
    w : float
        Periastron argument

    Attributes
    ----------
    See parameters
    """

    names = ["k", "t0", "p", "a", "i", "e", "w"]

    def __init__(self, k : float,  t0 : float , p : float , a: float , i: float , e : float , w : float):
        self.k = k
        self.t0 = t0
        self.p = p
        self.a = a
        self.i = i
        self.e = e
        self.w = w
    
    @abstractmethod
    def getFlux(self, times: np.ndarray, ldc: tuple):
        """
        Returns normalised transit flux for given timeseries and limb darkening coefficients
        """
        pass
    
    @property
    def paramList(self):
        """
        Returns list of all transit parameters
        """
        return np.array([self.k, self.t0, self.p, self.a, self.i, self.e, self.w])

    def _setParameters(self, l):
        self.__init__(*l)

        

class Planet(TransitObject):
    """
    Abstract class representing transitting object. Currently only subclass is Planet. Binary star may be added in the future
    
    Parameters
    ----------
    k : float
        Ratio between planet radius and host star radius
    t0 : float
        Time at the middle of the transit
    p : float
        Orbital period
    a : float
        Semi-long axis divided by radius host star
    i : float
        Inclination angle (radians)
    e : float
        Eccentricity
    w : float
        Periastron argument

    Attributes
    ----------
    See parameters
    """
    def __init__(self, k : float,  t0 : float , p : float , a: float , i: float , e : float = 0 , w : float = 0):
        super().__init__(k, t0, p, a, i, e, w)
    
    def getFlux(self, times: np.ndarray, ldc: tuple):
        """
        Returns normalised transit flux for given timeseries and limb darkening coefficients
        """
        tm = QuadraticModel()
        tm.set_data(times)
        try:
            return tm.evaluate_ps(self.k, ldc, self.t0,self.p,self.a,self.i,self.e,self.w)
        except Exception as e:
            print("bad parameters: " + str(self.paramList))
            raise e


class Transit:
    """
    Class representing a transit of a transitObject aroud host star

    Parameters
    ----------
    host : Star
        Star around which transitobject 
    transitObject : TransitObject
        Transitting object

    Attributes
    ----------
    See parameters

    """
    def __init__(self, host: Star, transitObject: TransitObject):
        self.host = host
        self.transitObject = transitObject
    
    @staticmethod
    def fluxToMagnitude(flux):
        """Returns difference in magnitude for normalised flux"""
        return -2.5*np.log10(flux)
    
    def getFluxNorm(self, times: np.ndarray, passband: Filter):
        """Returns normalised flux curve, calculated by PyTransit"""
        return self.transitObject.getFlux(times, self.host.getLdc(passband))
    
    def getFlux(self, times: np.ndarray, passband: Filter):
        """Returns flux in absolute units (electrons)"""
        return self.host.getFlux(passband) * self.getFluxNorm(times, passband)
    
    def getDeltaMag(self, times: np.ndarray, passband: Filter):
        """Get difference in magnitude between transitted and untransitted star"""
        return self.fluxToMagnitude(self.getFluxNorm(times, passband))
    
    def writeToFile(self, times: np.ndarray, fileName: str, passband: Filter): #problem if comparing directly with PlatoSim output!
        """Write delta magnitude for a timeseries to a given file"""
        if os.path.exists(fileName):
            os.remove(fileName)
        data = np.array( [times*3600 , self.getDeltaMag(times, passband)])
        np.savetxt(fileName , data.T , fmt="%f")


class StarConfiguration:
    """
    Class representing a number of indistinguishable stars. Any of these stars, may undergo a transit.
    
    Parameters
    ----------
    stars : List[Star]
        List of all the stars in the configuration
    transits: List[Transit]
        List of all transits occuring in the configuration. Star attribute of each transit must be in stars (previous argument))
    
    Attributes
    ----------
    See parameters
    """
    def __init__(self, stars: List[Star], transits: List[Transit], baseFluxes: dict = {}):
        self.stars = stars
        if transits is None:
            self.transits = []
        else:
            self.transits = transits

        if any([transit.host not in stars for transit in self.transits]):
            raise Exception("transit host star must be in stars")

        self.baseFluxes = baseFluxes

    def setupSimulation(self, sim: Simulation, numExp: int, passband: Filter, appendix: str, path: str = "temp"):
        """
        Sets up a given simulation object to simulate all stars and transits.
        The method creates the necessary files in `path` and refers the `sim` object to these files.
        `appendix` is a string appended to each file name to avoid overwriting files
        """
        multiplier_timeseries = 1
        filename_stars = path + "/starFile" + appendix + ".txt"
        filename_times = path + "/timeseries"
        filename_varsource = path + "/varsource" + appendix + ".txt"

        # Write stars file
        if not all([star.Ra and star.Dec for star in self.stars]):
            raise Exception("No star coordinates")
        properties = np.array([[star.Ra, star.Dec, star.getMagnitude(passband), ID] for ID, star in enumerate(self.stars)])
        np.savetxt(filename_stars, properties, fmt='%f')

        # Write Timeseries file
        expTime = 2.5
        varStars = []
        times = np.linspace(0, numExp*expTime/3600, numExp*multiplier_timeseries)
        for transit in self.transits:
            try:
                i = self.stars.index(transit.host)
                varStars.append(i)
            except ValueError:
                raise "Star corresponding to transit {} not in list".format(i)
            transit.writeToFile(times, filename_times + str(i) + appendix + ".txt", passband)

        # Write varsource file
        if len(self.transits) != 0:
            with open(filename_varsource, "w") as file:
                file.write("".join([str(i) + "    " + filename_times + str(i) + appendix + ".txt\n" for i in varStars]))

            sim["Sky/VariableSourceList"] = filename_varsource

        # Setting up simulation
        sim = sim


        quarter = 0


        sim["ObservingParameters/StarCatalogFile"] = filename_stars
        sim["ObservingParameters/NumExposures"]    = numExp
        
    
    def getContaminatedMagnitude(self, times: np.ndarray, passband: Filter):
        """Get the total cotaminated magnitude of all stars with their transits"""
        transitted = [x.host for x in self.transits]
        total_flux = np.zeros(times.shape)
        a = np.log(10)/2.5
        for star in self.stars:
            if star in transitted:
                transit = self.transits[transitted.index(star)]
                total_flux += np.exp(-a * star.getMagnitude() + transit.getDeltaMag(times, passband))
            else:
                total_flux += np.exp(-a * star.getMagnitude(passband))
        
        return -2.5*np.log10(total_flux)
    

    def getContaminatedBaseFlux(self, passband: Filter):
        """Get the total cotaminated flux of all stars without any transits"""
        if (passband in self.baseFluxes):
            return self.baseFluxes[passband]
        else:
            return np.sum([star.getFlux(passband) for star in self.stars])

    def getContaminatedFlux(self, times: np.ndarray, passband: Filter):
        """Get the total cotaminated flux of all stars with their transits"""
        transitted = [x.host for x in self.transits]
        total_flux = np.zeros(times.shape)
        for star in self.stars:
            if star in transitted:
                transit = self.transits[transitted.index(star)]
                total_flux += transit.getFlux(times, passband)
            else:
                total_flux += star.getFlux(passband)
        # Special case to solve inaccurate baselevel
        if passband in self.baseFluxes and len(self.transits) == 1:
            return total_flux * self.getContaminatedBaseFlux(passband) / np.sum([star.getFlux(passband) for star in self.stars])
        else:
            return total_flux
    
    def getContaminatedFluxNorm(self, times: np.ndarray, passband: Filter):
        flux = self.getContaminatedFlux(times, passband)
        return flux / self.getContaminatedBaseFlux(passband)


class LimbDarkening(funcFit.OneDFit):
    """
    Class for fitting the Limb Darkening Coefficients.
    """

    def __init__(self):
        funcFit.OneDFit.__init__(self, ['u1', 'u2'])

    def evaluate(self, x):

        model = 'quadratic'

        if model == 'linear':
            coef = ['u1']
        else:
            coef = ['u1', 'u2']

        y = 1. - self['u1']*(1. - x)  # Linear model
        if model == 'quadratic':   y = y - self['u2']*(1. - x)**2
        if model == 'square-root': y = y - self['u2']*(1. - np.sqrt(x))
        if model == 'logarithmic': y = y - self['u2']*x*np.log(x)
        if model == 'exponential': y = y - self['u2']/(1 - np.exp(x))
        if model == 'nonlinear':   y = 1. - self['u1']*(1 - np.sqrt(x)) - self['u2']*(1. - x)

        return y

class LightCurveGeneral:

    def __init__(self, times: np.ndarray, flux: np.ndarray, stddev: np.ndarray):
        self.times = times
        self.flux = flux
        self.stddev = stddev
    

    def applyMedian(self, N):
        Nbins = int(len(self.times) / N)

        ts = []
        fs = []
        es = []
        for i in range(Nbins):
            mn = N*i
            mx = N*(i+1)
            ts.append(self.times[int((mn+mx)/2)])
            fs.append(np.median(self.flux[mn:mx]))
            es.append(np.median(self.stddev[mn:mx])/np.sqrt(N))
    
        return LightCurveGeneral(*np.array([ts, fs, es]))

    def fitTransit(self, host: Star, guess: TransitObject, passband: Filter, contaminationLevel=0, rescaling="default", pandas=False, **kwargs):
        if type(rescaling) == str and rescaling == "default":
            rescaling = np.array([10, 1, 1/20, 1, 5])
        elif rescaling is None:
            rescaling = np.array([1, 1, 1, 1, 1])

        baseLevel = host.getFlux(passband)

        fittingTransit = Transit(host, deepcopy(guess))
        defaults = guess.paramList[:-2]
        remains = guess.paramList[-2:]

        def model(times, k,  t0 , p , a , i):
            fittingTransit.transitObject._setParameters(np.concatenate([np.array([k,  t0 , p , a , i])/rescaling, remains]))
            return baseLevel * (fittingTransit.getFluxNorm(times, passband) + contaminationLevel)

        defaults_rescaled = defaults*rescaling
        
        result = fit(self.times, self.flux, self.stddev, model, defaults_rescaled, pandas=pandas, silent=False, **kwargs)/rescaling[:,np.newaxis]
        
        if pandas:
            import pandas as pd
            return result.append(pd.DataFrame(np.transpose([remains, [0, 0]]), index=["e", "w"], columns=["value", "error"]))
        else:
            return np.concatenate([result, np.transpose([remains, [0, 0]])])


class LightCurve(LightCurveGeneral):
    
    # Correction to account for imperfect Poisson distribution
    # The real flux is approximately distributed as the sum of a Poisson and a normal distribution
    # This value corresponds to sigma**2 - mu of the added normal distribution
    poissonCorrection = 45740.876
    #poissonCorrection = 45740000.876

    def __init__(self, times: np.ndarray, flux: np.ndarray):
        
        stddev = np.sqrt(flux + self.poissonCorrection)
        super().__init__(times, flux, stddev)
    
    @classmethod
    def fromSimFile(cls, simFile , starID):
        lc = simFile.getLightCurve(starID)
        return LightCurve(lc[0]/3600, lc[1])


class PhotometryData:

    def __init__(self, stars: List[Star], lightCurves: List[LightCurveGeneral], passbands: List[Filter], baseInterval: tuple = None):
        self.stars = stars
        self.lightCurves = lightCurves
        self.times = self.lightCurves[0].times
        if not all([lc.times.shape == self.times.shape for lc in self.lightCurves]):
            raise ValueError("All lightcurves must have identical timeseries")
        self.passbands = passbands
        if not len(self.lightCurves) == len(passbands):
            raise ValueError("The same number of lightCurves and passbands must be supplied")
        
        self.baseValues = None
        if baseInterval is not None:
            self.baseValues = self.calculateBaseValues(baseInterval[0], baseInterval[1])

    @classmethod
    def fromSimFiles(cls, stars: List[Star], simfiles: List[SimFile], ID: int, passbands: List[Filter], baseInterval: tuple = None):
        return PhotometryData(stars, [LightCurve.fromSimFile(sim, ID) for sim in simfiles] , passbands, baseInterval)

    def getLightCurve(self, passband: Filter):
        return self.lightCurves[self.passbands.index(passband)]

    def getFluxData(self, passband: Filter):
        return self.getLightCurve(passband).flux
    
    def getStddevData(self, passband: Filter):
        return self.getLightCurve(passband).stddev
    

    def getLikelihoodRatio(self, transit1: Transit, transit2: Transit):
        config1 = StarConfiguration([transit1.host, transit2.host], [transit1], self.baseValues)
        config2 = StarConfiguration([transit1.host, transit2.host], [transit2], self.baseValues)
        f1 = np.array([config1.getContaminatedFlux(self.times, band) for band in self.passbands])
        f2 = np.array([config2.getContaminatedFlux(self.times, band) for band in self.passbands])
        x = np.array([lc.flux for lc in self.lightCurves])
        modelRatio = f1/f2
        modelDifference = f1 - f2


        lnlambda = np.sum(x*np.log(modelRatio) - modelDifference)

        # p1
        mu = np.sum(f1*np.log(modelRatio))
        sigma = np.sqrt(np.sum(f1*np.log(modelRatio)**2))
        res = np.sum(modelDifference)
        statistic = (lnlambda - mu + res)/sigma
        p1 = norm.cdf(statistic)

        # p2
        mu2 = np.sum(-f2*np.log(modelRatio))
        sigma2 = np.sqrt(np.sum(f2*np.log(modelRatio)**2))
        statistic2 = (-lnlambda - mu2 - res)/sigma2
        p2 = norm.cdf(statistic2)
    
        return lnlambda, p1, p2

    def testModels(self, guess: TransitObject, significance: float, **kwargs):
        if len(self.stars) != 2:
            raise NotImplementedError("Deciding between multiple stars not yet implemented")
        
        star1 = self.stars[0]
        star2 = self.stars[1]
        fit1 = self.fitAll(star1, guess, showErrors=False, **kwargs)
        fit2 = self.fitAll(star2, guess, showErrors=False, **kwargs)
        l, p1, p2 = self.getLikelihoodRatio(fit1, fit2)
    
    def calculateBaseValues(self, minimum, maximum):
        mask = np.logical_and(self.times > minimum, self.times < maximum)
        return {band: np.mean(lc.flux[mask]) for lc, band in zip(self.lightCurves, self.passbands)}

    def fitAll(self, host: Star, guess: TransitObject, constants=[], method=None, rescaling="default", pandas=False, showErrors=True, **kwargs):
        """Fit data taking into """
        if type(rescaling) == str and rescaling == "default":
            rescaling = np.array([10, 1, 1/20, 1, 5])
        elif rescaling is None:
            rescaling = np.array([1, 1, 1, 1, 1])

        fittingTransit = Transit(host, deepcopy(guess))
        configuration = StarConfiguration(self.stars, [fittingTransit], self.baseValues)

        defaults = guess.paramList[:-2]
        remains = guess.paramList[-2:]
        constantIndices = [TransitObject.names.index(c) for c in constants]
        constantValues = defaults.take(constantIndices)
        def model(times, k,  t0 , p , a , i):
            params = np.array([k,  t0 , p , a , i])
            params.put(constantIndices, constantValues)
            fittingTransit.transitObject._setParameters(np.concatenate([params/rescaling, remains]))
            return np.array([configuration.getContaminatedFlux(times, band) for band in self.passbands])

        fluxes = np.array([lc.flux for lc in self.lightCurves])
        stds = np.array([lc.stddev for lc in self.lightCurves])

        defaults_rescaled = defaults*rescaling
        result = fit(self.times, fluxes, stds, model, defaults_rescaled, pandas=pandas, method=method, silent=False, **kwargs)/rescaling[:,np.newaxis]
        
        # Check if the fit produced something
        if pandas:
            params = np.array(result.value)
        else:
            params = result[:,0]
        params.put(constantIndices, constantValues)
        if (all(params == defaults)):
            print("FIT FAILED")
            return False

        if pandas:
            import pandas as pd
            return result.append(pd.DataFrame(np.transpose([remains, [0, 0]]), index=["e", "w"], columns=["value", "error"]))
        elif showErrors == True:
            return np.concatenate([result, np.transpose([remains, [0, 0]])])
        else:
            return Transit(host, Planet(*result[:,0]))

    def configurationFromFit(self, host: Star, fitParameters: List[float]) -> StarConfiguration:
        """Convenience function, quickly create StarConfiguration object from fit parameters"""
        return StarConfiguration(self.stars, [Transit(host, Planet(*fitParameters))], self.baseValues)

    def fitRatio(self, host: Star, band1: Filter, band2: Filter, guess: TransitObject, method=None, rescaling="default", pandas=False, **kwargs):
        """UNFINISHED Fit the ratio of the fluxes in two bandpasses"""
        if type(rescaling) == str and rescaling == "default":
            rescaling = np.array([10, 1, 1/20, 1, 5])
        elif rescaling is None:
            rescaling = np.array([1, 1, 1, 1, 1])


        fittingTransit = Transit(host, deepcopy(guess))
        configuration = StarConfiguration(self.stars, [fittingTransit], self.baseValues)

        defaults = guess.paramList[:-2]
        remains = guess.paramList[-2:]

        def model(times, k,  t0 , p , a , i):
            fittingTransit.transitObject._setParameters(np.concatenate([np.array([k,  t0 , p , a , i])/rescaling, remains]))
            f1 = configuration.getContaminatedFlux(times, band1)
            f2 = configuration.getContaminatedFlux(times, band2)
            return f1/f2

        defaults_rescaled = defaults*rescaling
        
        lc1 = self.getLightCurve(band1)
        lc2 = self.getLightCurve(band2)
        fluxRatio = lc1.flux / lc2.flux
        stds = np.sqrt((lc1.stddev/lc2.flux)**2 + (lc1.flux*lc2.stddev/lc2.flux**2)**2)
        result = fit(self.times, fluxRatio, stds, model, defaults_rescaled, pandas=pandas, silent=False, **kwargs)/rescaling[:,np.newaxis]
        
        if pandas:
            import pandas as pd
            return result.append(pd.DataFrame(np.transpose([remains, [0, 0]]), index=["e", "w"], columns=["value", "error"]))
        else:
            return np.concatenate([result, np.transpose([remains, [0, 0]])])
