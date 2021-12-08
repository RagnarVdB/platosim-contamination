from contamination import *
from filters import Filter, filterP, filterB, filterR
import numpy as np
from platosim.simulation import Simulation
import os

print("Hello world")

myStar = Star(5500 , 6)
print("Magnitude     :" , myStar.getMagnitude(filterB))
print("MagnitudeB    :",myStar.getMagnitudeB())
print("TotalFlux", myStar.getTotalFlux(filterR))

myPlanet = Planet(0.4 , 50 , 60 , 0.4 , np.pi/2 , 0.3 , 0)
print(myPlanet.k)


myTransit = Transit(myStar , myPlanet)
myTimes = np.linspace(0,100,100)
print("getFlux :  " , myTransit.getFlux(myTimes , filterR))

myStarConfiguration = StarConfiguration([myStar] , [myTransit])


name = 'testInputfiles/testrun1'
inputfiles = "inputfiles/"
output = os.environ["PLATO_WORKDIR"]+"/platocon/tests/outputs/output_" + name
mySim = Simulation(name, configurationFile=inputfiles+"inputfile.yaml", outputDir=output, debug=False)




myAppendix = "test"

myStarConfiguration.setupSimulation(mySim , 20 , filterP , myAppendix, "testInputfiles/inputfiles")
