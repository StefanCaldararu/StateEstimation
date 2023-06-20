import numpy as np
import math
import matplotlib.pyplot as plt

class randomWalk(object):
    def __init__(self, dt):
        self.dt = dt
        self.widthOfDistr = 1/4
        self.crntValueOfRndmWalk = 0
        self.dCrntValue = 0
        self.sigma = 0.016289174978068626
        self.maxv = 0.04/8
        self.maxa = 0.00875/16
        self.maxNudge = 0.15*self.sigma

    def nudge(self):
        myNudge = -self.crntValueOfRndmWalk/self.widthOfDistr
        if(myNudge>1):
            myNudge = 1
        elif(myNudge<-1):
            myNudge = -1
        return myNudge
    def step(self):
        mean = self.maxNudge*self.nudge()
        a = np.random.normal(mean, self.sigma)
        if(a>self.maxa):
            a = self.maxa
        elif(-a>self.maxa):
            a = -self.maxa
        self.crntValueOfRndmWalk = self.crntValueOfRndmWalk+self.dCrntValue+a
        self.dCrntValue = self.dCrntValue+a
        if(self.dCrntValue>self.maxv):
            self.dCrntValue = self.maxv
        elif(-self.dCrntValue > self.maxv):
            self.dCrntValue = -self.maxv
        return self.crntValueOfRndmWalk