# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:42:14 2025

@author: vishw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import time
import numba

import randomgen
rng = np.random.Generator(randomgen.Xoshiro512())

#Parameters of the ellipse
a = 1
b = 0.6
n = b/a
c = a*np.sqrt(1-n**2)


x_lower = np.linspace(-c, c, 100)
ybot = -b*np.sqrt(1-(x_lower/a)**2) + n*b/2
yborder = -ybot[-1]
foc = 0.5*c
x_upper = np.linspace(-c, c, 100)

def emissivity(theta):  #theta in radians from -pi/2 to pi/2
    # If B < 0 emissivity_up < emissivity_down else opposite
    # These parameters are fitted from Zhou et al, Joule
    # A = 0.524804
    # B = 0.37734
    # C = 2.880735
    # D = 0.27247
    
    
    A = 0.9
    B = 0.0
    C = 0.0
    D = 0.0
    
    emm = A + B*np.tanh(C*(theta-D))
    return emm

boolplot = False


curve_upper = 'Plane'  #'Ellipse', 'Plane', 'Parabola'
curve_lower = 'Ellipse'  #'Ellipse', 'Plane', 'Parabola'

def get_curve_upper(x):
    if curve_upper == 'Ellipse':
        return b*np.sqrt(1-(x/a)**2) - n*b/2  #ellipse
    elif curve_upper == 'Plane':
        return yborder*np.ones(x.size) #plane
    elif curve_upper == 'Parabola':
        y = -4*foc*x**2 + yborder + 4*foc*c**2
        return y

def get_curve_lower(x):
    if curve_lower == 'Ellipse':
        return -1*(b*np.sqrt(1-(x/a)**2) - n*b/2)  #ellipse
    elif curve_lower == 'Plane':
        return -yborder*np.ones(x.size) #plane
    elif curve_lower == 'Parabola':
        y = -1*(-4*foc*x**2 + yborder + 4*foc*c**2)
        return y
        

if boolplot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot([-a,a],[0,0], 'k')
    ax.plot([0,0],[-a,a], 'k')
    
    ytop = get_curve_upper(x_upper)
    ax.plot(x_upper, ytop, 'b')
    ybot = get_curve_lower(x_lower)
    ax.plot(x_lower, ybot, 'b')
    
    ax.plot([x_lower[0], x_upper[0]], [yborder, -yborder], 'bo')
    ax.plot([c, c], [yborder, -yborder], 'bo')



def get_intersection_with_curve_upper(p0, d):
    [x0, y0] = p0
    [dx, dy] = d
    
    if curve_upper == 'Ellipse':
        #Ellipse
        A = dy**2 + n**2*dx**2
        B = 2*(dy*(y0+n*b/2)+n**2*dx*x0)
        C = (y0 + n*b/2)**2 + n**2*x0**2 - b**2
        
        if B**2 - 4*A*C >= 0:
            alphaplus = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            alphaminus = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            alpha = max([alphaplus, alphaminus])
            return alpha
        else:
            return -1
    elif curve_upper == 'Plane':    
        #Plane
        if dy != 0:
            alpha = (yborder - y0)/dy                  
            return alpha
        else:
            return -1
    elif curve_upper == 'Parabola':
        #Parabola
        A = 4*foc*dx**2
        B = 8*foc*x0*dx + dy
        C = 4*foc*x0**2 + y0 - yborder - 4*foc*c**2
        if B**2 - 4*A*C >= 0:
            alphaplus = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            alphaminus = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            alpha = max([alphaplus, alphaminus])
            return alpha
        else:
            return -1
        
def get_intersection_with_curve_lower(p0, d):
    [x0, y0] = p0
    [dx, dy] = d
    
    if curve_lower == 'Ellipse':
        #Ellipse
        A = dy**2 + n**2*dx**2
        B = 2*(dy*(y0-n*b/2)+n**2*dx*x0)
        C = (y0 - n*b/2)**2 + n**2*x0**2 - b**2
        
        if B**2 - 4*A*C >= 0:
            alphaplus = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            alphaminus = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            alpha = max([alphaplus, alphaminus])
            return alpha
        else:
            return -1
    elif curve_lower == 'Plane':    
        #Plane
        if dy != 0:
            alpha = (-yborder - y0)/dy                  
            return alpha
        else:
            return -1
    elif curve_lower == 'Parabola':
        #Parabola
        A = 4*foc*dx**2
        B = 8*foc*x0*dx - dy
        C = 4*foc*x0**2 - y0 - yborder - 4*foc*c**2
        if B**2 - 4*A*C >= 0:
            alphaplus = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            alphaminus = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            alpha = max([alphaplus, alphaminus])
            return alpha
        else:
            return -1

def get_ellipse_lower(x):
    return -b*np.sqrt(1-(x/a)**2) + n*b/2

def get_normal_to_curve_upper(x):
    if curve_upper == 'Ellipse':
        #Ellipse
        y = get_curve_upper(x)
        normal = np.array([n**2*x/(y+n*b/2),1])
        normal /= np.linalg.norm(normal)
    elif curve_upper == 'Plane':    
        #Plane
        normal = np.array([0, -1])
    elif curve_upper == 'Parabola':
        #Parabola
        normal = np.array([8*foc*x, 1])
        normal /= np.linalg.norm(normal)
    return normal

def get_normal_to_curve_lower(x):
    if curve_lower == 'Ellipse':
        #Ellipse
        y = get_curve_lower(x)
        normal = np.array([n**2*x/(y-n*b/2),1])
        normal /= np.linalg.norm(normal)
    elif curve_lower == 'Plane':    
        #Plane
        normal = np.array([0, 1])
    elif curve_lower == 'Parabola':
        #Parabola
        normal = np.array([-8*foc*x, 1])
        normal /= np.linalg.norm(normal)
    return normal



def get_intersection_with_surfaces(p, d):
    [x0, y0] = p
    [dx, dy] = d
    
    listalpha = []
    listsurface = []
    
    alpha = get_intersection_with_curve_upper(p, d)
    if alpha > -1:                  
        yintersect = y0 + alpha*dy
        xintersect = x0 + alpha*dx
        if x_upper[0] < xintersect < c:
            if yintersect > 0:
                if alpha > 1e-9:
                    listalpha.append(alpha)
                    listsurface.append('upper')           

 
    alpha = get_intersection_with_curve_lower(p, d)
    if alpha > -1:                  
        yintersect = y0 + alpha*dy
        xintersect = x0 + alpha*dx
        if -a < xintersect < c:
            if alpha > 1e-9:
                listalpha.append(alpha)
                listsurface.append('lower')           

    
    if dx != 0:
        x1 = c
        alpha = (x1 - x0)/dx
        if alpha > 1e-9:
            listalpha.append(alpha)
            listsurface.append('window')
    
    if dx != 0:
        x1 = -c
        alpha = (x1 - x0)/dx
        if alpha > 1e-9:
            listalpha.append(alpha)
            listsurface.append('receiver')
    
    alpha = min(listalpha)
    surface = listsurface[np.argmin(listalpha)]
    
    return alpha, surface
    

Tamb = 300 #K
InitialPopulationPerBin = 1000 #i.e. Tamb**4 corresponds to intial population per bin
Nbins = 2
listenergy = np.ones(Nbins)*InitialPopulationPerBin
listborders = np.linspace(0, 1, Nbins+1)

areareceiver = 2*yborder
areabin = areareceiver/Nbins
areawindow = 2*yborder

energy_window = InitialPopulationPerBin/areabin*areawindow



if boolplot:
    Niterations = 1
    Nbatch = 1
else:
    Niterations = 1000
    Nbatch = 1000


def get_distribution():
    
    for i in range(Nbatch):
        #Selecting a source and direction of ray
        rbin = rng.random()
        pbins = np.concatenate(((listenergy/areabin)**4*areabin,
                                    np.array([(energy_window/areawindow)**4])*areawindow))
        pbins = pbins/pbins.sum()
        
        pbinsum = 0
        for ibin in range(len(pbins)):
            pbinsum += pbins[ibin]
            if pbinsum > rbin:
                break

        if ibin < Nbins: #emit from receiver
            
            if boolplot:
                print('Emitted from Receiver')
                
            intensity_emitted = 1  #receiver modelled as a black body

            listenergy[ibin] -= intensity_emitted
            ybin_lower = -yborder + ibin/Nbins*(2*yborder)
            ybin_upper = -yborder + (ibin+1)/Nbins*(2*yborder)


            rsource = rng.random()
            ysource = ybin_lower + rsource*(ybin_upper - ybin_lower)
    
            v = rng.random()
            
            phi = np.arcsin(2*v - 1)
            x = np.sin(phi)
            z = np.cos(phi) 
            
            xsource = -c
                            
            p_source = np.array([xsource, ysource])
            d_source = np.array([z, x]); d_source /= np.linalg.norm(d_source)
        else: #emit from window

            if boolplot:
                print('Emitted from Window')
            rsource = rng.random()
            ysource = -yborder + rsource*(2*yborder)
            
            v = rng.random()
            phi = np.arcsin(2*v - 1)
            x = np.sin(phi)
            z = np.cos(phi) 
            
            intensity_emitted = emissivity(phi)
                        
            p_source = np.array([c, ysource])
            d_source = np.array([-z, x]); d_source /= np.linalg.norm(d_source)
            
        #Getting ray trace
        if boolplot:
            ray = [p_source]
            xplot = [p_source[0]]
            yplot = [p_source[1]]
        boolgo = True
        while boolgo:
            alpha, surface = get_intersection_with_surfaces(p_source, d_source)
            x_, y_ = p_source + d_source*alpha*(1-1e-9)

            if boolplot:
                print(surface)
                xplot.append(x_)
                yplot.append(y_)
                ax.plot(xplot, yplot, 'c')
                plt.pause(0.01)
            if surface in ['upper', 'lower', 'window']:
                if surface == 'upper':
                    normal = get_normal_to_curve_upper(x_)
                    if np.dot(d_source, normal) > 0:
                        normal *= -1                
                    d_source = d_source - 2*np.dot(d_source, normal)*normal
                    p_source = np.array([x_, y_])
                    
                    if boolplot:
                        ray.append(p_source)
                elif surface == 'lower':
                    normal = get_normal_to_curve_lower(x_)
                    if np.dot(d_source, normal) > 0:
                        normal *= -1
                    d_source = d_source - 2*np.dot(d_source, normal)*normal
                    p_source = np.array([x_, y_])
                    if boolplot:
                        ray.append(p_source)
                elif surface == 'window':
                    normal = np.array([-1,0])
                    costheta = np.dot(d_source, normal)
                    d_source = d_source - 2*np.dot(d_source, normal)*normal
                    xsrc, ysrc = p_source
                    p_source = np.array([x_, y_])
                    if boolplot:
                        ray.append(p_source)
                    theta = np.arccos(-costheta)
                    phi = theta if (ysrc > y_) else -theta
                    fraction_absorbed = emissivity(phi)
                    intensity_emitted = intensity_emitted*(1-fraction_absorbed)
            else:
                boolgo = False
                if surface == 'receiver':
                    yfraction = (y_ + yborder)/(2*yborder)
                    ibin = int(yfraction*(1-1e-9)*Nbins)
                    listenergy[ibin] += intensity_emitted
                    if boolplot:
                        print(yfraction, ibin)
                if boolplot:
                    ray.append(np.array([x_, y_]))

n_equilibration = 100

if boolplot:
    get_distribution()
else:
    fig2 = plt.figure(); ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)
    
    listT = []
    listi = []
    listibinmax = []
    listibinmin = []
    listdT = []
    
    for i in range(Niterations):     
        print('Iteration', i)
        tick = time.time()
        get_distribution()
        srcdens = energy_window/areawindow
        recdens = listenergy.sum()/areareceiver
        avgT = (recdens/srcdens)*Tamb
        
        tock = time.time()
        ax2.cla()    
        listTbin = []
                
        for ibin in range(Nbins):
            xleft = ibin/Nbins
            xright = (ibin+1)/Nbins
            recdens = listenergy[ibin]/areabin
            srcdens = energy_window/areawindow
            ytop = (recdens/srcdens)*Tamb
            listTbin.append(ytop)
            ybot = 273.15
            #ax2.plot([xleft, xleft, xright, xright], [ybot, ytop, ytop, ybot],'r')
        #ax2.plot([0,1],[avgT, avgT],'b')
        #ax2.yaxis.label.set_text("Temperature of bin K")
        #ax2.xaxis.label.set_text('Bin Number')
        
        
        
        listi.append(i)
        listT.append(avgT)
        listdT.append(listTbin[1] - listTbin[0])
        
        ax2.cla()
        ax2.plot(listi, listdT, 'r')
        ax2.yaxis.label.set_text("Temperature difference between bins K")
        ax2.xaxis.label.set_text("Simulation Epoch of 1000 rays")
        if len(listT) > n_equilibration+5:
            Tdiff = np.mean(listdT[n_equilibration:])
            StdTdiff = np.std(listdT[n_equilibration:])
            ax2.text(0.02, 0.98, 
                     f"Avg dTbins = {Tdiff:.2f} K \nStd-dev of dTbins = {StdTdiff:.2f} K",
                     transform=ax2.transAxes,
                     ha='left', va='top'
                     )
        else:
            ax2.text(0.02, 0.98, 
                     "Avg dTbins = Waiting for equilibration \nStd-dev of dTbins = Waiting of equilibration",
                     transform=ax2.transAxes,
                     ha='left', va='top'
                     )
        
        
        ax3.cla()
        ax3.plot(listi, listT, 'r')
        ax3.yaxis.label.set_text("Average Temperature of Receiver K")
        ax3.xaxis.label.set_text("Simulation Epoch of 1000 rays")
        if len(listT) > n_equilibration + 5:
            AverageT = np.mean(listT[n_equilibration:])
            StdT = np.std(listT[n_equilibration:])
            print('Avg T K', AverageT, 'StandardDeviation K', StdT, 'dT bins', Tdiff, 'Std-dev of dTbins', StdTdiff)
            ax3.text(0.02, 0.98, 
                     f"Avg T = {AverageT:.2f} K\n Std Dev = {StdT:.2f} K",
                     transform=ax3.transAxes,
                     ha='left', va='top'
                     )
        else:
            ax3.text(0.02, 0.98, 
                     "Avg T = Waiting for equilibration \n Std Dev = Waiting for equilibration",
                     transform=ax3.transAxes,
                     ha='left', va='top'
                     )
        
        plt.pause(0.01)
        
