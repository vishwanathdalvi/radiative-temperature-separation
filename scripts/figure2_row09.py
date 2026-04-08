# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:42:14 2025

@author: vishw
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

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

def emissivity(theta):  #theta in radians from -pi/2 to pi/2.  
    A = 0.524804
    B = -0.37734 #B -ve means lower emissivity is up
    C = 2.880735
    D = 0.27247
    
    
    # A = 0.9
    # B = 0.0
    # C = 0.0
    # D = 0.0
    emm = A + B*np.tanh(C*(theta-D))
    return emm


effective_emissivity = scint.quad(emissivity, -np.pi/2, np.pi/2)[0]/np.pi

boolplot = False


curve_upper = 'Plane'  #'Ellipse', 'Plane', 'Parabola'
curve_lower = 'Plane'  #'Ellipse', 'Plane', 'Parabola'


Tamb = 300 #K
InitialPopulationPerBin = 1000 #i.e. Tamb**4 corresponds to intial population per bin
Nbins = 2
listenergy = np.ones(Nbins)*InitialPopulationPerBin
listborders = np.linspace(0, 1, Nbins+1)

threshold_intensity = 0.001

areareceiver = 2*yborder
areabin = areareceiver/Nbins
areawindow = 2*yborder


energy_window = InitialPopulationPerBin/areabin*areawindow

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
    



if boolplot:
    Niterations = 1
    Nbatch = 1
else:
    Niterations = 1000
    Nbatch = 1000


def get_distribution(energy_window):
    
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
            bool_window = False
            if boolplot:
                print('Emitted from Receiver')
                
            

            ybin_lower = -yborder + ibin/Nbins*(2*yborder)
            ybin_upper = -yborder + (ibin+1)/Nbins*(2*yborder)


            rsource = rng.random()
            ysource = ybin_lower + rsource*(ybin_upper - ybin_lower)
    
            v = rng.random()
            
            phi = np.arcsin(2*v - 1)
            x = np.sin(phi)
            z = np.cos(phi) 
            
            xsource = -c
            
            
            intensity_emitted = emissivity(phi)
                    
            listenergy[ibin] -= intensity_emitted
            
 
                            
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
                
            energy_window -= intensity_emitted
          
                        
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
            if surface in ['upper', 'lower', 'window', 'receiver']:
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
                    energy_window += intensity_emitted*fraction_absorbed
                    intensity_emitted = intensity_emitted*(1-fraction_absorbed)
                    if intensity_emitted < threshold_intensity:
                        boolgo = False
                        energy_window += intensity_emitted
                    
                if surface == 'receiver':
                    yfraction = (y_ + yborder)/(2*yborder)
                    ibin = int(yfraction*(1-1e-9)*Nbins)
                    normal = np.array([1,0])
                    costheta = np.dot(d_source, normal)
                    d_source = d_source - 2*np.dot(d_source, normal)*normal
                    xsrc, ysrc = p_source
                    p_source = np.array([x_, y_])
                    if boolplot:
                        ray.append(p_source)
                    if boolplot:
                        print(yfraction)
                    theta = np.arccos(-costheta)
                    phi = theta if (ysrc > y_) else -theta
                    fraction_absorbed = emissivity(phi)

                    listenergy[ibin] += intensity_emitted*fraction_absorbed
                    intensity_emitted = intensity_emitted*(1-fraction_absorbed)
                    if intensity_emitted < threshold_intensity:
                        boolgo = False
                        listenergy[ibin] += intensity_emitted
    return energy_window

n_equilibration = 100

if boolplot:
    _ = get_distribution(energy_window)
else:
    fig2 = plt.figure(); ax2 = fig2.add_subplot(211)
    ax3 = fig2.add_subplot(212)
    
    listTrec = []
    listTsrc = []
    listi = []
    listdT = []    
    
    for i in range(Niterations):     
        print('Iteration', i)
        energy_window = get_distribution(energy_window)
        initialdens = InitialPopulationPerBin/areabin
        srcdens = energy_window/areawindow
        recdens = listenergy.sum()/areareceiver
        avgTrec = (recdens/initialdens)*Tamb
        avgTsrc = (srcdens/initialdens)*Tamb
        
        ax2.cla()   
        ax3.cla()
        listTbins = []
        
        for ibin in range(Nbins):
            xleft = ibin/Nbins
            xright = (ibin+1)/Nbins
            recdens = listenergy[ibin]/areabin
            srcdens = energy_window/areawindow
            ytop = (recdens/initialdens)*Tamb
            listTbins.append(ytop)
            ybot = 270
        listi.append(i)
        listTrec.append(avgTrec)
        listTsrc.append(avgTsrc)
        listdT.append(listTbins[1]-listTbins[0])
        ax2.plot(listi, listdT, 'r')
        ax2.xaxis.label.set_text('Simulation Epoch of 1000 days')
        ax2.yaxis.label.set_text('Temperature Difference between bins K')
        
        ax3.plot(listi, listTrec, 'r')
        ax3.plot(listi, listTsrc, 'b')
        ax3.xaxis.label.set_text('Simulation Epoch of 1000 days')
        ax3.yaxis.label.set_text('Temperature of source (blue) and receiver (red) in K')
        
        if len(listTrec) > n_equilibration+5:            
            print('Avg Trec K', np.mean(listTrec[n_equilibration:]), 'StandardDeviation K', np.std(listTrec[n_equilibration:]))
            print('Avg Tsrc K', np.mean(listTsrc[n_equilibration:]), 'StandardDeviation K', np.std(listTsrc[n_equilibration:]))
            print('Diff Tbins', np.mean(listdT[n_equilibration:]), 'Stdev dTbins',np.std(listdT[n_equilibration:]))
            
            ax2.text(0.02, 0.98, 
                     f"Avg dTbins = { np.mean(listdT[n_equilibration:]):.2f} K \nStd-dev of dTbins = {np.std(listdT[n_equilibration:]):.2f} K",
                     transform=ax2.transAxes,
                     ha='left', va='top'
                     )

            ax3.text(0.02, 0.98, 
                     f"Avg T_source = { np.mean(listTsrc[n_equilibration:]):.2f} K \nStd-dev of T_source = {np.std(listTsrc[n_equilibration:]):.2f} K \n Avg T_receiver = { np.mean(listTrec[n_equilibration:]):.2f} K \nStd-dev of T_receiver = {np.std(listTrec[n_equilibration:]):.2f} K ",
                     transform=ax3.transAxes,
                     ha='left', va='top'
                     )
        
        plt.pause(0.01)
        
