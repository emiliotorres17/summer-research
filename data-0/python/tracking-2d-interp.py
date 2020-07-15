#!/usr/bin/env python3
"""========================================================================
Purpose:
    The purpose of this script is to experiment with the RK4 and
    interpolation schemes for back tracking a particle.

Author:
    Emilio Torres
========================================================================"""
#=========================================================================#
# Preamble                                                                #
#=========================================================================#
#-------------------------------------------------------------------------#
# Python packages                                                         #
#-------------------------------------------------------------------------#
import os
import sys
from subprocess import call
from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator  as Reginterp
#=========================================================================#
# User defined functions                                                  #
#=========================================================================#
#-------------------------------------------------------------------------#
# Ghost cells                                                             #
#-------------------------------------------------------------------------#
def ghost_cells(
        uF):

    """ Adding in ghost cells for 0 and 2*pi """
    #---------------------------------------------------------------------#
    # Preallocating space                                                 #
    #---------------------------------------------------------------------#
    dim     = uF.shape
    field   = zeros((dim[0] + 2, dim[1] + 2,  dim[2] + 2, dim[3]))
    #---------------------------------------------------------------------#
    # Adding in the ghost cells (left boundary)                           #
    #---------------------------------------------------------------------#
    for k in range(0, 64):
        for j in range(0, 64):
            field[0,j+1,k+1]    = 0.5*(uF[0,j,k] + uF[-1,j,k])
            field[-1,j+1,k+1]   = 0.5*(uF[0,j,k] + uF[-1,j,k])

    for k in range(0, 64):
        for i in range(0, 64):
            field[i+1,0,k+1] = 0.5*(uF[i,0,k] + uF[i,-1,k])
            field[i+1,-1,k+1] = 0.5*(uF[i,0,k] + uF[i,-1,k])

    for j in range(0, 64):
        for i in range(0, 64):
            field[i+1,j+1,0] = 0.5*(uF[i,j,0] + uF[i,j,0])
            field[i+1,j+1,-1] = 0.5*(uF[i,j,0] + uF[i,j,-1])


    test    = ''
    for j in range(0,66):
        for i in range(0,66):
            test += '%35.18E'           %(field[i,j, 0, 0])
        test += '\n'
    f   = open('test.dat', 'w')
    f.write(test)
    f.close()
    sys.exit(24)

    return field
#-------------------------------------------------------------------------#
# Ghost cells vector                                                      #
#-------------------------------------------------------------------------#
def ghost_cell_vectors():

    """ Generating the x, y, and z vectors with ghost cells """
    #---------------------------------------------------------------------#
    # Generating vector                                                   #
    #---------------------------------------------------------------------#
    N       = 64
    dx      = 2.0*pi/float(N)
    X       = zeros(66)
    xtemp   = linspace(0.5*dx, 2.0*pi-0.5*dx, N)
    X[0]    = 0.0
    X[-1]   = 2.0*pi
    X[1:-1] = xtemp
    Y       = copy(X)
    Z       = copy(X)

    return X, Y, Z
#-------------------------------------------------------------------------#
# Ghost cells vector                                                      #
#-------------------------------------------------------------------------#
def cell_vectors():

    """ Generating the x, y, and z vectors with ghost cells """
    #---------------------------------------------------------------------#
    # Generating vector                                                   #
    #---------------------------------------------------------------------#
    N       = 64
    dx      = 2.0*pi/float(N)
    X       = zeros(66)
    xtemp   = linspace(0.5*dx, 2.0*pi-0.5*dx, N)
    Y       = copy(X)
    Z       = copy(X)

    return X, Y, Z
#=========================================================================#
# Main                                                                    #
#=========================================================================#
if __name__ == '__main__':
    #---------------------------------------------------------------------#
    # Main preamble                                                       #
    #---------------------------------------------------------------------#
    call(['clear'])
    sep             = os.sep
    pwd             = os.getcwd()
    media_path      = pwd + '%c..%cmedia%c'         %(sep, sep, sep)
    data_path       = pwd + '%c..%cdata%c'          %(sep, sep, sep)
    #---------------------------------------------------------------------#
    # Loading data                                                        #
    #---------------------------------------------------------------------#
    print('Loading data:')
    u1      = load(data_path + 'velocity1.npy')
    u1      = ghost_cells(u1)
    print('\tvelocity-1')
    u2      = load(data_path + 'velocity2.npy')
    u2      = ghost_cells(u2)
    print('\tvelocity-2')
    u3      = load(data_path + 'velocity3.npy')
    u3      = ghost_cells(u3)
    print('\tvelocity-3')
    time    = load(data_path + 'time.npy')
    print('\ttime')
    #---------------------------------------------------------------------#
    # Domain variables                                                    #
    #---------------------------------------------------------------------#
    (x,y,z) = ghost_cell_vectors()
    #---------------------------------------------------------------------#
    # Interpolating function                                              #
    #---------------------------------------------------------------------#
    u1func  = Reginterp((x,y,z,time), u1, method='linear')
    u2func  = Reginterp((x,y,z,time), u2, method='linear')
    u3func  = Reginterp((x,y,z,time), u3, method='linear')
    test    = ''
    for k in range(0,66):
        for j in range(0,66):
            for i in range(0,66):
                test += '%35.18E'           %(u1[i,j,k, 123])
            test += '\n'
        test += '\n\n'
    f   = open('test.dat', 'w')
    f.write(test)
    f.close()
    sys.exit(24)
    #---------------------------------------------------------------------#
    # Initial conditions                                                  #
    #---------------------------------------------------------------------#
    x1_0    = x[13]
    x2_0    = y[43]
    x3_0    = z[21]
    t       = time[-1]
    #=====================================================================#
    # RK4 back tracking                                                   #
    #=====================================================================#
    out = ''
    for i in range(455, 192, -1):
        print(i)
        print(x2_0)
        dt = abs(time[i] - time[i-1])
        #-----------------------------------------------------------------#
        # K_{i,1}                                                         #
        #-----------------------------------------------------------------#
        k11     = u1func((x1_0, x2_0, x3_0, t))
        k21     = u2func((x1_0, x2_0, x3_0, t))
        k31     = u3func((x1_0, x2_0, x3_0, t))
        #-----------------------------------------------------------------#
        # K_{i,2}                                                         #
        #-----------------------------------------------------------------#
        k12     = u1func((x1_0 - 0.5*dt*k11, x2_0 - 0.5*dt*k21, \
                            x3_0 - 0.5*dt*k31, t - 0.5*dt))
        k22     = u2func((x1_0 - 0.5*dt*k11, x2_0 - 0.5*dt*k21, \
                            x3_0 - 0.5*dt*k31, t - 0.5*dt))
        k32     = u3func((x1_0 - 0.5*dt*k11, x2_0 - 0.5*dt*k21, \
                            x3_0 - 0.5*dt*k31, t - 0.5*dt))
        #-----------------------------------------------------------------#
        # K_{i,3}                                                         #
        #-----------------------------------------------------------------#
        k13     = u1func((x1_0 - 0.5*dt*k12, x2_0 - 0.5*dt*k22, \
                            x3_0 - 0.5*dt*k32, t - 0.5*dt))
        k23     = u2func((x1_0 - 0.5*dt*k12, x2_0 - 0.5*dt*k22, \
                            x3_0 - 0.5*dt*k32, t - 0.5*dt))
        k33     = u3func((x1_0 - 0.5*dt*k12, x2_0 - 0.5*dt*k22, \
                            x3_0 - 0.5*dt*k32, t - 0.5*dt))
        #-----------------------------------------------------------------#
        # K_{i,4}                                                         #
        #-----------------------------------------------------------------#
        k14     = u1func((x1_0 - dt*k13, x2_0 - dt*k23, \
                            x3_0 - dt*k33, t - dt))
        k24     = u2func((x1_0 - dt*k13, x2_0 - dt*k23, \
                            x3_0 - dt*k33, t - dt))
        k34     = u3func((x1_0 - dt*k13, x2_0 - dt*k23, \
                            x3_0 - dt*k33, t - dt))
        #-----------------------------------------------------------------#
        # Back tracking                                                   #
        #-----------------------------------------------------------------#
        x1      = x1_0 - (dt/6.0) * (k11 + 2.*k12 + 2.*k13 + k14)
        x2      = x2_0 - (dt/6.0) * (k21 + 2.*k22 + 2.*k23 + k24)
        x3      = x3_0 - (dt/6.0) * (k31 + 2.*k32 + 2.*k33 + k34)
        #-----------------------------------------------------------------#
        # Print statement                                                 #
        #-----------------------------------------------------------------#
        out +=  'time = %20.16f ---> %20.16f ---> %20.16f\n'    %(t, t-dt, time[i-1])
        out +=  'x1 = %20.16f ---> %20.16f\n'                   %(x1_0, x1)
        out +=  'x2 = %20.16f ---> %20.16f\n'                   %(x2_0, x2)
        out +=  'x3 = %20.16f ---> %20.16f\n'                   %(x3_0, x3)
        out +=  '\n'
        #-----------------------------------------------------------------#
        # Updating values                                                 #
        #-----------------------------------------------------------------#
        t       = time[i-1]
        x1_0    = x1
        x2_0    = x2
        x3_0    = x3
    #---------------------------------------------------------------------#
    # Storing print out                                                   #
    #---------------------------------------------------------------------#
    f   = open('output.out', 'w')
    f.write(out)
    f.close()

    print('**** Successful run ****')
    sys.exit(0)


