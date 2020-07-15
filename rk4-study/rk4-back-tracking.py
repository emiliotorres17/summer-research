#!/usr/bin/env python3
"""========================================================================
Purpose:
    The purpose of this script is to solve the following system of ODEs
    using the RK4 method.

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
from numpy import linspace, zeros, exp, sqrt,\
                    amax
import matplotlib.pyplot as plt
#=========================================================================#
# User defined functions                                                  #
#=========================================================================#
#-------------------------------------------------------------------------#
# dy1/dx function                                                         #
#-------------------------------------------------------------------------#
def func1(
        Y1,
        Y2,
        Y3,
        X):

    """ Calculating the Y1' function """
    #---------------------------------------------------------------------#
    # dy1/dx function                                                     #
    #---------------------------------------------------------------------#
    dy1     = -Y1 + sqrt(Y2) - Y3*exp(2.0*X)

    return dy1
#-------------------------------------------------------------------------#
# dy2/dx function                                                         #
#-------------------------------------------------------------------------#
def func2(
        Y1,
        Y2,
        Y3,
        X):

    """ Calculating the Y2' function """
    #---------------------------------------------------------------------#
    # dy1/dx function                                                     #
    #---------------------------------------------------------------------#
    dy2     = -2.0*Y1**2.0

    return dy2
#-------------------------------------------------------------------------#
# dy3/dx function                                                         #
#-------------------------------------------------------------------------#
def func3(
        Y1,
        Y2,
        Y3,
        X):

    """ Calculating the Y3' function """
    #---------------------------------------------------------------------#
    # dy1/dx function                                                     #
    #---------------------------------------------------------------------#
    dy3     = -3.0*Y1*Y2

    return dy3
#=========================================================================#
# Main                                                                    #
#=========================================================================#
if __name__ == '__main__':
    #---------------------------------------------------------------------#
    # Main preamble                                                       #
    #---------------------------------------------------------------------#
    call(['clear'])
    sep         = os.sep
    pwd         = os.getcwd()
    media_path  = pwd + '%cmedia%c'                 %(sep, sep)
    #---------------------------------------------------------------------#
    # Domain variables                                                    #
    #---------------------------------------------------------------------#
    num     = [2, 4, 8, 16,32, 64, 128, 256, 512, 2048, 4096]
    err1    = zeros(len(num))
    err2    = zeros(len(num))
    err3    = zeros(len(num))
    dx_vec  = zeros(len(num))
    for k in range(0, len(num)):
        N           = num[k]
        dx          = 1.0/float(N)
        dx_vec[k]   = dx
        #-----------------------------------------------------------------#
        # Initial condition                                               #
        #-----------------------------------------------------------------#
        x0      = 1.0000000000000
        y1_0    = 0.3678794411199
        y2_0    = 0.1353352832401
        y3_0    = 0.0497870684156
        #-----------------------------------------------------------------#
        # Preallocating solution vectors                                  #
        #-----------------------------------------------------------------#
        x       = zeros(N+1)
        y1_vec  = zeros(N+1)
        y2_vec  = zeros(N+1)
        y3_vec  = zeros(N+1)
        #-----------------------------------------------------------------#
        # Saving the first time step                                      #
        #-----------------------------------------------------------------#
        x[0]        = x0
        y1_vec[0]   = y1_0
        y2_vec[0]   = y2_0
        y3_vec[0]   = y3_0
        #-----------------------------------------------------------------#
        # Solution variables                                              #
        #-----------------------------------------------------------------#
        xexact  = linspace(1.0, 0.0, N+1)
        print(xexact)
        y1_sol  = exp(-xexact)
        print(y1_sol)
        y2_sol  = exp(-2.0*xexact)
        print(y2_sol)
        y3_sol  = exp(-3.0*xexact)
        print(y3_sol)
        #-----------------------------------------------------------------#
        # Looping over the domain                                         #
        #-----------------------------------------------------------------#
        for i in range(1,N+1):
            #-------------------------------------------------------------#
            # Finding k1,n                                                #
            #-------------------------------------------------------------#
            k1_1    = func1(y1_0, y2_0, y3_0, x0)
            k1_2    = func2(y1_0, y2_0, y3_0, x0)
            k1_3    = func3(y1_0, y2_0, y3_0, x0)
            print('k1_1 = %.6f'         %(k1_1))
            print('k1_2 = %.6f'         %(k1_2))
            print('k1_3 = %.6f'         %(k1_3))
            print('\n')
            #-------------------------------------------------------------#
            # Finding k2,n                                                #
            #-------------------------------------------------------------#
            k2_1    = func1(y1_0 - 0.5*dx*k1_1, y2_0 - 0.5*dx*k1_2, y3_0 - 0.5*dx*k1_3, x0 - 0.5*dx)
            k2_2    = func2(y1_0 - 0.5*dx*k1_1, y2_0 - 0.5*dx*k1_2, y3_0 - 0.5*dx*k1_3, x0 - 0.5*dx)
            k2_3    = func3(y1_0 - 0.5*dx*k1_1, y2_0 - 0.5*dx*k1_2, y3_0 - 0.5*dx*k1_3, x0 - 0.5*dx)
            print('k2_1 = %.6f'         %(k2_1))
            print('k2_2 = %.6f'         %(k2_2))
            print('k2_3 = %.6f'         %(k2_3))
            print('\n')
            #-------------------------------------------------------------#
            # Finding k3,n                                                #
            #-------------------------------------------------------------#
            k3_1    = func1(y1_0 - 0.5*dx*k2_1, y2_0 - 0.5*dx*k2_2, y3_0 - 0.5*dx*k2_3, x0 - 0.5*dx)
            k3_2    = func2(y1_0 - 0.5*dx*k2_1, y2_0 - 0.5*dx*k2_2, y3_0 - 0.5*dx*k2_3, x0 - 0.5*dx)
            k3_3    = func3(y1_0 - 0.5*dx*k2_1, y2_0 - 0.5*dx*k2_2, y3_0 - 0.5*dx*k2_3, x0 - 0.5*dx)
            print('k3_1 = %.6f'         %(k3_1))
            print('k3_2 = %.6f'         %(k3_2))
            print('k3_3 = %.6f'         %(k3_3))
            print('\n')
            #-------------------------------------------------------------#
            # Finding k4,n                                                #
            #-------------------------------------------------------------#
            k4_1    = func1(y1_0 - dx*k3_1, y2_0 - dx*k3_2, y3_0 - dx*k3_3, x0-dx)
            k4_2    = func2(y1_0 - dx*k3_1, y2_0 - dx*k3_2, y3_0 - dx*k3_3, x0-dx)
            k4_3    = func3(y1_0 - dx*k3_1, y2_0 - dx*k3_2, y3_0 - dx*k3_3, x0-dx)
            print('k4_1 = %.6f'         %(k4_1))
            print('k4_2 = %.6f'         %(k4_2))
            print('k4_3 = %.6f'         %(k4_3))
            print('\n')
            #-------------------------------------------------------------#
            # Finding k4,n                                                #
            #-------------------------------------------------------------#
            y1      = y1_0 - (dx/6.0)*(k1_1 + 2.0*k2_1 + 2.0*k3_1 + k4_1)
            y2      = y2_0 - (dx/6.0)*(k1_2 + 2.0*k2_2 + 2.0*k3_2 + k4_2)
            y3      = y3_0 - (dx/6.0)*(k1_3 + 2.0*k2_3 + 2.0*k3_3 + k4_3)
            #-------------------------------------------------------------#
            # Updating the y values                                       #
            #-------------------------------------------------------------#
            x0      -= dx
            y1_0    = y1
            y2_0    = y2
            y3_0    = y3
            #-------------------------------------------------------------#
            # Updating the y values                                       #
            #-------------------------------------------------------------#
            x[i]        = x0
            y1_vec[i]   = y1_0
            y2_vec[i]   = y2_0
            y3_vec[i]   = y3_0
            print('iteration --> %i'        %(k))
            print('y1 = %.5f'               %(y1_vec[i]))
            print('y2 = %.5f'               %(y2_vec[i]))
            print('y3 = %.5f'               %(y3_vec[i]))
            print('\n')
        err1[k]     = amax(abs(y1_sol - y1_vec))
        err2[k]     = amax(abs(y2_sol - y2_vec))
        err3[k]     = amax(abs(y3_sol - y3_vec))
    #---------------------------------------------------------------------#
    #  Write solutions values                                             #
    #---------------------------------------------------------------------#
    f   = open(media_path + 'back-tracking-solutions.dat', 'w')
    out = ''
    for i in range(len(y1_vec)-1, -1,-1):
        out += '%25.13f\t%25.13f\t%25.13f\t%25.13f\n'\
                        %(x[i], y1_vec[i], y2_vec[i], y3_vec[i])  
    f.write(out)
    f.close()
    #---------------------------------------------------------------------#
    # Font settings                                                       #
    #---------------------------------------------------------------------#
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    SMALL_SIZE = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 12
    plt.rc('font',      size=SMALL_SIZE)            # controls default text sizes
    plt.rc('axes',      titlesize=SMALL_SIZE)       # fontsize of the axes title
    plt.rc('axes',      labelsize=MEDIUM_SIZE)      # fontsize of the x and y labels
    plt.rc('xtick',     labelsize=SMALL_SIZE)       # fontsize of the tick labels
    plt.rc('ytick',     labelsize=SMALL_SIZE)       # fontsize of the tick labels
    plt.rc('legend',    fontsize=SMALL_SIZE)        # legend fontsize
    plt.rc('figure',    titlesize=BIGGER_SIZE)      # fontsize of the figure title
    #---------------------------------------------------------------------#
    # Solution 1                                                          #
    #---------------------------------------------------------------------#
    plt.plot(x, y1_vec, 'r*', label='RK4')
    plt.plot(xexact, y1_sol, 'k--', lw=1.75, label='Exact')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(media_path + 'y1-back-tracking.png')
    plt.clf()
    #---------------------------------------------------------------------#
    # Solution 2                                                          #
    #---------------------------------------------------------------------#
    plt.plot(x, y2_vec, 'r*', label='RK4')
    plt.plot(xexact, y2_sol, 'k--', lw=1.75, label='Exact')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(media_path + 'y2-back-tracking.png')
    plt.clf()
    #---------------------------------------------------------------------#
    # Solution 3                                                          #
    #---------------------------------------------------------------------#
    plt.plot(x, y3_vec, 'r*', label='RK4')
    plt.plot(xexact, y3_sol, 'k--', lw=1.75, label='Exact')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(media_path + 'y3-back-tracking.png')
    plt.clf()
    #---------------------------------------------------------------------#
    # Error plot                                                          #
    #---------------------------------------------------------------------#
    plt.loglog(dx_vec, err1, 'ro-', lw=1.5, label='$y_{1}$')
    plt.loglog(dx_vec, err2, 'bo-', lw=1.5, label='$y_{2}$')
    plt.loglog(dx_vec, err3, 'ko-', lw=1.5, label='$y_{3}$')
    plt.loglog(dx_vec, 0.1*dx_vec**4.0 , 'g', lw=1.5, label="$\sim c_{1} x^{4}$")
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig(media_path + 'error-back-tracking.png')
    plt.clf()

    print('**** Successful run ****')
    sys.exit(0)