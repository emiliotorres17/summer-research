"""=============================================================================
Description:
    244-coefficient truncated Volterra series ALES model static test program
    **** Notes:
            run `mpiexec -n 1 python ales244_static_test.py -h` for help
    
Authors:
    Colin Towery, colin.towery@colorado.edu
    
    Turbulence and Energy Systems Laboratory
    Department of Mechanical Engineering
    University of Colorado Boulder
    http://tesla.colorado.edu
    https://github.com/teslacu/teslapy.git
    https://github.com/teslacu/spectralLES.git
============================================================================="""
#==============================================================================#
# Preamble                                                                     #
#==============================================================================#
#------------------------------------------------------------------------------#
# Python packages                                                              #
#------------------------------------------------------------------------------#
from mpi4py import MPI
import numpy as np
import sys
import time
from math import sqrt
import argparse
from spectralLES import spectralLES
from teslacu import mpiWriter
from teslacu.fft import rfft3, irfft3, shell_average
from teslacu.stats import psum
#------------------------------------------------------------------------------#
# MPI                                                                          #
#------------------------------------------------------------------------------#
comm = MPI.COMM_WORLD


def timeofday():
    return time.strftime("%H:%M:%S")
#==============================================================================#
# Extend the spectralLES class                                                 #
#==============================================================================#
class ales244_solver(spectralLES):
    """=========================================================================
    Purpose:
        Just adding extra memory and the ales244 SGS model. By using the
        spectralLES class as a super-class and defining a subclass for each
        SGS model we want to test, spectralLES doesn't get cluttered with
        an excess of models over time.
     
     Author:
        Eric Stallcup
        ====================================================================="""

    #--------------------------------------------------------------------------#
    # Class Constructor                                                        #
    #--------------------------------------------------------------------------#
    def __init__(self, comm, N, L, nu, epsilon, Gtype, **kwargs):
        """
        Purpose:
            The purpose of this part of the code is to add variables to the
            solver.
        """
        super().__init__(comm, N, L, nu, epsilon, Gtype, **kwargs)
        #----------------------------------------------------------------------#
        # Adding global variables                                              #
        #----------------------------------------------------------------------#
        self.t_ds           = 15
        self.tau_hat        = np.empty((6, *self.nnk), dtype=complex)
        self.UU_hat         = np.empty_like(self.tau_hat)
        self.tau            = np.empty((6, *self.nnx))          # subgrid stress
        self.Sij            = np.empty((6, *self.nnx))          # subgrid stress
        self.Cs2            = np.empty((1, *self.nnx))          # subgrid stress
        self.k_test         = 15
        self.test_filter    = self.filter_kernel(self.k_test, Gtype)
    #--------------------------------------------------------------------------#
    # Subgrid stress calculations                                              #
    #--------------------------------------------------------------------------#
    def computeSource_ales244_SGS(self, H_244, **ignored):
        """=====================================================================
        Purpose:
            h_ij Fortran column-major ordering:  11,12,13,22,23,33
            equivalent ordering for spectralLES: 22,21,20,11,10,00

            **** Note:
                    sparse tensor indexing for ales244_solver UU_hat and tau_hat:
                    m == 0 -> ij == 22
                    m == 1 -> ij == 21
                    m == 2 -> ij == 20
                    m == 3 -> ij == 11
                    m == 4 -> ij == 10
                    m == 5 -> ij == 00

        H_244 - ALES coefficients h_ij for 244-term Volterra series
                truncation. H_244.shape = (6, 244)
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Defining domain variables                                            #
        #----------------------------------------------------------------------#
        t_ds        = self.t_ds             # simulation time
        #if t_ds > 200.0:
        #    C_DS    = 0.65                  # C_DS coefficient
        #else:
        #    C_DS    = 0.0
        C_DS        = 0.0
        C_BS        = 1.0 - C_DS            # C_BS coefficient
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        tau_hat     = self.tau_hat
        tau         = self.tau
        Sij         = self.Sij
        W_hat       = self.W_hat
        omega       = self.omega
        Cs2         = self.Cs2
        #----------------------------------------------------------------------#
        # Preallocating variables                                              #
        #----------------------------------------------------------------------#
        tau_BS      = np.empty((6, *self.nnx), dtype=np.float64)
        tau_hat_BS  = np.empty((6, *self.nnk), dtype=complex)
        tau_DS      = np.empty((6, *self.nnx), dtype=np.float64)
        Uhat        = np.empty((3, *self.nnx), dtype=np.float64)
        Uijhat      = np.empty((6, *self.nnx), dtype=np.float64)
        SSijT       = np.empty((6, *self.nnx), dtype=np.float64)
        Lij         = np.empty((6, *self.nnx), dtype=np.float64)
        Mij         = np.empty((6, *self.nnx), dtype=np.float64)
        #Sij         = np.empty((6, *self.nnx), dtype=np.float64)
        Sijhat      = np.empty((6, *self.nnx), dtype=np.float64)
        S           = np.empty([*self.nnx], dtype=np.float64)
        Shat        = np.empty([*self.nnx], dtype=np.float64)
        #Cs2         = np.empty([*self.nnx],dtype=np.float64)
        LijMij      = np.empty([*self.nnx],dtype=np.float64)
        MklMkl      = np.empty([*self.nnx],dtype=np.float64)
        #----------------------------------------------------------------------#
        # Not sure what this is                                                #
        #----------------------------------------------------------------------#
        W_hat[:] = self.les_filter*self.U_hat
        irfft3(self.comm, W_hat[0], self.W[0])
        irfft3(self.comm, W_hat[1], self.W[1])
        irfft3(self.comm, W_hat[2], self.W[2])
        #----------------------------------------------------------------------#
        # Getting the time                                                     #
        #----------------------------------------------------------------------#
        t0 = time.time()
        #----------------------------------------------------------------------#
        # Filtered Velocities on the LES grid                                  #
        #----------------------------------------------------------------------#
        Uhat[2] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2])).real
        Uhat[1] = irfft3(comm,self.test_filter*rfft3(comm,self.W[1])).real
        Uhat[0] = irfft3(comm,self.test_filter*rfft3(comm,self.W[0])).real
        #----------------------------------------------------------------------#
        # Filtered velocity components on the test grid                        #
        #   **** Note:                                                         #
        #           hat(tilde(u_i)* tilde(u_j))                                #
        #----------------------------------------------------------------------#
        Uijhat[0] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[2])).real
        Uijhat[1] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[1])).real
        Uijhat[2] = irfft3(comm,self.test_filter*rfft3(comm,self.W[2]*self.W[0])).real
        Uijhat[3] = irfft3(comm,self.test_filter*rfft3(comm,self.W[1]*self.W[1])).real
        Uijhat[4] = irfft3(comm,self.test_filter*rfft3(comm,self.W[1]*self.W[0])).real
        Uijhat[5] = irfft3(comm,self.test_filter*rfft3(comm,self.W[0]*self.W[0])).real
        #----------------------------------------------------------------------#
        # L_{ij} Dynamic Smagorinsky variable                                  #
        #----------------------------------------------------------------------#
        Lij[0] = Uijhat[0] - Uhat[2] * Uhat[2]
        Lij[1] = Uijhat[1] - Uhat[2] * Uhat[1]
        Lij[2] = Uijhat[2] - Uhat[2] * Uhat[0]
        Lij[3] = Uijhat[3] - Uhat[1] * Uhat[1]
        Lij[4] = Uijhat[4] - Uhat[1] * Uhat[0]
        Lij[5] = Uijhat[5] - Uhat[0] * Uhat[0]
        #----------------------------------------------------------------------#
        # Scales                                                               #
        #----------------------------------------------------------------------#
        deltl = self.dx[0]*2.0*np.pi/self.k_dealias             # LES scale
        deltt = self.dx[0]*2.0*np.pi/self.k_test                # test scale
        #----------------------------------------------------------------------#
        # Strain rates on LES scale                                            #
        #----------------------------------------------------------------------#
        Sij[0] = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[2]+1j*self.K[2]*W_hat[2])
        Sij[1] = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[1]+1j*self.K[1]*W_hat[2])
        Sij[2] = 0.5*irfft3(self.comm,1j*self.K[2]*W_hat[0]+1j*self.K[0]*W_hat[2])
        Sij[3] = 0.5*irfft3(self.comm,1j*self.K[1]*W_hat[1]+1j*self.K[1]*W_hat[1])
        Sij[4] = 0.5*irfft3(self.comm,1j*self.K[1]*W_hat[0]+1j*self.K[0]*W_hat[1])
        Sij[5] = 0.5*irfft3(self.comm,1j*self.K[0]*W_hat[0]+1j*self.K[0]*W_hat[0])
        self.Sij    = Sij
        #----------------------------------------------------------------------#
        # Strain rates on test filter                                          #
        #----------------------------------------------------------------------#
        Sijhat[0]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[0])).real
        Sijhat[1]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[1])).real
        Sijhat[2]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[2])).real
        Sijhat[3]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[3])).real
        Sijhat[4]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[4])).real
        Sijhat[5]   = irfft3(comm,self.test_filter*rfft3(comm,Sij[5])).real
        # Sij = strainRateFunction.strainrate(self.U,Sij,self.nx[0],1,self.dx[0],1)
        # Sijhat = strainRateFunction.strainrate(Uhat,Sijhat,self.nx[0],1,self.dx[0],1)
        #----------------------------------------------------------------------#
        # Magnitude of the strain rates on test filter                         #
        #----------------------------------------------------------------------#
        S = np.sqrt(2.0*(Sij[0]*Sij[0]+2.0*Sij[1]*Sij[1]+2.0*Sij[2]*Sij[2]+\
                            Sij[3]*Sij[3]+2.0*Sij[4]*Sij[4]+Sij[5]*Sij[5]))
        #----------------------------------------------------------------------#
        # Magnitude of the strain rates on test filter                         #
        #----------------------------------------------------------------------#
        Shat = np.sqrt(2.0*(Sijhat[0]*Sijhat[0]+2.0*Sijhat[1]*Sijhat[1]+\
                                2.0*Sijhat[2]*Sijhat[2]+Sijhat[3]*Sijhat[3]+\
                                2.0*Sijhat[4]*Sijhat[4]+Sijhat[5]*Sijhat[5]))
        #----------------------------------------------------------------------#
        # Strain rate on the test filter                                       #
        #----------------------------------------------------------------------#
        SSijT[0]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[0])).real
        SSijT[1]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[1])).real
        SSijT[2]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[2])).real
        SSijT[3]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[3])).real
        SSijT[4]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[4])).real
        SSijT[5]    = irfft3(comm,self.test_filter*rfft3(comm,S*Sij[5])).real
        #----------------------------------------------------------------------#
        # M_{ij} variables for Dynamic Smagorinsky                             #
        #----------------------------------------------------------------------#
        Mij[0]      = 2.0*(deltl**2.0)*SSijT[0] - 2.0*(deltt**2.0)*Shat*Sijhat[0]
        Mij[1]      = 2.0*(deltl**2.0)*SSijT[1] - 2.0*(deltt**2.0)*Shat*Sijhat[1]
        Mij[2]      = 2.0*(deltl**2.0)*SSijT[2] - 2.0*(deltt**2.0)*Shat*Sijhat[2]
        Mij[3]      = 2.0*(deltl**2.0)*SSijT[3] - 2.0*(deltt**2.0)*Shat*Sijhat[3]
        Mij[4]      = 2.0*(deltl**2.0)*SSijT[4] - 2.0*(deltt**2.0)*Shat*Sijhat[4]
        Mij[5]      = 2.0*(deltl**2.0)*SSijT[5] - 2.0*(deltt**2.0)*Shat*Sijhat[5]
        #----------------------------------------------------------------------#
        # Calculating the Cs (for Dynamic Smagorinsky)                         #
        #----------------------------------------------------------------------#
        LijMij      = Mij[0]*Lij[0] + 2.0*Mij[1]*Lij[1] + 2.0*Mij[2]*Lij[2] +\
                            Mij[3]*Lij[3] + 2.0*Mij[4]*Lij[4] + Mij[5]*Lij[5]
        MklMkl      = Mij[0]*Mij[0] + 2.0*Mij[1]*Mij[1] + 2.0*Mij[2]*Mij[2] +\
                            Mij[3]*Mij[3] + 2.0*Mij[4]*Mij[4] + Mij[5]*Mij[5]
        Cs2[0]      = np.divide(LijMij, MklMkl)
        self.Cs2    = Cs2
        #----------------------------------------------------------------------#
        # subgrid stress for Dynamic Smagorinsky                               #
        #----------------------------------------------------------------------#
        tau_DS[0]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[0]
        tau_DS[1]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[1]
        tau_DS[2]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[2]
        tau_DS[3]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[3]
        tau_DS[4]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[4]
        tau_DS[5]   =  -2.0 * Cs2 * deltl**2.0 * S * Sij[5]
        #----------------------------------------------------------------------#
        # BS dissipation                                                       # 
        #----------------------------------------------------------------------#
        delta       = 2.0*np.pi/self.k_dealias
        Cs          = 0.17
        tau_BS[0]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[0]
        tau_BS[1]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[1]
        tau_BS[2]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[2]
        tau_BS[3]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[3]
        tau_BS[4]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[4]
        tau_BS[5]   = -2.0 * Cs**2.0 * delta**2.0 * S * Sij[5]
        #----------------------------------------------------------------------#
        # Total subgrid stress                                                 #
        #----------------------------------------------------------------------#
        tau[0]      = C_BS * tau_BS[0] + C_DS * tau_DS[0]
        tau[1]      = C_BS * tau_BS[1] + C_DS * tau_DS[1]
        tau[2]      = C_BS * tau_BS[2] + C_DS * tau_DS[2]
        tau[3]      = C_BS * tau_BS[3] + C_DS * tau_DS[3]
        tau[4]      = C_BS * tau_BS[4] + C_DS * tau_DS[4]
        tau[5]      = C_BS * tau_BS[5] + C_DS * tau_DS[5]
        self.tau    = tau
        #----------------------------------------------------------------------#
        # Print statement for testing purposes (uncomment for testing)         #
        #----------------------------------------------------------------------#
        #print("ALES Testing:")
        #print("tau = ",tau[0,1:3,1:3,1:3])
        #sys.exit(120)
        #----------------------------------------------------------------------#
        # Fourier transform of the subgrid stress                              #
        #----------------------------------------------------------------------#
        tau_hat[0] = rfft3(comm,tau[0])
        tau_hat[1] = rfft3(comm,tau[1])
        tau_hat[2] = rfft3(comm,tau[2])
        tau_hat[3] = rfft3(comm,tau[3])
        tau_hat[4] = rfft3(comm,tau[4])
        tau_hat[5] = rfft3(comm,tau[5])
        #----------------------------------------------------------------------#
        # Adding the subgrid stress to the right hand side                     #
        #----------------------------------------------------------------------#
        m = 0
        for i in range(2, -1, -1):
            for j in range(i, -1, -1):
                self.dU[i] -= 1j*self.K[j]*tau_hat[m]
                if i != j:
                    self.dU[j] -= 1j*self.K[i]*tau_hat[m]
                m+=1
        #----------------------------------------------------------------------#
        # Getting the finished time                                            #
        #----------------------------------------------------------------------#
        t1 = time.time()

        return

    #--------------------------------------------------------------------------#
    # Subroutine for the time                                                  #
    #--------------------------------------------------------------------------#
    def get_time(self, time, **ignored):
        self.t_ds   = time

        return
    #--------------------------------------------------------------------------#
    # A term calculation                                                       #
    #--------------------------------------------------------------------------#
    def A_enstrophy_transport(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this script is to compute the A term in the enstrophy
            transport equation.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        Sij         = self.Sij
        omega       = self.omega
        #Enst        = self.enstrophy()
        #----------------------------------------------------------------------#
        # Calculating A term                                                   #
        #----------------------------------------------------------------------#
        A           =  omega[2]*Sij[0]*omega[2] + 2.0*omega[2]*Sij[1]*omega[1]\
                    + 2.0*omega[2]*Sij[2]*omega[0] + omega[1]*Sij[3]*omega[1]\
                    + 2.0*omega[1]*Sij[4]*omega[0] + omega[0]*Sij[5]*omega[0]
        #A           /= Enst
        
        return A
    #--------------------------------------------------------------------------#
    # B term calculation                                                       #
    #--------------------------------------------------------------------------#
    def B_enstrophy_transport(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the B term in the 
            enstrophy transport equation.
            
            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        nu          = self.nu
        Enst        = self.enstrophy()
        #print("Ksq shape =")
        #print(Ksq.shape)
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        enst        = 0.5*(omega[0]**2.0 + omega[1]**2.0 + omega[2]**2.0) 
        enst_hat    = rfft3(comm, enst)
        #----------------------------------------------------------------------#
        # Calculating B term                                                   #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            Ksq     = self.Ksq
            B_hat   = -nu*Ksq*enst_hat
            B       = irfft3(comm, B_hat).real
        else:
            h       = (2.0*np.pi)/64.0
            B       = np.gradient(np.gradient(Enst, h, edge_order=2)[0],\
                                    h, edge_order=2)[0]
            B       += np.gradient(np.gradient(Enst, h, edge_order=2)[1],\
                                    h, edge_order=2)[1]
            B       += np.gradient(np.gradient(Enst, h, edge_order=2)[2],\
                                    h, edge_order=2)[2]
            B       *= nu
        #print("B_hat shape =")
        #print(B_hat.shape)
        #sys.exit(150)

        return B
    #--------------------------------------------------------------------------#
    # D term calculation                                                       #
    #--------------------------------------------------------------------------#
    def D_enstrophy_transport(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the D term in the 
            enstrophy transport equation.
        
            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        nu          = self.nu
        #Enst        = self.enstrophy()
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega_hat_2 = rfft3(self.comm, omega[2])
        omega_hat_1 = rfft3(self.comm, omega[1])
        omega_hat_0 = rfft3(self.comm, omega[0])
        #----------------------------------------------------------------------#
        # Calculating D term                                                   #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            D       = np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_2))
            D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_2))
            D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_2))
            D       += np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_1))
            D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_1))
            D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_1))
            D       += np.square(irfft3(self.comm, 1j*self.K[2]*omega_hat_0))
            D       += np.square(irfft3(self.comm, 1j*self.K[1]*omega_hat_0))
            D       += np.square(irfft3(self.comm, 1j*self.K[0]*omega_hat_0))
            D       *= -nu
        else:
            h       = (2.0*np.pi)/64.0
            grad1   = np.gradient(omega[2], h, edge_order=2)
            grad2   = np.gradient(omega[1], h, edge_order=2)
            grad3   = np.gradient(omega[0], h, edge_order=2)
            D       = np.square(grad1[2])
            D       += np.square(grad1[1])
            D       += np.square(grad1[0])
            D       += np.square(grad2[2])
            D       += np.square(grad2[1])
            D       += np.square(grad2[0])
            D       += np.square(grad3[2])
            D       += np.square(grad3[1])
            D       += np.square(grad3[0])
            D       *= -nu

        return D
    #--------------------------------------------------------------------------#
    # SGS transport term calculation                                           #
    #--------------------------------------------------------------------------#
    def psi_enstrophy_transport(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the psi operator in
            the enstrophy transport equation.
            
            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Preallocating                                                        #
        #----------------------------------------------------------------------#
        Psi = np.empty((9, *self.nnx), dtype=np.float64)
        #----------------------------------------------------------------------#
        # Calculating Psi                                                      #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            tau_hat     = self.tau_hat
            Psi[0]      = irfft3(self.comm, 1j*self.K[1]*tau_hat[2]).real-\
                            irfft3(self.comm, 1j*self.K[0]*tau_hat[1]).real
            Psi[1]      = irfft3(self.comm, 1j*self.K[1]*tau_hat[4]).real-\
                            irfft3(self.comm, 1j*self.K[0]*tau_hat[3]).real
            Psi[2]      = irfft3(self.comm, 1j*self.K[1]*tau_hat[5]).real-\
                            irfft3(self.comm, 1j*self.K[0]*tau_hat[4]).real
            Psi[3]      = irfft3(self.comm, 1j*self.K[0]*tau_hat[0]).real-\
                            irfft3(self.comm, 1j*self.K[2]*tau_hat[2]).real
            Psi[4]      = irfft3(self.comm, 1j*self.K[0]*tau_hat[1]).real-\
                            irfft3(self.comm, 1j*self.K[2]*tau_hat[4]).real
            Psi[5]      = irfft3(self.comm, 1j*self.K[0]*tau_hat[2]).real-\
                            irfft3(self.comm, 1j*self.K[2]*tau_hat[5]).real
            Psi[6]      = irfft3(self.comm, 1j*self.K[2]*tau_hat[1]).real-\
                            irfft3(self.comm, 1j*self.K[1]*tau_hat[0]).real
            Psi[7]      = irfft3(self.comm, 1j*self.K[2]*tau_hat[3]).real-\
                            irfft3(self.comm, 1j*self.K[1]*tau_hat[1]).real
            Psi[8]      = irfft3(self.comm, 1j*self.K[2]*tau_hat[4]).real-\
                            irfft3(self.comm, 1j*self.K[1]*tau_hat[2]).real
        else:
            tau         = self.tau
            h           = (2.0*np.pi)/64.0
            
            Psi[0]      = np.gradient(tau[2],h, edge_order=2)[1] -\
                            np.gradient(tau[1], h, edge_order=2)[0]
            
            Psi[1]      = np.gradient(tau[4],h, edge_order=2)[1] -\
                            np.gradient(tau[3], h, edge_order=2)[0]
            
            Psi[2]      = np.gradient(tau[5],h, edge_order=2)[1] -\
                            np.gradient(tau[4], h, edge_order=2)[0]
            
            Psi[3]      = np.gradient(tau[0],h, edge_order=2)[0] -\
                            np.gradient(tau[2], h, edge_order=2)[2]
            
            Psi[4]      = np.gradient(tau[1],h, edge_order=2)[0] -\
                            np.gradient(tau[4], h, edge_order=2)[2]
            
            Psi[5]      = np.gradient(tau[2],h, edge_order=2)[0] -\
                            np.gradient(tau[5], h, edge_order=2)[2]
            
            
            Psi[6]      = np.gradient(tau[1],h, edge_order=2)[2] -\
                            np.gradient(tau[0], h, edge_order=2)[1]
            
            Psi[7]      = np.gradient(tau[3],h, edge_order=2)[2] -\
                            np.gradient(tau[1], h, edge_order=2)[1]
            
            Psi[8]      = np.gradient(tau[4],h, edge_order=2)[2] -\
                            np.gradient(tau[2], h, edge_order=2)[1]

        return Psi
    #--------------------------------------------------------------------------#
    # SGS transport term calculation                                           #
    #--------------------------------------------------------------------------#
    def Pi_enstrophy_transport(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the SGS transport
            term in the enstrophy transport equation.
            
            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        psi         = self.psi_enstrophy_transport(spec_flag)
        #----------------------------------------------------------------------#
        # Calculating omega_{i}*Psi_{il}                                       #
        #----------------------------------------------------------------------#
        term1   = omega[2]*psi[0]
        term2   = omega[2]*psi[1]
        term3   = omega[2]*psi[2]
        term4   = omega[1]*psi[3]
        term5   = omega[1]*psi[4]
        term6   = omega[1]*psi[5]
        term7   = omega[0]*psi[6]
        term8   = omega[0]*psi[7]
        term9   = omega[0]*psi[8]
        #----------------------------------------------------------------------#
        # Taking the derivatives                                               #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            Pi      = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,term1)).real
            Pi      += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,term2)).real
            Pi      += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,term3)).real
            Pi      += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,term4)).real
            Pi      += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,term5)).real
            Pi      += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,term6)).real
            Pi      += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,term7)).real
            Pi      += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,term8)).real
            Pi      += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,term9)).real
            Pi      *= -1.0
        else:
            h       = (2.0*np.pi)/64.0
            #-----------------------------------------------------------------#
            # Terms 1-3 (i = 1)                                               #
            #-----------------------------------------------------------------#
            Pi      = np.gradient(term1, h, edge_order=2)[2]
            Pi      += np.gradient(term2, h, edge_order=2)[1]
            Pi      += np.gradient(term3, h, edge_order=2)[0]
            #-----------------------------------------------------------------#
            # Terms 4-6 (i = 2)                                               #
            #-----------------------------------------------------------------#
            Pi      += np.gradient(term4, h, edge_order=2)[2]
            Pi      += np.gradient(term5, h, edge_order=2)[1]
            Pi      += np.gradient(term6, h, edge_order=2)[0]
            #-----------------------------------------------------------------#
            # Terms 7-9 (i = 3)                                               #
            #-----------------------------------------------------------------#
            Pi      += np.gradient(term7, h, edge_order=2)[2]
            Pi      += np.gradient(term8, h, edge_order=2)[1]
            Pi      += np.gradient(term9, h, edge_order=2)[0]
            Pi      *= -1.0

        return Pi
    #--------------------------------------------------------------------------#
    # SGS production term calculation                                          #
    #--------------------------------------------------------------------------#
    def P_enstrophy_transport(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the SGS production
            term in the enstrophy transport equation.

            **** Note:
                    if spec_flag is True then spectral differentiation is
                    applied.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        psi         = self.psi_enstrophy_transport(spec_flag)
        #----------------------------------------------------------------------#
        # Calculating grad(omega)                                              #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            term1   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,omega[2])).real
            term2   = irfft3(self.comm, 1j*self.K[1]*rfft3(comm,omega[2])).real
            term3   = irfft3(self.comm, 1j*self.K[0]*rfft3(comm,omega[2])).real
            term4   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,omega[1])).real
            term5   = irfft3(self.comm, 1j*self.K[1]*rfft3(comm,omega[1])).real
            term6   = irfft3(self.comm, 1j*self.K[0]*rfft3(comm,omega[1])).real
            term7   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,omega[0])).real
            term8   = irfft3(self.comm, 1j*self.K[1]*rfft3(comm,omega[0])).real
            term9   = irfft3(self.comm, 1j*self.K[0]*rfft3(comm,omega[0])).real
        else:
            h       = (2.0*np.pi)/64.0
            #------------------------------------------------------------------#
            # Calculating grad(omega)                                          #
            #------------------------------------------------------------------#
            grad_omega1 = np.gradient(omega[2], h, edge_order=2)
            grad_omega2 = np.gradient(omega[1], h, edge_order=2)
            grad_omega3 = np.gradient(omega[0], h, edge_order=2)
            #------------------------------------------------------------------#
            # Calculating grad(omega) terms                                    #
            #------------------------------------------------------------------#
            term1   = grad_omega1[2]            # d(omega_1)/dx_1
            term2   = grad_omega1[1]            # d(omega_1)/dx_2 
            term3   = grad_omega1[0]            # d(omega_1)/dx_3 
            term4   = grad_omega2[2]            # d(omega_2)/dx_1  
            term5   = grad_omega2[1]            # d(omega_2)/dx_2  
            term6   = grad_omega2[0]            # d(omega_2)/dx_3  
            term7   = grad_omega3[2]            # d(omega_3)/dx_1  
            term8   = grad_omega3[1]            # d(omega_3)/dx_2  
            term9   = grad_omega3[0]            # d(omega_3)/dx_3  
        #----------------------------------------------------------------------#
        # Calculating Production term                                          #
        #----------------------------------------------------------------------#
        prod    =  psi[0]*term1
        prod    += psi[1]*term2
        prod    += psi[2]*term3
        prod    += psi[3]*term4
        prod    += psi[4]*term5
        prod    += psi[5]*term6
        prod    += psi[6]*term7
        prod    += psi[7]*term8
        prod    += psi[8]*term9
        
        return prod
    #--------------------------------------------------------------------------#
    # Calculating the enstrophy                                                #
    #--------------------------------------------------------------------------#
    def enstrophy(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the enstrophy in
            order to compare it to the results from the extraction.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        omega       = self.omega
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        enst        = 0.5*(omega[0]**2.0 + omega[1]**2.0 + omega[2]**2.0) 

        return enst
    #--------------------------------------------------------------------------#
    # Calculating the kinetic energy                                           #
    #--------------------------------------------------------------------------#
    def kinetic_energy(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the kinetic energy 
            in order to compare it to the results from the extraction.
        
        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        U   = self.U
        #----------------------------------------------------------------------#
        # Calculating enstrophy                                                #
        #----------------------------------------------------------------------#
        ke  = 0.5*(U[0]**2.0 + U[1]**2.0 + U[2]**2.0) 

        return ke
    #--------------------------------------------------------------------------#
    # Calculating the A term in the  kinetic energy transport equation         #
    #--------------------------------------------------------------------------#
    def A_KE(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the A term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        press   = self.compute_pressure()[2]
        #print(press.shape)
        #print(press[1,23,23])
        #sys.exit(187)
        vel     = self.U
        Ke      = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating A                                                        #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            A   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,press*vel[2])).real
            A   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,press*vel[1])).real
            A   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,press*vel[0])).real
            A   *= -1.0
        else:
            h   = (2.0*np.pi)/64.0
            A   = np.gradient(press*vel[2], h, edge_order=2)[0]
            A   = np.gradient(press*vel[1], h, edge_order=2)[1]
            A   = np.gradient(press*vel[0], h, edge_order=2)[2]
            A   *= -1.0

        return A
    #--------------------------------------------------------------------------#
    # Calculating the B term in the  kinetic energy transport equation         #
    #--------------------------------------------------------------------------#
    def C_KE(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the B term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        tau = self.tau
        vel = self.U
        Ke  = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating B                                                        #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            C   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,tau[0]*vel[2])).real
            C   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,tau[1]*vel[2])).real
            C   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,tau[2]*vel[2])).real
            C   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,tau[1]*vel[1])).real
            C   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,tau[3]*vel[1])).real
            C   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,tau[4]*vel[1])).real
            C   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,tau[2]*vel[0])).real
            C   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,tau[4]*vel[0])).real
            C   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,tau[5]*vel[0])).real
            C   *= -1.0
        else:
            h   = (2.0*np.pi)/64.0
            C   = np.gradient(tau[0]*vel[2], h, edge_order=2)[0]
            C   += np.gradient(tau[1]*vel[2], h, edge_order=2)[1]
            C   += np.gradient(tau[2]*vel[2], h, edge_order=2)[2]
            C   += np.gradient(tau[1]*vel[1], h, edge_order=2)[0]
            C   += np.gradient(tau[3]*vel[1], h, edge_order=2)[1]
            C   += np.gradient(tau[4]*vel[1], h, edge_order=2)[2]
            C   += np.gradient(tau[2]*vel[0], h, edge_order=2)[0]
            C   += np.gradient(tau[4]*vel[0], h, edge_order=2)[1]
            C   += np.gradient(tau[5]*vel[0], h, edge_order=2)[2]
            C   *= -1.0

        return C
    #--------------------------------------------------------------------------#
    # Calculating the C term in the kinetic energy transport equation          #
    #--------------------------------------------------------------------------#
    def B_KE(self, spec_flag, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the C term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        nu  = self.nu
        Sij = self.Sij
        vel = self.U
        Ke  = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating B                                                        #
        #----------------------------------------------------------------------#
        if spec_flag is True:
            B   = irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[0]*vel[2])).real
            B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[1]*vel[2])).real
            B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[2]*vel[2])).real
            B   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[1]*vel[1])).real
            B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[3]*vel[1])).real
            B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[4]*vel[1])).real
            B   += irfft3(self.comm, 1j*self.K[2]*rfft3(comm,Sij[2]*vel[0])).real
            B   += irfft3(self.comm, 1j*self.K[1]*rfft3(comm,Sij[4]*vel[0])).real
            B   += irfft3(self.comm, 1j*self.K[0]*rfft3(comm,Sij[5]*vel[0])).real
            B   *= 2.0*nu
        else:
            h   = (2.0*np.pi)/64.0
            B   = np.gradient(Sij[0]*vel[2], h, edge_order=2)[0]
            B   += np.gradient(Sij[1]*vel[2], h, edge_order=2)[1]
            B   += np.gradient(Sij[2]*vel[2], h, edge_order=2)[2]
            B   += np.gradient(Sij[1]*vel[1], h, edge_order=2)[0]
            B   += np.gradient(Sij[3]*vel[1], h, edge_order=2)[1]
            B   += np.gradient(Sij[4]*vel[1], h, edge_order=2)[2]
            B   += np.gradient(Sij[2]*vel[0], h, edge_order=2)[0]
            B   += np.gradient(Sij[4]*vel[0], h, edge_order=2)[1]
            B   += np.gradient(Sij[5]*vel[0], h, edge_order=2)[2]
            B   *= 2.0*nu

        return B
    #--------------------------------------------------------------------------#
    # Calculating the D term in the kinetic energy transport equation          #
    #--------------------------------------------------------------------------#
    def D_KE(self, **ignored):
        """=====================================================================
        Purpose:
            The purpose of this subroutine is to calculate the D term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ====================================================================="""
        #----------------------------------------------------------------------#
        # Calling global variables                                             #
        #----------------------------------------------------------------------#
        nu  = self.nu
        Sij = self.Sij
        Ke  = self.kinetic_energy()
        #----------------------------------------------------------------------#
        # Calculating D                                                        #
        #----------------------------------------------------------------------#
        D   = Sij[0]*Sij[0]
        D   += 2.0*Sij[1]*Sij[1]
        D   += 2.0*Sij[2]*Sij[2]
        D   += Sij[3]*Sij[3]
        D   += 2.0*Sij[4]*Sij[4]
        D   += Sij[5]*Sij[5]
        D   *= -2.0*nu
        #D   /= Ke
        
        return D
    #---------------------------------------------------------------------#
    # Calculating the P term in the kinetic energy transport equation     #
    #---------------------------------------------------------------------#
    def P_KE(self, **ignored):
        """================================================================
        Purpose:
            The purpose of this subroutine is to calculate the P term in  the
            kinetic energy transport equation.

        Author:
            Emilio Torres
        ================================================================"""
        #-----------------------------------------------------------------#
        # Calling global variables                                        #
        #-----------------------------------------------------------------#
        nu  = self.nu
        tau = self.tau
        Sij = self.Sij
        Ke  = self.kinetic_energy()
        #-----------------------------------------------------------------#
        # Calculating D                                                   #
        #-----------------------------------------------------------------#
        P   = tau[0]*Sij[0]
        P   += 2.0*tau[1]*Sij[1]
        P   += 2.0*tau[2]*Sij[2]
        P   += tau[3]*Sij[3]
        P   += 2.0*tau[4]*Sij[4]
        P   += tau[5]*Sij[5]
        #P   /= Ke
        
        return P
    #---------------------------------------------------------------------#
    # Calculating the sum of P + Pi for the enstrophy transport           #
    #---------------------------------------------------------------------#
    def sgs_enstrophy(self, **ignored):

        """ Calculating the sum P and Pi term in the enstrophy transport
            term """
        #-----------------------------------------------------------------#
        # calling global variables                                        #
        #-----------------------------------------------------------------#
        tau     = self.tau
        omega   = self.omega
        h       = 2.0*np.pi/64.0
        #-----------------------------------------------------------------#
        # i = 1, j = 2, k = 3                                             #
        #-----------------------------------------------------------------#
        out  =  omega[2]*np.gradient(\
                    np.gradient(tau[2], h, edge_order=2)[2] +\
                    np.gradient(tau[4], h, edge_order=2)[1] +\
                    np.gradient(tau[5], h, edge_order=2)[0],
                    h, edge_order=2)[1]
        #-----------------------------------------------------------------#
        # i = 1, j = 2, k = 3                                             #
        #-----------------------------------------------------------------#
        out  -= omega[2]*np.gradient(\
                    np.gradient(tau[1], h, edge_order=2)[2] +\
                    np.gradient(tau[3], h, edge_order=2)[1] +\
                    np.gradient(tau[4], h, edge_order=2)[0],
                    h, edge_order=2)[0]
        #-----------------------------------------------------------------#
        # i = 2, j = 3, k = 1                                             #
        #-----------------------------------------------------------------#
        out  += omega[1]*np.gradient(\
                    np.gradient(tau[0], h, edge_order=2)[2] +\
                    np.gradient(tau[1], h, edge_order=2)[1] +\
                    np.gradient(tau[2], h, edge_order=2)[0],
                    h, edge_order=2)[0]
        #-----------------------------------------------------------------#
        # i = 2, j = 1, k = 3                                             #
        #-----------------------------------------------------------------#
        out  -= omega[1]*np.gradient(\
                    np.gradient(tau[2], h, edge_order=2)[2] +\
                    np.gradient(tau[4], h, edge_order=2)[1] +\
                    np.gradient(tau[5], h, edge_order=2)[0],
                    h, edge_order=2)[2]
        #-----------------------------------------------------------------#
        # i = 3, j = 1, k = 2                                             #
        #-----------------------------------------------------------------#
        out  += omega[0]*np.gradient(\
                    np.gradient(tau[1], h, edge_order=2)[2] +\
                    np.gradient(tau[3], h, edge_order=2)[1] +\
                    np.gradient(tau[4], h, edge_order=2)[0],
                    h, edge_order=2)[2]
        #-----------------------------------------------------------------#
        # i = 3, j = 2, k = 1                                             #
        #-----------------------------------------------------------------#
        out  -= omega[0]*np.gradient(\
                    np.gradient(tau[0], h, edge_order=2)[2] +\
                    np.gradient(tau[1], h, edge_order=2)[1] +\
                    np.gradient(tau[2], h, edge_order=2)[0],
                    h, edge_order=2)[1]
        
        return out
##############################################################################
# Define the problem ("main" function)
###############################################################################
def ales244_static_les_test(pp=None, sp=None):
    """
    Arguments:
    ----------
    pp: (optional) program parameters, parsed by argument parser
        provided by this file
    sp: (optional) solver parameters, parsed by spectralLES.parser
    """

    if comm.rank == 0:
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation of problem \n"
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))
        print("----------------------------------------------------------")
    # ------------------------------------------------------------------
    # Get the problem and solver parameters and assert compliance
    if pp is None:
        pp = hit_parser.parse_known_args()[0]

    if sp is None:
        sp = spectralLES.parser.parse_known_args()[0]

    if comm.rank == 0:
        print('\nProblem Parameters:\n-------------------')
        for k, v in vars(pp).items():
            print(k, v)
        print('\nSpectralLES Parameters:\n-----------------------')
        for k, v in vars(sp).items():
            print(k, v)
        print("\n----------------------------------------------------------\n")

    assert len(set(pp.N)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal mesh dimensions')
    N = pp.N[0]
    assert len(set(pp.L)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal domain dimensions')
    L = pp.L[0]

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Error: job started with improper number of MPI tasks for '
                  'the size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Configure the LES solver
    solver = ales244_solver(comm, **vars(sp))

    solver.computeAD = solver.computeAD_vorticity_form
    Sources = [solver.computeSource_linear_forcing,
               solver.computeSource_ales244_SGS]

    H_244 = np.loadtxt('h_ij.dat', usecols=(1, 2, 3, 4, 5, 6), unpack=True)
    kwargs = {'H_244': H_244, 'dvScale': None}

    U_hat       = solver.U_hat
    U           = solver.U
    omega       = solver.omega
    tau         = solver.tau
    Sij         = solver.Sij
    cs          = solver.Cs2
    Pres        = solver.Pres
    Kmod        = np.floor(np.sqrt(solver.Ksq)).astype(int)
    # ------------------------------------------------------------------
    # form HIT initial conditions from either user-defined values or
    # physics-based relationships
    Urms = 1.083*(pp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit= getattr(pp, 'Einit', None) or Urms**2   # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak= getattr(pp, 'kpeak', None) or N//4      # ~ kmax/2

    # currently using a fixed random seed for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    # ------------------------------------------------------------------
    # Configure a spatial field writer
    writer = mpiWriter(comm, odir=pp.odir, N=N)
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters
    tauK = sqrt(pp.nu/pp.epsilon)           # Kolmogorov time-scale
    taul = 0.11*sqrt(3)*L/Urms              # 0.11 is empirical coefficient

    if pp.tlimit == np.Inf:
        pp.tlimit = 200*taul

    dt_rst = getattr(pp, 'dt_rst', None) or taul
    dt_spec= getattr(pp, 'dt_spec', None) or 0.2*taul
    dt_drv = getattr(pp, 'dt_drv', None) or 0.25*tauK

    t_sim = t_rst = t_spec = t_drv = 0.0
    tstep = irst = ispec = 0
    tseries = []

    if comm.rank == 0:
        print('\ntau_ell = %.6e\ntau_K = %.6e\n' % (taul, tauK))

    # -------------------------------------------------------------------------
    # Run the simulation
    #while t_sim < pp.tlimit:
    path_name = 'data-0'
    while t_sim < pp.tlimit+1.e-10 and tstep < 10:
        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_constant_nu(pp.cfl)
        t_test = t_sim + 0.5*dt
        # -- output/store a log every step if needed/wanted
        KE      = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
        Omega   = 0.5*comm.allreduce(psum(np.square(omega)))/solver.Nx
        tseries.append([tstep, t_sim, KE, Omega])
        #-----------------------------------------------------------------#
        # Calculating enstrophy transport terms spectral tool             #
        #-----------------------------------------------------------------#
        A_enst          = solver.A_enstrophy_transport()
        B_enst          = solver.B_enstrophy_transport(False)
        D_enst          = solver.D_enstrophy_transport(False)
        Pi_enst         = solver.Pi_enstrophy_transport(False)
        P_enst          = solver.P_enstrophy_transport(True)
        enst            = solver.enstrophy()
        #-----------------------------------------------------------------#
        # Calculating kinetic energy transport terms                      #
        #-----------------------------------------------------------------#
        A_ke    = solver.A_KE(False)
        B_ke    = solver.B_KE(False)
        C_ke    = solver.C_KE(False)
        D_ke    = solver.D_KE()
        P_ke    = solver.P_KE()
        ke      = solver.kinetic_energy()
        # -- output KE and enstrophy spectra
        if t_test >= 1.0e-16*t_spec:
            #-------------------------------------------------------------#
            # Storing the time data                                       #
            #-------------------------------------------------------------#
            #np.save(path_name + '/time/SimulationTime_%(a)3.3d' % {'a': ispec}, t_sim)
            ##------------------------------------------------------------------#
            ## Storing the enstrophy transport terms (spectral)                 #
            ##------------------------------------------------------------------#
            #np.save(path_name + '/A-enst/A_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   A_enst)
            #np.save(path_name + '/B-enst/B_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   B_enst)
            #np.save(path_name + '/D-enst/D_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   D_enst)
            #np.save(path_name + '/Pi-enst/Pi_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Pi_enst)
            #np.save(path_name + '/P-enst/P_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank},   P_enst)
            ##------------------------------------------------------------------#
            ## Storing the kinetic energy transport terms                       #
            ##------------------------------------------------------------------#
            #np.save(path_name + '/A-ke/A_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, A_ke)
            #np.save(path_name + '/B-ke/B_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, B_ke)
            #np.save(path_name + '/C-ke/C_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, C_ke)
            #np.save(path_name + '/D-ke/D_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, D_ke)
            #np.save(path_name + '/P-ke/P_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, P_ke)
            ##------------------------------------------------------------------#
            ## Storing the velocity data                                        #
            ##------------------------------------------------------------------#
            #np.save(path_name + '/velocity1/Velocity1_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[2])
            #np.save(path_name + '/velocity2/Velocity2_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[1])
            #np.save(path_name + '/velocity3/Velocity3_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, U[0])
            #------------------------------------------------------------------#
            # Storing the pressure data                                        #
            #------------------------------------------------------------------#
            #np.save('data-5/pressure/Pressure1_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Pres[0])
            #------------------------------------------------------------------#
            # Storing the kinetic energy and enstrophy                         #
            #------------------------------------------------------------------#
            #np.save(path_name + '/ke/ke_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, ke)
            #np.save(path_name + '/enst/enstrophy_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, enst)
            ##------------------------------------------------------------------#
            ## Storing the subgrid stress data                                  #
            ##------------------------------------------------------------------#
            #np.save(path_name + '/tau/tau11_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[0])
            #np.save(path_name + '/tau/tau12_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[1])
            #np.save(path_name + '/tau/tau13_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[2])
            #np.save(path_name + '/tau/tau22_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[3])
            #np.save(path_name + '/tau/tau23_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[4])
            #np.save(path_name + '/tau/tau33_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, tau[5])
            ##------------------------------------------------------------------#
            ## Storing the strain rates                                         #
            ##------------------------------------------------------------------#
            #np.save(path_name + '/strain-rates/S11_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[0])
            #np.save(path_name + '/strain-rates/S12_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[1])
            #np.save(path_name + '/strain-rates/S13_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[2])
            #np.save(path_name + '/strain-rates/S22_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[3])
            #np.save(path_name + '/strain-rates/S23_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[4])
            #np.save(path_name + '/strain-rates/S33_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, Sij[5])
            #------------------------------------------------------------------#
            # Storing the vorticity data                                       #
            #------------------------------------------------------------------#
            #np.save(path_name + '/omega1/Omega1_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[2])
            #np.save(path_name + '/omega2/Omega2_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[1])
            #np.save(path_name + '/omega3/Omega3_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, omega[0])
            #------------------------------------------------------------------#
            # Storing Cs2 data                                                 #
            #------------------------------------------------------------------#
            #np.save(path_name + '/Cs2/Cs2_%(a)3.3d_%(b)3.3d' % {'a': ispec, 'b': comm.rank}, cs)

            ## # -- output message log to screen on spectrum output only
            ## if comm.rank == 0:
            #     print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
            #           % (tstep, t_sim, dt, KE))
            #
            # # -- output kinetic energy spectrum to file
            # spect3d = np.sum(np.real(U_hat*np.conj(U_hat)), axis=0)
            # spect3d[..., 0] *= 0.5
            # spect1d = shell_average(comm, spect3d, Kmod)
            #
            # if comm.rank == 0:
            #     fname = '%s/%s-%3.3d_KE.spectra' % (pp.adir, pp.pid, ispec)
            #     fh = open(fname, 'w')
            #     metadata = Ek_fmt('u_i')
            #     fh.write('%s\n' % metadata)
            #     spect1d.tofile(fh, sep='\n', format='% .8e')
            #     fh.close()

            t_spec += dt_spec
            ispec += 1

        # -- output physical-space solution fields for restarting and analysis
        if t_test >= t_rst:
            # writer.write_scalar('%s-Velocity1_%3.3d.rst' %
            #                     (pp.pid, irst), U[0], np.float64)
            # writer.write_scalar('%s-Velocity2_%3.3d.rst' %
            #                     (pp.pid, irst), U[1], np.float64)
            # writer.write_scalar('%s-Velocity3_%3.3d.rst' %
            #                     (pp.pid, irst), U[2], np.float64)

            t_rst += dt_rst
            irst += 1

        # -- Update the forcing mean scaling
        if t_test >= t_drv:
            # call solver.computeSource_linear_forcing to compute dvScale only
            kwargs['dvScale'] = Sources[0](computeRHS=False)
            t_drv += dt_drv
        # -- integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)
        #----------------------------------------------------------------------#
        # testing center for printing and defining variables                   #
        #----------------------------------------------------------------------#
        # Calculate Pressure term
        #solver.compute_pressure()
        #print(A.shape)
        #print('A = ',A[0,2,23:27,23:27])
        #sys.exit(100)
        #t3 = time.time()

        #print('Step Time: ',t3-t2)

        if comm.rank == 0:
            print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
                  % (tstep, t_sim, dt, KE))

        t_sim       += dt
        solver.get_time(t_sim)
        tstep       += 1
        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation

    KE      = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
    Omega   = 0.5*comm.allreduce(psum(np.square(omega)))/solver.Nx
    tseries.append([tstep, t_sim, KE, Omega])

    if comm.rank == 0:
        #fname = '%s/%s-%3.3d_KE_tseries.txt' % (pp.adir, pp.pid, ispec)
        fname   = path_name + '/ke-omega-avg.txt' 
        header  = 'Kinetic Energy Timeseries,\n# columns: tstep, time, KE, Omega'
        np.savetxt(fname, tseries, fmt='%10.5e', header=header)

        print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
              % (tstep, t_sim, dt, KE))
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation finished at {}."
              .format(timeofday()))
        print("----------------------------------------------------------")

    # # -- output kinetic energy spectrum to file
    # spect3d = np.sum(np.real(U_hat*np.conj(U_hat)), axis=0)
    # spect3d[..., 0] *= 0.5
    # spect1d = shell_average(comm, spect3d, Kmod)

    # if comm.rank == 0:
    #     fh = open('%s/%s-%3.3d_KE.spectra' %
    #               (pp.adir, pp.pid, ispec), 'w')
    #     metadata = Ek_fmt('u_i')
    #     fh.write('%s\n' % metadata)
    #     spect1d.tofile(fh, sep='\n', format='% .8e')
    #     fh.close()

    # # -- output physical-space solution fields for restarting and analysis
    # writer.write_scalar('%s-Velocity1_%3.3d.rst' %
    #                     (pp.pid, irst), U[0], np.float64)
    # writer.write_scalar('%s-Velocity2_%3.3d.rst' %
    #                     (pp.pid, irst), U[1], np.float64)
    # writer.write_scalar('%s-Velocity3_%3.3d.rst' %
    #                     (pp.pid, irst), U[2], np.float64)


    return


###############################################################################
# Add a parser for this problem
###############################################################################
hit_parser = argparse.ArgumentParser(prog='Homogeneous Isotropic Turbulence',
                                     parents=[spectralLES.parser])

hit_parser.description = ("A large eddy simulation model testing and analysis "
                          "script for homogeneous isotropic turbulence")
hit_parser.epilog = ('This program uses spectralLES, %s'
                     % spectralLES.parser.description)

config_group = hit_parser._action_groups[2]

config_group.add_argument('-p', '--pid', type=str, default='test',
                          help='problem prefix for analysis outputs')
config_group.add_argument('--dt_drv', type=float,
                          help='refresh-rate of forcing pattern')

time_group = hit_parser.add_argument_group('time integration arguments')

time_group.add_argument('--cfl', type=float, default=0.45, help='CFL number')
time_group.add_argument('-t', '--tlimit', type=float, default=np.inf,
                        help='solution time limit')
time_group.add_argument('-w', '--twall', type=float,
                        help='run wall-time limit (ignored for now!!!)')

init_group = hit_parser.add_argument_group('initial condition arguments')

init_group.add_argument('-i', '--init', '--initial-condition',
                        metavar='IC', default='GamieOstriker',
                        choices=['GamieOstriker', 'TaylorGreen'],
                        help='use specified initial condition')
init_group.add_argument('--kexp', type=float,
                        help=('Gamie-Ostriker power-law scaling of '
                              'initial velocity condition'))
init_group.add_argument('--kpeak', type=float,
                        help=('Gamie-Ostriker exponential-decay scaling of '
                              'initial velocity condition'))
init_group.add_argument('--Einit', type=float,
                        help='specify KE of initial velocity field')

rst_group = hit_parser.add_argument_group('simulation restart arguments')

rst_group.add_argument('-l', '--last', '--restart-from-last', dest='restart',
                       action='store_const', const=-1,
                       help='restart from last *.rst checkpoint in IDIR')
rst_group.add_argument('-r', '--rst', '--restart-from-num', type=int,
                       dest='restart', metavar='NUM',
                       help=('restart from specified checkpoint in IDIR, '
                             'negative numbers index backwards from last'))
rst_group.add_argument('--idir', type=str, default='./data/',
                       help='input directory for restarts')

io_group = hit_parser.add_argument_group('simulation output arguments')

io_group.add_argument('--odir', type=str, default='./data/',
                      help='output directory for simulation fields')
io_group.add_argument('--dt_rst', type=float,
                      help='time between restart checkpoints')
io_group.add_argument('--dt_bin', type=float,
                      help='time between single-precision outputs')

anlzr_group = hit_parser.add_argument_group('analysis output arguments')

anlzr_group.add_argument('--adir', type=str, default='./analysis/',
                         help='output directory for analysis products')
anlzr_group.add_argument('--dt_stat', type=float,
                         help='time between statistical analysis outputs')
anlzr_group.add_argument('--dt_spec', type=float,
                         help='time between isotropic power spectral density'
                              ' outputs')


###############################################################################
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    ales244_static_les_test()
