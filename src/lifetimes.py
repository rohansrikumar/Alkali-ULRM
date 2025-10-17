"""                     Alkali-ULRM: lifetimes Module

This module contains functions to compute the lifetimes, and electronic transitions of 
alkali atoms, as well as the Rydberg Molecules formed by them.
"""


import numpy as np
from sympy.physics.quantum.cg import CG

from scipy.constants import physical_constants, epsilon_0, hbar
from scipy.constants import c as C_c #speed of light
from scipy.constants import h as C_h #Planck's constant
from scipy.constants import e as C_e #electron charge

def atomic_lifetime(atom,n,l,j,mj):
    """
    Calculates the lifetime of an alkali atom in a specific Rydberg state |n,l,j,mj> in an unpolarized light field.
    Parameters:
        atom : Atom class instance (from arc package)
        n : Principal quantum number
        l : Orbital angular momentum quantum number
        j : Total angular momentum quantum number
        mj : Magnetic quantum number

    Returns:
        Lifetime: seconds
    
    """

    n1,l1,j1,mj1 = n,l,j,mj
    nmin = max(atom.groundStateN  ,l1)
     
    #default alkali atom ground-state |n,l=0,s=1/2,j=1/2>
    #GHz energy to Hz
    freq_eq = atom.getEnergy(n1,l1,j1)*C_e/C_h

    Transition_rate = 0
    
    #list of states with possible spontaneous emissions
    #via dipole coupling
    final_states = []

    #all the states ranging from the n >= groundstateN to our initial n-state \
    # that follow dipole selection rules
    #n+4 is to account for quantum defect splitting of alkali atoms
    #that might allow for another emission channel (for example 18d -> 19s)
    
    final_states = [ (n2, l2, j2) \
                    for n2 in range(nmin, n1 + 4)\
                    for l2 in range(0, n2)\
                    for j2 in np.arange(np.abs(l2 - 0.5), l2 + 1.5, 1)\
                    if np.abs(l2 - l1) == 1 and np.abs(j2 - j1) <= 1 and not (j2 == j1 == 0)]
    
    #We use a single nested list complehension, with conditionals 
    #instead of nested for loops 


    #States with n < groundstateN that are energetically above grouonstateN
    #for example Rubidium has 4d and 4f valence states, above 5s groundstate
    for state in atom.extraLevels:
        if abs(j1 - state[2]) <= 1 and abs(state[1] - l1) == 1:
                final_states.append(state)

        
    for state in final_states:
        n2,l2,j2 = state

        #Transition frequency
        Omega = 2*np.pi * (freq_eq - atom.getEnergy(n2,l2,j2)*C_e/C_h)
        #imported from arc, C_e: e, C_h: h i.e. electron charge and plancks constant

        if Omega <=0:
                #Only transitions to lower energy states
                continue

        for mj2 in np.arange(-j2,j2+1,1):

            if np.abs(mj2-mj1)>1:
                 continue
            
            Dipole_full = atom.getDipoleMatrixElement(n1,l1,j1,mj1,n2,l2,j2,mj2,mj2-mj1) \
                * (physical_constants["Bohr radius"][0]) * C_e
            
            #Sum over all transition rates |n1,l1,j1,mj1> --> |n2,l2,j2,mj2>
            #for all relevant values of |n2,l2,j2,mj2>
            Transition_rate += Dipole_full**2 *Omega**3/ (3 * np.pi * epsilon_0 * hbar * C_c**3) 

    
            
            
    return (1/Transition_rate)



def atomic_lifetime_averaged(atom,n,l,j):
    """
    Calculates the lifetime of an alkali atom in a specific Rydberg state |n,l,j> by averaging over the magnetic
    quantum numbers, utlizing wigner eckart theorem.
    Parameters:
        atom : Atom class instance (from arc package)
        n : Principal quantum number
        l : Orbital angular momentum quantum number
        j : Total angular momentum quantum number
        mj : Magnetic quantum number

    Returns:
        Lifetime: seconds
    
    """

    n1,l1,j1 = n,l,j
    nmin = max(atom.groundStateN  ,l1)
     
    #default alkali atom ground-state |n,l=0,s=1/2,j=1/2>

    #GHz energy to Hz
    freq_eq = atom.getEnergy(n1,l1,j1)*C_e/C_h

    Transition_rate = 0
    
    #list of states with possible spontaneous emissions
    #via dipole coupling
    final_states = []

    for n2 in range(nmin,n1+1):
        for l2 in range(0,n2):
            jj = np.arange(np.abs(l2-0.5),l2+1.5,1)
            for j2 in jj:

                #Pre-screening dipole transistion rules
                if  np.abs(l2 - l1) !=1 or np.abs(j2 - j1) >1:
                    continue;
                
                if j2==j1 and j1==0:
                    continue;

                final_states.append((n2,l2,j2))

    #States with n < groundstateN that are energetically above grouonstateN
    #for example Rubidium has 4d and 4f valence states, above 5s groundstate
    for state in atom.extraLevels:
        if abs(j1 - state[2]) <= 1 and abs(state[1] - l1) == 1:
                final_states.append(state)

        
    for state in final_states:
        n2,l2,j2 = state

        #Transition frequency
        Omega = 2*np.pi * (freq_eq - atom.getEnergy(n2,l2,j2)*C_e/C_h)
        #imported from arc, C_e: e, C_h: h i.e. electron charge and plancks constant

        degeneracy = (2.0*j2 + 1.0)/(2.0*j1 + 1.0)
                #to average over different mj correctly

        if Omega <= 0:
            #print("Error: not emission")
            degeneracy=0
            #To not count higher energy states

        Dipole_reduced = atom.getReducedMatrixElementJ_asymmetric(n1,l1,j1,n2,l2,j2,s=0.5) \
                    * (physical_constants["Bohr radius"][0]) * C_e
                
                
        #Sum over all transition rates |n1,l1,j1> --> |n2,l2,j2>
        #for all relevant values of |n2,l2,j2>
        Transition_rate +=  Dipole_reduced**2 *Omega**3/ (3 * np.pi * epsilon_0 * hbar * C_c**3)*degeneracy 


            
    return (1/Transition_rate)

















#Calculating the photo excitation strength matrix 
#between all possible Rydberg states in the |alpha> basis 
# and a given final state (Ryd class instance)
def photo_excitation(alpha,states,atom,initial_state):
    
    dipole_list = dipole_excitation_basis(alpha,atom,initial_state) 
    total_rate = np.abs(np.einsum('ar,a->r',states,dipole_list))

    return total_rate


def dipole_emission_basis(alpha,atom,final_state):
    """
    Computes the dipole transition matrix elements between all 
    possible Rydberg states in the |alpha> basis and a given final state. 
    Useful for calculating decay rates and lifetimes of Rydberg atoms and molecules.

    Parameters:
        alpha : list of Rydberg states (Ryd class instances)
        atom : Atom class instance (from arc package)
        final_state : dictionary with keys 'n', 'l', 'j', 'mj' representing the final state

    Returns:
        list_dipole : numpy array of dipole matrix elements
    """

    list_dipole = np.zeros([len(alpha)])
    b=final_state
    for a in alpha:
        list_dipole = atom.getDipoleMatrixElement(a.n,a.l,a.j,a.mj,b['n'],b['l'],b['j'],b['mj'],b['mj'] - a.mj)

    return list_dipole


def dipole_excitation_basis(alpha,atom,initial_state):
    """
    Computes the dipole transition matrix elements between a given initial state 
    (including hyperfine structure of perturber) and all 
    possible Rydberg states in the |alpha> basis. 
    Useful for calculating photo excitation strengths. 

    Parameters:
        alpha : list of Rydberg states (Ryd class instances)
        atom : Atom class instance (from arc package)
        final_state : dictionary with keys 'n', 'l', 'j', 'mj', 'F', 'MF' representing the initial state

    Returns:
        list_dipole : numpy array of dipole matrix elements
    """

    list_dipole = np.zeros([len(alpha)])
    b=initial_state
    for i,a in enumerate(alpha):
        list_dipole[i] = atom.getDipoleMatrixElement(b['n'],b['l'],b['j'],b['mj'],a.n,a.l,a.j,a.mj, a.mj - b['mj'])

    return list_dipole * float(CG(a.I,a.mi,a.s2,a.ms2,b['F'], b['MF']).doit())

def molecule_lifetime(alpha,atom,state_coefs_eq,energy_eq,n0):

    n2 = atom.groundStateN  
    #default alkali atom ground-state |n,l=0,s=1/2,j=1/2>

    #GHz energy to Hz
    freq_eq = energy_eq*10**9

    Transition_rate = 1e-30

    for n in range(n2,n0):
        for l in range(0,n-1):
            jj = np.arange(np.abs(l-0.5),l+1.5,1)
            for j in jj:
                mmj = np.arange(-j,j+1,1)
                for mj in mmj:
                    final_state = {'n':n,'l':l,'j':j,'mj':mj}
                    dip_list_alpha  = dipole_emission_basis(alpha,atom,final_state)

                    Omega = 2*np.pi * (freq_eq - atom.getEnergy(n,l,j)*C_e/C_h)
                    #imported from arc, C_e: e, C_h: h i.e. electron charge and plancks constant

                    Dipole_molecule = np.abs(np.sum(dip_list_alpha*state_coefs_eq)) \
                        * (physical_constants["Bohr radius"][0]) * C_e
                    
                    degeneracy = 1#(2*j + 1)/(2*j1 + 1)
                    #
                    
                    Transition_rate = Transition_rate + Dipole_molecule**2 *Omega**3/ (3 * np.pi * epsilon_0 * hbar * C_c**3)*degeneracy

    return (1/Transition_rate)

