"""                     Alkali-ULRM: Spin Operators Module

This module contains functions to compute projectors and operators for relevant
observables and interactions in the context of ultralong-range Rydberg molecules (ULRM).

The observables/projectors are based on the Hyperfine operator F=s2+i, the Rydberg electronic state |n,l,j>,
total electronic spin operator S = s1+s2, and a quasi symmetry operator N = s1 + s2 + i 

The operators are defined in the |alpha> basis, where |alpha> = |n,l,j,s1,mj; i,mi,s2,ms2> (see Electronic State Module)

"""

import numpy as np  
from sympy.physics.quantum.cg import CG
from itertools import product



def F_projector(F,alpha):
    
    """
    Constructs the F projector matrix, for a given F, in the |alpha> basis.

    Parameters:
        F (int/float): Hyperfine quantum number.
        alpha (list of RydbergBasis objects): Basis states.

    Returns:    
        np.ndarray: F projector matrix.

    Description:
        projector for a given F value, explicitly expanded in the |alpha> basis (hyperfine uncoupled)
        where F = I+s2, the hyperfine operator for the ground state alkali atom
       
        
    """
    MF = np.arange(-F,F+1,1)
    P_F = np.zeros([len(alpha),len(alpha)])
    
    
    for i,a in enumerate(alpha):        
        for j,b in enumerate(alpha):

            #F-projector is diagonal in the Rydberg atom subspace
            if a.Ryd_quantum_numbers() != b.Ryd_quantum_numbers():
                continue;
            
            #P_F = Sum |F,MF><F,MF|
            #|F,MF><F,MF| in alpha basis =  <alpha: i,mi,s2,ms2|F,MF><F,MF|beta: i,mi,s2,ms2> \
            #                                   * dirac_delta(alpha_{n,l,j,mj},beta_{n,l,j,mj})
            for mf in MF:
                P_F[i,j] +=  float(CG(a.s2, a.ms2, a.I, a.mi,F, mf).doit()) \
                            * float(CG(b.s2, b.ms2, b.I, b.mi,F, mf).doit())
    
    return P_F

def S_projector(S,alpha):
    """
    Constructs the S projector matrix, for a given S, in the |alpha> basis.
    Parameters:
        S (int/float): Total spin quantum number.
        alpha (list of RydbergBasis objects): Basis states.
    Returns:
        np.ndarray: S projector matrix.
    Description:
        projector for a given S value, explicitly expanded in the |alpha> basis (spin-orbit coupled, spin-spin uncoupled)
        where S = s1+s2, the total Spin operator for the scattering system
        we first uncouple the total spin basis |S,MS> --> |s1,ms1,s2,ms2>;  MS = ms1 + ms2
        then recouple the spin-orbit basis |l,s1,ml,ms1> --> |l,s1,j,mj>; ml = mj-ms1 (alpha basis)
    """
    MSlist = np.arange(-S,S+1,1)
    P_S = np.zeros([len(alpha),len(alpha)])

    
    #possible values of ms1
    ms1 = [-0.5,0.5]

    for i,a in enumerate(alpha):
        for j,b in enumerate(alpha):
            
            #S^2 operator is diagonal in the (n,l ; I,mi) subspace
            if (a.n,a.I,a.mi,a.l) != (b.n,b.I,b.mi,b.l):
                continue;
            
            # Use itertools.product instead of nested loops
            for ms1_a, ms1_b in product(ms1, ms1):
                ml_a = a.mj - ms1_a
                ml_b = b.mj - ms1_b
                
                # MS has to be ms2 + ms1, and a.MS == b.MS,
                # otherwise CG coefficient is zero
                if b.ms2 + ms1_b != a.ms2 + ms1_a:
                    continue

                MS = ms1_a + a.ms2
                

                S_coup=0
                for MS in MSlist:
                    #for MS in range(-S,S+1):
                    #CG coefficients for the coupling of the total-spin basis 
                    S_coup +=  float(CG(a.s1, ms1_a, a.s2, a.ms2,S,MS).doit()) \
                                    * float(CG(b.s1, ms1_b, b.s2, b.ms2,S, MS).doit())

                # CG coefficients for the coupling of the spin-orbit basis i.e. Back to |alpha> basis.  
                # mj has to be ml + ms1, otherwise CG coefficient is zero 
                P_S[i,j] += S_coup * float(CG(a.l, ml_a, a.s1, ms1_a, a.j, a.mj).doit()) * \
                                float(CG(b.l, ml_b, b.s1, ms1_b, b.j, b.mj).doit())
                
    return np.array(P_S)



def MN_projector(N,MN,alpha,GSatom,s2=0.5):
    
    """
    Constructs the |N,MN><N,MN| projector matrix, for a given |N,MN>, in the |alpha> basis.

    Parameters:
        N (int/float)
        alpha (list of RydbergBasis objects): Basis states.

    Returns:    
        np.ndarray: N projector matrix.

    Description:
    we first uncouple the total spin basis |N,MN> --> |s1,ms1,F-MF> --> |s1,ms1,s2,ms2,I,MI>;
    MN = ms1 + ms2 + MI
    then recouple the spin-orbit basis |l,s1,ml,ms1,I,MI> --> |l,s1,j,mj,I,MI>;
    ml = mj-ms1
        
    """
    
    #F ranges from |I-s2| to (I+s2)
    FF = np.arange(np.abs(GSatom.I-s2),GSatom.I+s2+1,1)
    P_N = np.zeros([len(alpha),len(alpha)])
    
    #possible values of ms1
    ms = [-0.5,0.5]

    for i,a in enumerate(alpha):
        for j,b in enumerate(alpha):
            
            #N^2 operator is diagonal in the (n) subspace
            if a.n != b.n:
                continue;
            
            #N^2 = Sum_N Sum_{MN = -N to N} N(N+1) |N,MN><N,MN|
            
            #Use zip to iterate through MM and MM simultaneously
            for ms1_a, ms1_b in product(ms,ms):
                ml_a = a.mj - ms1_a
                ml_b = b.mj - ms1_b

                #MF has to be ms2 + mi, otherwise CG coefficient is zero
                if b.ms2 + b.mi != a.ms2 + a.mi:
                    continue

                #MN has to be ms1+ ms2 + mi, otherwise CG coefficient is zero
                if a.ms2 + a.mi + ms1_a != b.ms2 + b.mi + ms1_b:
                    continue

                N_coup=0
                for F_a,F_b in product(FF,FF):

                    N_coup +=  float(CG(a.s2, a.ms2, a.I, a.mi, F_a, a.ms2 + a.mi).doit()) \
                                * float(CG(a.s1, ms1_a, F_a, a.ms2 + a.mi, N, MN).doit()) \
                                * float(CG(b.s2, b.ms2, b.I, b.mi, F_b, b.ms2 + b.mi).doit()) \
                                * float(CG(b.s1, ms1_b, F_b, b.ms2 + b.mi, N, MN).doit()) \

                P_N[i,j] += N_coup * float(CG(a.l, ml_a, a.s1, ms1_a, a.j, a.mj).doit()) * \
                                float(CG(b.l, ml_b, b.s1, ms1_b, b.j, b.mj).doit())

    return P_N


def F2_operator(alpha,GSatom,s2=0.5):

    """
    Constructs the F^2 operator matrix in the |alpha> basis.
    Parameters:
        alpha (list of RydbergBasis objects): Basis states.
        GSatom (RydbergBasis object): Ground state alkali atom.
        s2 (float): Spin of the alkali atom (default is 0.5 for alkali atoms).
        
    Returns:
        np.ndarray: F^2 operator matrix.
        
    Description:    
        F^2 operator, explicitly expanded in the |alpha> basis (hyprefine uncoupled)
        where F = I2+S2, the hyperfine operator for the ground state alkali atom
        Acts as: F^2 |F,MF> = F(F+1) |F,MF> (diagonal in the coupled basis)
        we uncouple the total hyperfine basis |F,MF> --> |s2,ms2,I,mi>;  MF = ms2 + mi
    """

    #F ranges from |I-s2| to (I+s2+1)
    FF = np.arange(np.abs(GSatom.I-s2),s2+GSatom.I+1,1)
    
    F2 = np.zeros([len(alpha),len(alpha)])
    
    
    for i,a in enumerate(alpha):        
        for j,b in enumerate(alpha):

            #Hyperfine structure is diagonal in the Rydberg atom subspace
            if a.Ryd_quantum_numbers() != b.Ryd_quantum_numbers():
                continue;
            
            #F^2 = Sum_F Sum_{MF = -F to F} F(F+1) |F,MF><F,MF|
            #|F,MF><F,MF| in alpha basis =  <alpha: i,mi,s2,ms2|F,MF><F,MF|beta: i,mi,s2,ms2> \
            #                                   * dirac_delta(alpha_{n,l,j,mj},beta_{n,l,j,mj})

            for F in FF:
                #mf has to be mi + ms2
                F2[i,j] +=  F*(F+1) * float(CG(a.s2, a.ms2, a.I, a.mi,F, a.ms2 + a.mi).doit()) \
                            * float(CG(b.s2, b.ms2, b.I, b.mi,F, b.mi + b.ms2).doit())
    
    return F2





def S2_operator(alpha):
    """
    Constructs the S^2 operator matrix in the |alpha> basis.
    Parameters:
        alpha (list of RydbergBasis objects): Basis states.
    
    Returns:
        np.ndarray: S^2 operator matrix.

    Description:
        S^2 operator, explicitly expanded in the |alpha> basis (spin-orbit coupled, spin-spin uncoupled)
        where S = s1+s2, the total Spin operator for the scattering system
        Acts as: S^2 |S,MS> = S(S+1) |S,MS>, i.e. diagonal in the |S,MS> basis
        we first uncouple the total spin basis |S,MS> --> |s1,ms1,s2,ms2>;  MS = ms1 + ms2
        then recouple the spin-orbit basis |l,s1,ml,ms1> --> |l,s1,j,mj>; ml = mj-ms1
    """ 

    #For alkali atoms S= 0 or 1,  (from |s1-s2| to (s1+s2))
    SS = [0,1]
    S2 = np.zeros([len(alpha),len(alpha)])
    
    #possible values of ms1
    ms1 = [-0.5,0.5]

    for i,a in enumerate(alpha):
        for j,b in enumerate(alpha):
            
            #S^2 operator is diagonal in the (n,l ; I,mi) subspace
            if (a.n,a.I,a.mi,a.l) != (b.n,b.I,b.mi,b.l):
                continue;
            
            # Use itertools.product instead of nested loops
            for ms1_a, ms1_b in product(ms1, ms1):
                ml_a = a.mj - ms1_a
                ml_b = b.mj - ms1_b
                
                # MS has to be ms2 + ms1, and a.MS == b.MS,
                # otherwise CG coefficient is zero
                if b.ms2 + ms1_b != a.ms2 + ms1_a:
                    continue

                MS = ms1_a + a.ms2
                

                S_coup=0
                for S in SS:
                    #for MS in range(-S,S+1):
                    #CG coefficients for the coupling of the total-spin basis 
                    S_coup +=  S*(S+1) * float(CG(a.s1, ms1_a, a.s2, a.ms2,S,MS).doit()) \
                                    * float(CG(b.s1, ms1_b, b.s2, b.ms2,S, MS).doit())

                # CG coefficients for the coupling of the spin-orbit basis i.e. Back to |alpha> basis.  
                # mj has to be ml + ms1, otherwise CG coefficient is zero 
                
                S2[i,j] += S_coup * float(CG(a.l, ml_a, a.s1, ms1_a, a.j, a.mj).doit()) * \
                                float(CG(b.l, ml_b, b.s1, ms1_b, b.j, b.mj).doit())
                
                
                
                
    
    return S2



def N2_operator(alpha,GSatom,s2=0.5):
    """
    Constructs the N^2 operator matrix in the |alpha> basis.

    Parameters:
        alpha (list of RydbergBasis objects): Basis states.

    Returns:
        np.ndarray: N^2 operator matrix.

    Description:
        N = S + I = F + s1 = I +s2 + s1, the quasi-symmetry operator for the scattering system
        acts as N |N,MN> = N(N+1) |N,MN>, i.e. diagonal in the |N,MN> basis
        we first uncouple the total spin basis |N,MN> --> |s1,ms1,F,MF> --> |s1,ms1,s2,ms2,I,MI>;
        MN = ms1 + ms2 + MI
        then recouple the spin-orbit basis |l,s1,ml,ms1,I,MI> --> |l,s1,j,mj,I,MI>;
        ml = mj-ms1
    """

    
    FF = np.arange(np.abs(GSatom.I-s2),GSatom.I+s2+1,1)
    #for alkali atoms N ranges from |I-S| to I+S, where S=0,1
    NN = np.arange(np.abs(GSatom.I-1),GSatom.I+1+1,1)
    N2 = np.zeros([len(alpha),len(alpha)])
    
    #possible values of ms1
    ms = [-0.5,0.5]

    for i,a in enumerate(alpha):
        for j,b in enumerate(alpha):
            
            #N^2 operator is diagonal in the (n) subspace
            if a.n != b.n or a.l != b.l:
                continue;
            
            #N^2 = Sum_N Sum_{MN = -N to N} N(N+1) |N,MN><N,MN|
            
            # Use itertools.product instead of nested loops
            for ms1_a, ms1_b in product(ms,ms):
                ml_a = a.mj - ms1_a
                ml_b = b.mj - ms1_b

                
                #MN has to be ms1+ ms2 + mi, otherwise CG coefficient is zero
                if a.ms2 + a.mi + ms1_a != b.ms2 + b.mi + ms1_b:
                    continue

                if ml_a != ml_b:
                    print("N2 ml not equal")

                N_coup=0
                for F_a,F_b in product(FF,FF):
                    

                    for N in NN:
                        N_coup += N*(N+1) * float(CG(a.s2, a.ms2, a.I, a.mi, F_a, a.ms2 + a.mi).doit()) \
                                            * float(CG(a.s1, ms1_a, F_a, a.ms2 + a.mi, N, a.ms2 + a.mi + ms1_a).doit()) \
                                            * float(CG(b.s2, b.ms2, b.I, b.mi, F_b, b.ms2 + b.mi).doit()) \
                                            * float(CG(b.s1, ms1_b, F_b, b.ms2 + b.mi, N, b.ms2 + b.mi + ms1_b).doit()) \

                    
                N2[i,j] += N_coup * float(CG(a.l, ml_a, a.s1, ms1_a, a.j, a.mj).doit()) * \
                                float(CG(b.l, ml_b, b.s1, ms1_b, b.j, b.mj).doit())
                
    return N2

def Rydberg_projector(n,l,j,alpha):
    """
    Constructs the Rydberg projector matrix for a given electronic state |n,l,j> in the |alpha> basis.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        j (float): Total angular momentum quantum number.
        alpha (list of RydbergBasis objects): Basis states.
    
    Returns:
        np.ndarray: Rydberg projector matrix.
    """

    P_ryd = np.zeros([len(alpha),len(alpha)])
    p=0
    for a in alpha:
        q=0
        for b in alpha:
            T1 = (a.n==b.n and a.l==b.l and a.j==b.j)
            T2 = (a.GS_quantum_numbers() == b.GS_quantum_numbers())
             
            if a.n==n and a.l==l and a.j==j and T1 and T2:
                P_ryd[p,q] = 1.0
                
            
            q=q+1
            #P_{nlj} = Sum(|n,l,j,quant_num><n,l,j,quant_num|) for all other quantum numbers
            #including mj = -j to j
        p = p+1

    return P_ryd



def Rydberg_projector_alt(n,l,j,alpha):
    """
    Constructs the Rydberg projector matrix for a given electronic state |n,l,j> in the |alpha> basis.
    Alternate function definition that uses a different approach to construct the projector.

    Parameters:
        n (int): Principal quantum number.
        l (int): Orbital angular momentum quantum number.
        j (float): Total angular momentum quantum number.
        alpha (list of RydbergBasis objects): Basis states.
    
    Returns:
        np.ndarray: Rydberg projector matrix.
    """
    P_ryd = np.zeros([len(alpha),len(alpha)])
    
    for i,a in enumerate(alpha):
        if a.n==n and a.l==l and a.j==j:
            f_mj = np.zeros(len(alpha))
            f_mj[i] = 1.0
            P_ryd = P_ryd + np.outer(f_mj,f_mj)
            
            #P_{nlj} = Sum(|n,l,j,quant_num><n,l,j,quant_num|) for all other quantum numbers
            #including mj = -j to j
        
    return P_ryd








