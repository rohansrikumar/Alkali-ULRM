
"""                                 Alkali-ULRM 

This program calculates the electronic structure of Ultralong-range Rydberg molecules,
composed of an Alkali Rydberg atom bound to a ground state atom via quantum scattering.

The file contains the class structure needed to represent the Rydberg atom and the
Ground-state atom in a given basis, construct the Hamiltonian, and calculate the electronic 
wavefunctions via diagonalization.

Refer to "Hamiltonian for the inclusion of spin effects in long-range Rydberg molecules"
by M. Eiles et. al for theory and cited equations 
https://journals.aps.org/pra/pdf/10.1103/PhysRevA.95.042515
"""


import numpy as np  
import arc  # Import ARC (Alkali Rydberg Calculator)
from scipy.interpolate import interp1d
from sympy.physics.quantum.cg import CG
import scipy as sp
from scipy.constants import physical_constants, epsilon_0, hbar


from scipy.constants import c as C_c #speed of light
from scipy.constants import h as C_h #Planck's constant
from scipy.constants import e as C_e #electron charge


gamma = sp.special.gamma
sph_harm = sp.special.sph_harm
legendre = sp.special.lpmv

unit = 2*3.289841960355* (10**6) #a.u. to GHz
eV   = 27.2113245703 #ev to a.u.


#the ground-state class denoting the electron-atom scattering basis
#with the ground-state atom as the center of coordinates
class ScatteringBasis:
    def __init__(self,S,L,J,MJ,MI,Omega):

        #S: total electron spin, L: total orbital angular momentum
        #J: total angular momentum, MJ: projection of J along the z-axis
        #I: nuclear spin, MI: projection of I along the z-axis
        self.L = L
        self.J = J
        self.S = S
        self.MJ = MJ
        self.MI = MI

        #Omega = MJ + MI, projection of total angular momentum along the z-axis
        #is a conserved quantum number in the field-free scattering process
        self.Omega =Omega
        
    #S,L,J dependent electron-atom scattering lengths
    def a_SL(self,k,phase_shifts):
        
        if self.L == 0 and self.S == 0 and self.J == 0:
            a = -np.tan(phase_shifts[0](k))/k
        elif self.L == 0 and self.S == 1 and self.J== 1:
            a = -np.tan(phase_shifts[1](k))/k
        elif self.L == 1 and self.S == 0 and self.J == 1:
            a = -np.tan(phase_shifts[2](k))/k**3
        elif self.L == 1 and self.S == 1 and self.J == 0:
            a = -np.tan(phase_shifts[3](k))/k**3
        elif self.L == 1 and self.S == 1 and self.J == 1:
            a = -np.tan(phase_shifts[4](k))/k**3
        elif self.L == 1 and self.S == 1 and self.J == 2:
            a = -np.tan(phase_shifts[5](k))/k**3
            
        return  a
    
    def Scattering_quantum_numbers(self):
        #return list of quantum numbers of the scattering system
        return [self.S,self.L,self.J,self.Omega]
    
        


#the rydberg class denoting the electronic structure of the Rydberg-atom 
#and ground-state atom with the Rydberg ionic core as the center of coordinates
class RydbergBasis:                                     
    def __init__(self,RydbergAtom,n,l,j,mj,s1,n2,l2,j2,s2,ms2,I,mi):
        
        #Rydberg electron quantum numbers
        #n: principal quantum number, l: orbital angular momentum
        #j: total electronic angular momentum, mj: projection of j along the z-axis
        #s1: electron spin of the Rydberg atom
        self.RydbergAtom = RydbergAtom
        self.n = n
        self.l = l
        self.j = j
        self.mj = mj
        self.s1 = s1
        
        #Ground-state atom  (valence electron) quantum numbers
        #n2: principal quantum number; l2: orbital angular momentum
        #j2: total electronic angular momentum; mj2: projection of j2 along the z-axis
        #s2: electron spin; ms2: projection of s2 along the z-axis
        #I: nuclear spin; mi: projection of I along the z-axis
        self.n2 = n2
        self.l2 = l2
        self.j2 = s2
        self.s2 = s2
        self.ms2 = ms2
        self.I = I
        self.mi = mi
        
    def Ryd_quantum_numbers(self):
        #return list of quantum numbers of the Rydberg atom 
        return [self.n,self.l,self.j,self.mj,self.s1]
    
    def GS_quantum_numbers(self):
        #return list of quantum numbers of the ground-state atom
        return [self.I,self.mi,self.s2,self.ms2]
    

    #Rydberg wavefunction in the Rydberg atom frame of reference
    #defined for any non-uniform or uniform grid
    def Rydberg_wavefunction(self,R):

        RydbergAtom = self.RydbergAtom
        n,l,j,_,s1=self.Ryd_quantum_numbers()

        R_min = RydbergAtom.alphaC/3.0
        R_in, R_out,R_step =  R_min, R[-1]+1, 0.005
        
        #Rydberg wavefunction: Look at ARC documentation for function details
        R_nlj, f_nlj = RydbergAtom.radialWavefunction(l,s1,j,\
                                    RydbergAtom.getEnergy(n, l, j)/eV,\
                                        R_in,R_out, R_step)

        #To utilize the same R-grid for all calculation
        #Relevant for non-uniform R grids
        wavefunc = interp1d(R_nlj,f_nlj,kind="cubic")
        
        return R,wavefunc(R)

    #General spherical harmonics (Spin-Orbit coupling)
    def GenSpher(self,theta):   
        
        s=self.s1
        mss =[-s,s]

        j=self.j
        mj = self.mj
        n=self.n
        
        l=self.l
        Y=np.zeros(len(theta))
        for ms in mss:
            ml = mj-ms
            Y = Y +float(CG(l,ml,s,ms,j, mj).doit())*Spher(l,ml,theta,0)
            #print("n,l,s,j,ml,ms,mj",n,l,s,j,ml,ms,mj,sph_harm(ml,l,0,0),Spher(l,ml,0,0))

        return Y

# Equations 10 to 12
def Q(l,L,ML,R,f_nlj):
    
    if L==0 and ML ==0:
        return f_nlj/R * np.sqrt((2*l+1)/4/np.pi)
    
    if L==1 and ML ==0:
        return np.gradient(f_nlj/R,R) * np.sqrt((2*l+1)/4/np.pi)
    
    if L==1 and np.abs(ML)==1:
        return f_nlj/R**2 * np.sqrt(l*(l+1)*(2*l+1)/8/np.pi)
    
    else:
        print("wrong input for Q: L <=1 and abs(ML) <=1 not satisfied")
        exit()

                            

#Basis: ground-state atom frame of reference  |L,S,J,MJ,MI,Omega>        
def beta_basis(Omega,GSatom):
    beta_full =[]
    SS =[0.0,1.0]  #Default alkali metals: spin 1/2
    LL =[0.0,1.0]  #Only S and P scattering channels
    for L in LL:
        for S in SS:
            JJ = np.arange(np.abs(S - L),S+L+1,1)
            for J in JJ:
                MMJ = np.arange(-J,J+1,1)
                for MJ in MMJ:
                    MMI = np.arange(-GSatom.I,GSatom.I+1,1)
                    for MI in MMI:
                        beta_full.append(ScatteringBasis(S,L,J,MJ,MI,Omega))
    
    
    ## limiting ground-state basis to only states with a given Omega 
    beta = []
    for b in beta_full:
        if b.MJ + b.MI == Omega:
            beta.append(b)
            
    return beta



#Basis: rydberg frame of referenece 
def alpha_basis(RydbergAtom,n_min,n_max,E_min,E_max,Omega,GSatom):
    
    n2,l2,s2,j2 = GSatom.groundStateN,0,0.5,0.5  
    #default alkali atom ground-state |n,l=0,s=1/2,j=1/2>
    alpha_full=[]
    s1=0.5  #default rydberg atom spin

    for n in range(n_min,n_max+5):
        for l in range(0,n):
            jj = np.arange(np.abs(l-s1),l+s1+1,1)
            for j in jj:
                E= RydbergAtom.getEnergy(n, l, j)/eV 
                if E >= E_min and E <= E_max:
                    mmj = np.arange(-j,j+1,1) 
                    for mj in mmj:
                        mms2 = [-0.5,0.5]
                        for ms2 in mms2:
                            mmi = np.arange(-GSatom.I,GSatom.I +1,1)
                            for mi in mmi:
                                alpha_full.append(RydbergBasis(RydbergAtom,n, l, j,mj,s1,n2,l2,j2,s2,ms2,GSatom.I,mi))


    ## limiting Rydberg basis to only states with a given Omega
    alpha=[]
    for a in alpha_full:
        if a.ms2 + a.mi + a.mj == Omega:
            alpha.append(a)  
            
    return alpha




#Scattering hamiltonian from Ground-state perspective
#Equation 18
def H_Scattering(beta,k,phase_shifts):
    H_U = np.zeros([len(beta),len(beta),k.size])
    p=0
    for b in beta:
        H_U[p,p,:] = (2*b.L +1)**2 * b.a_SL(k,phase_shifts) /2
        p=p+1
    return H_U

#Rydberg electron hamiltonian w.r.t Rydberg core; spin-orbit included
#Diagonal in rydberg atom basis
def H_Rydberg(alpha,RydbergAtom):
    H_ryd =[]
    for a in alpha:
        H_ryd.append(RydbergAtom.getEnergy(a.n, a.l, a.j) / eV  )

    H_ryd =np.diag(np.array(H_ryd))
    return H_ryd



##Hyperfine structure of Ground-state atom: Equation (7)
def H_Hyperfine(alpha,GSatom):

    n2,l2,j2,s2 = alpha[0].n2, alpha[0].l2, alpha[0].j2,alpha[0].s2
    A = GSatom.getHFSCoefficients(n2, l2, j2)[0]*10**(-9)/unit  
    #Hyperfine coefficient in a.u.
    #the function output is in GHz, so we convert it to a.u.
    
    H_hfs = np.zeros([len(alpha),len(alpha)])
    FF = np.arange(np.abs(GSatom.I-s2),GSatom.I+s2+1,1)
    p,q=0,0
    for a in alpha:
        q=0
        for b in alpha:
            if a.Ryd_quantum_numbers() == b.Ryd_quantum_numbers():
                s=0
                for F in FF:
                    MF = np.arange(-F,F+1,1)
                    for mf in MF:
                        if a.ms2 + a.mi == mf and b.ms2 + b.mi == mf: 
                            s = s+ float(CG(a.s2, a.ms2, a.I, a.mi,F, mf).doit()) \
                                * float(CG(b.s2, b.ms2, b.I, b.mi, F, mf).doit()) \
                                    * GSatom.getHFSEnergyShift(s2, F, A)

                H_hfs[p,q] = s 
            q=q+1
        p=p+1
        
    return H_hfs



#projector for a given F value, explicitly expanded in the |alpha> basis
#where |alpha> = |n,l,j,s,mj,i,mi,s2,ms2>
def F_projector(F,alpha):
    MF = np.arange(-F,F+1,1)
    P_F = np.zeros([len(alpha),len(alpha)])
    
    
    for i,a in enumerate(alpha):        
        for j,b in enumerate(alpha):

            #dirac delta criterion for Rydberg atom quantum number
            #dirac_delta(alpha_{n,l,j,mj},beta_{n,l,j,mj})
            if a.Ryd_quantum_numbers() != b.Ryd_quantum_numbers():
                continue;
            
            #P_F = Sum |F,MF><F,MF|
            #|F,MF><F,MF| in alpha basis =  <alpha: i,mi,s2,ms2|F,MF><F,MF|beta: i,mi,s2,ms2> \
            #                                   * dirac_delta(alpha_{n,l,j,mj},beta_{n,l,j,mj})
            for mf in MF:
                P_F[i,j] +=  float(CG(a.s2, a.ms2, a.I, a.mi,F, mf).doit()) \
                            * float(CG(b.s2, b.ms2, b.I, b.mi,F, mf).doit())
    
    return P_F


#Projector for a given Rydberg-electronic state, in the |alpha> basis
#Alternate function definition
def Rydberg_projector_alt(n,l,j,alpha):
    P_ryd = np.zeros([len(alpha),len(alpha)])
    
    for i,a in enumerate(alpha):
        if a.n==n and a.l==l and a.j==j:
            f_mj = np.zeros(len(alpha))
            f_mj[i] = 1.0
            P_ryd = P_ryd + np.outer(f_mj,f_mj)
            
            #P_{nlj} = Sum(|n,l,j,quant_num><n,l,j,quant_num|) for all other quantum numbers
            #including mj = -j to j
        
    return P_ryd

#Projector for a given Rydberg-electronic state, explicitly expanded in the |alpha> basis
def Rydberg_projector(n,l,j,alpha):
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


    
#Change of coordinates from Ground State to Rydberg center
#Equation (19)
def A_transform(alpha,beta,R):
    #alpha: Rydberg basis, beta: ground-state basis
    #A_matrix: is the transformation matrix from the ground-state basis to the Rydberg basis
    A_matrix = np.zeros([len(alpha),len(beta),len(R)])

    for p,a in enumerate(alpha):
        
        R,f_nlj = a.Rydberg_wavefunction(R)

        n,l,j,mj,s1 = a.Ryd_quantum_numbers()
        _,_,s2,ms2 = a.GS_quantum_numbers()

        for q,b in enumerate(beta):
            S,L,J,Om = b.Scattering_quantum_numbers()
            fL = np.sqrt(4*np.pi/(2*L+1))
            MLL = np.arange(-L,L+1,1 )
            s = np.zeros(len(R))

            #make sure that nuclear spin projection
            #remains the same in both reference frames
            if a.mi != b.MI:
                print("Check: mi of Rydberg atom does not match MI of ground-state atom")
                continue

            #mixing of MLs
            for ML in MLL:
                if np.abs(ML) <= l and np.abs(mj-ML)<=0.5 and np.abs(mj+ms2-ML) <= S and np.abs(ms2+mj)<= J:
                    s= s+ fL * float(CG(l, ML, s1, mj-ML, j, mj).doit()) * Q(l,L,ML,R,f_nlj) \
                        * float(CG(s1, mj-ML, s2, ms2, S, mj - ML + ms2).doit()) \
                            * float(CG(L,ML,S,mj-ML+ms2,J, mj + ms2).doit())


            A_matrix[p,q,:] = s.copy()
            
        

    return A_matrix


def J_projector(A_matrix,states):
    beta_w = np.zeros(A_matrix[0,:,:].shape)
   
    for i in range(A_matrix[0,0,:].size):
        beta_w[:,i] = np.dot((A_matrix[:,:,i].transpose()).conj(),states[:,i])
        #norm = np.sqrt(np.abs(np.vdot(beta_w[:,i],beta_w[:,i])))
        #beta_w[:,i] = beta_w[:,i]/norm
        
    beta_1S0 = np.abs(beta_w[0,:])**2
    beta_3S1 = np.sum(np.abs(beta_w[1:4,:])**2,axis=0)
    beta_1P1 = np.sum(np.abs(beta_w[4:7,:])**2,axis=0)
    beta_3P0 = np.abs(beta_w[7,:])**2
    beta_3P1 = np.sum(np.abs(beta_w[8:11,:])**2,axis=0)    
    beta_3P2 = np.sum(np.abs(beta_w[11:16,:])**2,axis=0)
    norm = beta_1S0 + beta_1P1 + beta_3S1 + beta_3P0 + beta_3P1+beta_3P2
    return (beta_1P1)/norm







#Calculating the electronic dipole transition matrix 
#between all possible Rydberg states in the |alpha> basis 
# and a given final state (Ryd class instance)
def dipole_transition_basis(alpha,atom,final_state):
    list_dipole = np.zeros([len(alpha)])
    b=final_state
    for a in alpha:
        list_dipole = atom.getDipoleMatrixElement(a.n,a.l,a.j,a.mj,b['n'],b['l'],b['j'],b['mj'],b['mj'] - a.mj)

    return list_dipole

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
                    dip_list_alpha  = dipole_transition_basis(alpha,atom,final_state)

                    Omega = 2*np.pi * (freq_eq - atom.getEnergy(n,l,j)*C_e/C_h)
                    #imported from arc, C_e: e, C_h: h i.e. electron charge and plancks constant

                    Dipole_molecule = np.abs(np.sum(dip_list_alpha*state_coefs_eq)) \
                        * (physical_constants["Bohr radius"][0]) * C_e
                    
                    degeneracy = 1#(2*j + 1)/(2*j1 + 1)
                    #
                    
                    Transition_rate = Transition_rate + Dipole_molecule**2 *Omega**3/ (3 * np.pi * epsilon_0 * hbar * C_c**3)*degeneracy

    return (1/Transition_rate)


def atomic_lifetime(atom,n,l,j,mj):

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



#utilizes wigner eckart theorem to average over m-values
def atomic_lifetime_averaged(atom,n,l,j):

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
             



#Calculating the electronic dipole transition matrix 
#between all possible Rydberg states in the |alpha> basis 
def dipole_matrix(alpha,atom):
    V_dipole = np.zeros([len(alpha),len(alpha)])
    for i in range(len(alpha)):
        for j in range(len(alpha)):
            a,b = alpha[i],alpha[j]

            #Pre-screening dipole transistion rules
            if  np.abs(a.l - b.l) !=1 or np.abs(a.j - b.j) >1:
                continue;

            #Making sure other quantum numbers are unchanged
            if a.GS_quantum_numbers() != b.GS_quantum_numbers():
                continue;
            V_dipole[i,j] = atom.getDipoleMatrixElement(a.n,a.l,a.j,a.mj,b.n,b.l,b.j,b.mj,b.mj - a.mj)
            

    return V_dipole

#d(R): dipole moment calculator of ULRM
#The rydberg dipole transition matrix is necessary input
def dipole_moment(state,V_dipole):
    
    sum_tot = np.einsum('nr,nm,mr-> r',state,V_dipole,state.conj())
    #Summation(m,n) C_m* C_n <phi_m|dipole|phi_n>
    return sum_tot
    

#Making sure the phase change is adiabatic along R
#smoothens out any discrete sign changes
def phase_smoothening(states):                         
    for i in range(len(states[0,0,:])-1):
        for j in range(len(states)):
            if np.vdot(states[:,j,i],states[:,j,i+1]) < 0.0:
                states[:,j,i+1] = -1*states[:,j,i+1]
    return states

#Diagonalizing electronic hamiltonian  
def diagonalization(H_ryd,H_hfs,H_U,A_matrix,R):

    
    energy=np.zeros([*H_ryd[0].shape, R.size])   
    states=np.zeros([*H_ryd.shape, R.size])
    
    for i,r in enumerate(R):
        H_V = np.dot(np.dot(A_matrix[:,:,i],H_U[:,:,i]),np.conjugate(A_matrix[:,:,i].transpose()))
        H = H_ryd + H_V + H_hfs 
        w,v = np.linalg.eigh(H)
        energy[:,i] = w.copy()
        states[:,:,i] = v.copy()
        
        #print(i)
        
        
    states = phase_smoothening(states)
    return energy,states


def elec_wf(coef,alpha,R):
    
    U_nlj = np.array([a.Rydberg_wavefunction(R)[1] for a in alpha]) #a.GenSpher([0.0]) for a in alpha])
    psi_nlj = np.sum(U_nlj*coef[:,None],axis=0)
    #print(psi_nlj.shape,U_nlj.shape,coef[:,None].shape)
    tt = np.linspace(0,2*np.pi,10)
    print("PSI:",psi_nlj)
    #print("My Ylm",[alpha[0].GenSpher([t]) for t in tt])
    
    return psi_nlj
    
def anion_wf(R,Rin):
    x = np.abs(R-Rin)
    l=1.35094
    r0 = 9.004786
    n=0
    z= np.sinh(x/r0)
    F12 = sp.special.hyp2f1(-n,-2*l + n +1,1.5,-z**2)
    #print("Hypergeometery:",F12)
    wf = (np.cosh(x/r0))**(-2*l)*z*F12
    #print("wf:",wf.shape,"x:",x.shape)
    return wf


#given phase-shifts are interpolated to be used as functions
#in the (k-dependent) calcualtions of H_U
def phase_interpolation(phases):
    phase_shifts = []
    for i in range(1,phases[0].size):
        phase_shifts.append(interp1d(phases[:,0],phases[:,i],kind="cubic"))

    return phase_shifts


#Semiclassical kinetic energy of the Rydberg electron
#k_cut gives a lower limit below which k is forced to be 0
@np.vectorize
def kinetic_energy(n,r,k_cut=0):                                                  
    k= np.emath.sqrt(2 * (1/r - 1/(2*n**2))).real
    if k < k_cut:
        k =k_cut
    return k


#Spherical harmonics Y_lm(theta,phi)
def Spher(l,m,theta,phi):
	c = legendre(m,l,np.cos(theta)) * np.exp(1j * m * phi)
	t = c * np.emath.sqrt(gamma(l-m+1)*(2*l+1)/(gamma(l+m+1)*4.0*np.pi))
	return t

