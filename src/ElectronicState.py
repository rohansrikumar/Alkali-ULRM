
"""                                 Alkali-ULRM: Electronic State Module 

This program calculates the electronic structure of Ultralong-range Rydberg molecules,
composed of an Alkali Rydberg atom bound to a ground state atom via quantum scattering.

The file contains the class structure needed to represent the Rydberg atom and the
Ground-state atom in a given basis, construct the Hamiltonian, and calculate the electronic 
wavefunctions via diagonalization.

Refer to "Hamiltonian for the inclusion of spin effects in long-range Rydberg molecules"
by M. Eiles et. al for theory and cited equations 
Ref [1]: https://journals.aps.org/pra/pdf/10.1103/PhysRevA.95.042515
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


#necessary functions from scipy.special
#Gamma function, spherical harmonics, and associated Legendre polynomials
gamma = sp.special.gamma
sph_harm = sp.special.sph_harm
legendre = sp.special.lpmv

#convert a.u. to GHz, 2*3.289841960355* (10**6) a.u. is 1GHz
GHz = 2*3.289841960355* (10**6) 

#convert eV to au, 1 a.u. is 27.2113245703 eV
eV   = 27.2113245703 

class ScatteringState:
    """
    Represents an electron-atom scattering state with the ground-state atom 
    as the center of coordinates.

    Attributes:
        S (int): Total electron spin (0 or 1 for alkaline ULRMs).
        L (int): Total orbital angular momentum (0 for S-wave, 1 for P-wave scattering).
        J (int): Total angular momentum, ranges from |L-S| to L+S.
        MJ (int): Projection of J along the z-axis.
        MI (int): Projection of nuclear spin I along the z-axis.
        Omega (int): Projection of total angular momentum along the z-axis (MJ + MI).
    """
    def __init__(self,S,L,J,MJ,MI,Omega):
        
        
        self.L = L
        self.J = J
        self.S = S
        self.MJ = MJ
        self.MI = MI
        self.Omega =Omega
        
    
    def a_SL(self,k,phase_shifts):
        """
        Calculates the S, L, J-dependent electron-atom scattering lengths.

        Parameters:
            k (np.ndarray): semi-classical electron momentum (or wave-number, hbar=1).
            phase_shifts (list of callable): List of phase shift functions.

        Returns:
            np.ndarray: Scattering length based on the given parameters.
        """

        match (self.L, self.S, self.J):
            case (0, 0, 0):
                a = -np.tan(phase_shifts[0](k)) / k
            case (0, 1, 1):
                a = -np.tan(phase_shifts[1](k)) / k
            case (1, 0, 1):
                a = -np.tan(phase_shifts[2](k)) / k**3
            case (1, 1, 0):
                a = -np.tan(phase_shifts[3](k)) / k**3
            case (1, 1, 1):
                a = -np.tan(phase_shifts[4](k)) / k**3
            case (1, 1, 2):
                a = -np.tan(phase_shifts[5](k)) / k**3
            case _:
                raise ValueError(f"Invalid combination of L={self.L}, S={self.S}, J={self.J}.")
            
        return a
    
    def Scattering_quantum_numbers(self):
        """
        List of quantum numbers that define the scattering state
        Returns:
            list: [S, L, J, Omega]
        """
        return [self.S,self.L,self.J,self.Omega]
    
        



class RydbergState:                                     
    def __init__(self,RydbergAtom,n,l,j,mj,s1,n2,l2,j2,s2,ms2,I,mi):
        """
        Represents the electronic states of a Rydberg atom and the ground-state atom
        in the Rydberg ionic core frame of reference.

        Attributes:
            RydbergAtom (object): Rydberg atom object from ARC.
            n (int): Principal quantum number of the Rydberg atom.
            l (int): Orbital angular momentum quantum number of the Rydberg atom.
            j (float): Total electronic angular momentum quantum number of the Rydberg atom.
            mj (float): Projection of j along the z-axis for the Rydberg atom.
            s1 (float): Electron spin of the Rydberg atom.

            n2 (int): Principal quantum number of the ground-state atom's valence electron.
            l2 (int): Orbital angular momentum quantum number of the ground-state atom's valence electron.
            j2 (float): Total electronic angular momentum quantum number of the ground-state atom's valence electron.
            s2 (float): Electron spin of the ground-state atom's valence electron.
            ms2 (float): Projection of s2 along the z-axis for the ground-state atom's valence electron.
            I (float): Nuclear spin of the ground-state atom.
            mi (float): Projection of I along the z-axis for the ground-state atom."""
        
        #Rydberg atom quantum numbers
        self.RydbergAtom = RydbergAtom
        self.n = n
        self.l = l
        self.j = j
        self.mj = mj
        self.s1 = s1
        
        #Ground-state atom  (valence electron) quantum numbers
        self.n2 = n2
        self.l2 = l2
        self.j2 = j2
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
    

    
    def Rydberg_wavefunction(self,R):
        """
        Calculates the Rydberg wavefunction on a given radial grid R.
        Parameters:
            R (np.ndarray): Radial grid points at which to evaluate the wavefunction.
        Returns:    
            R, wavefunction
        """

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

    
    def GenSpher(self,theta):   

        """
        Calculates the spherical harmonics for the Rydberg wavefunction with Spin-Orbit coupling

        Parameters:
            theta (np.ndarray): Angles at which to evaluate the spherical harmonics.

        Returns:
            np.ndarray: Spherical harmonics evaluated at the specified angles.
        
        """
        
        n,l,j,mj,s1=self.Ryd_quantum_numbers()
        ms =[-s1,s1]
        Y_j=np.zeros(len(theta))
        for ms1 in ms:
            ml = mj-ms1
            Y_j += float(CG(l,ml,s1,ms1,j, mj).doit())*Spher(l,ml,theta,0)
            #print("n,l,s,j,ml,ms,mj",n,l,s,j,ml,ms,mj,sph_harm(ml,l,0,0),Spher(l,ml,0,0))

        return Y_j


def Q(l,L,ML,R,f_nlj):
    """
    Calculates the Q function. See Equations 10 to 12 in Ref[1] for details.
    """
    
    if L==0 and ML ==0:
        Q=f_nlj/R * np.sqrt((2*l+1)/4/np.pi)
    
    elif L==1 and ML ==0:
        Q=np.gradient(f_nlj/R,R) * np.sqrt((2*l+1)/4/np.pi)
    
    elif L==1 and np.abs(ML)==1:
        Q=f_nlj/R**2 * np.sqrt(l*(l+1)*(2*l+1)/8/np.pi)
    
    else:
        raise ValueError("wrong input (L,ML) for Q")
    
    return Q
        

class ScatteringBasis:
    def __init__(self, Omega, GSatom, S_list=(0.0, 1.0), L_list=(0.0, 1.0)):
        """
        Constructs the scattering basis for the ground-state atom frame of reference 
        |beta> =  |L,S,J,MJ,MI,Omega> (ScatteringState instance).
        Default values for S are (0.0, 1.0) i.e, spin singlet and triplet states.
        Default values for L are (0.0, 1.0) i.e, S-wave and P-wave scattering.

        Parameters:
            Omega (int): Projection of total angular momentum along the z-axis (MJ + MI).
            GSatom (object): Ground-state atom object from ARC.
            S_list (list): List of total electron spin values to consider.
            L_list (list): List of total orbital angular momentum values to consider.

        Returns:
            list: List of ScatteringState instances representing the scattering basis.
        """

        self.Omega = Omega
        self.GSatom = GSatom
        #Ensure S_list and L_list are lists or tuples, otherwise convert to lists
        self.S_list = S_list if isinstance(S_list, (list, tuple)) else [S_list]
        self.L_list = L_list if isinstance(L_list, (list, tuple)) else [L_list]
        self.beta_basis = self._generate_basis()

    def _generate_basis(self):
        """
        Generates the full scattering basis and filters it based on Omega (internal method).
        Returns: List of ScatteringState instances.
        """
        beta_full = []
        for L in self.L_list:
            for S in self.S_list:
                JJ = np.arange(np.abs(S - L), S + L + 1, 1)
                for J in JJ:
                    MMJ = np.arange(-J, J + 1, 1)
                    for MJ in MMJ:
                        MMI = np.arange(-self.GSatom.I, self.GSatom.I + 1, 1)
                        for MI in MMI:
                            beta_full.append(ScatteringState(S, L, J, MJ, MI, self.Omega))

        #Filter states based on Omega
        return [b for b in beta_full if b.MJ + b.MI == self.Omega]

    def get_basis_states(self):
        """
        Returns the list of basis states.
        """
        return self.beta_basis
                            

       

class RydbergBasis:
    def __init__(self,RydbergAtom,n_range,E_range,Omega,GSatom):    
        """
        Constructs the electronic basis for the diatomic system in the rydberg ionic core frame of reference.
        |alpha> = {|n,l,j,mj,s1,n2,l2,j2,s2,ms2,I,mi>} (list of RydbergState instance).

        Parameters:
            RydbergAtom (object): Rydberg atom object from ARC.
            n_range (tuple): Range of principal quantum numbers (n_min, n_max).
            E_range (tuple): Range of energies (E_min, E_max).
            Omega (int): Projection of total angular momentum along the z-axis (mj + ms2 + mi).
            GSatom (object): Ground-state atom object with necessary properties and methods.
        
        """
        self.RydbergAtom = RydbergAtom
        self.Omega = Omega
        self.GSatom = GSatom
        self.n_min, self.n_max = n_range
        self.E_min, self.E_max = E_range

        if self.n_max < self.n_min:
            raise ValueError("n_max must be >= n_min.")
        if self.E_max < self.E_min:
            raise ValueError("E_max must be >= E_min.")
        
        self.alpha_basis = self._generate_basis()
        
    def _generate_basis(self):
        """
        Generates the full electronic basis and filters it based on Omega (internal method).
        Returns: List of RydbergState instances.
        """
        GSatom, RydbergAtom = self.GSatom, self.RydbergAtom

        #default alkali atom ground-state |n,l=0,s=1/2,j=1/2>
        n2,l2,s2,j2 = GSatom.groundStateN,0,0.5,0.5  
        
        alpha_full=[]
        s1=0.5  #default rydberg atom spin

        for n in range(self.n_min,self.n_max+5):
            for l in range(0,n):
                jj = np.arange(np.abs(l-s1),l+s1+1,1)
                for j in jj:
                    E= RydbergAtom.getEnergy(n, l, j)/eV 
                    if E >= self.E_min and E <= self.E_max:
                        mmj = np.arange(-j,j+1,1) 
                        for mj in mmj:
                            mms2 = [-0.5,0.5]
                            for ms2 in mms2:
                                mmi = np.arange(-GSatom.I,GSatom.I +1,1)
                                for mi in mmi:
                                    alpha_full.append(RydbergState(RydbergAtom,n, l, j,mj,s1,n2,l2,j2,s2,ms2,GSatom.I,mi))


        ## limiting Rydberg basis to only states with a given Omega
        return [ a for a in alpha_full if a.ms2 + a.mi + a.mj == self.Omega ]
        
    def get_basis_states(self):
        """
        Returns the list of basis states.
        """
        return self.alpha_basis







#Scattering hamiltonian from Ground-state perspective
#Equation 18
def H_Scattering(beta,k,phase_shifts):
    H_U = np.zeros([len(beta),len(beta),k.size])
    
    for p,b in enumerate(beta):
        H_U[p,p,:] = (2*b.L +1)**2 * b.a_SL(k,phase_shifts) /2
        
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
    A = GSatom.getHFSCoefficients(n2, l2, j2)[0]*10**(-9)/GHz
    #Hyperfine coefficient in a.u.
    #the function output is in GHz, so we convert it to a.u.
    
    H_hfs = np.zeros([len(alpha),len(alpha)])
    FF = np.arange(np.abs(GSatom.I-s2),GSatom.I+s2+1,1)
    
    for p,a in enumerate(alpha):   
        for q,b in enumerate(alpha):

            #Hyperfine structure is diagonal in the Rydberg atom subspace
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

        
        
    return H_hfs






    

def A_transform(alpha,beta,R):
    """
    Constructs the transformation matrix A from the ground-state basis to the Rydberg basis.
    see Equation (19) in Ref. [1] for details
    Parameters:
        alpha (list): List of RydbergBasis instances.
        beta (list): List of ScatteringState instances.
        R (np.ndarray): Radial grid points.
    Returns:    
        np.ndarray: Transformation matrix A with dimensions [len(alpha), len(beta), len(R)].
    """
    
    A_matrix = np.zeros([len(alpha),len(beta),len(R)])

    for p,a in enumerate(alpha):
        
        R,f_nlj = a.Rydberg_wavefunction(R)

        _,l,j,mj,s1 = a.Ryd_quantum_numbers()
        _,_,s2,ms2 = a.GS_quantum_numbers()

        for q,b in enumerate(beta):
            S,L,J,_ = b.Scattering_quantum_numbers()
            fL = np.sqrt(4*np.pi/(2*L+1))
            MLL = np.arange(-L,L+1,1 )
            s = np.zeros(len(R))

            #make sure that nuclear spin projection
            #remains the same in both reference frames
            if a.mi != b.MI:
                continue

            #mixing of MLs
            for ML in MLL:
                if np.abs(ML) <= l and np.abs(mj-ML)<=0.5 and np.abs(mj+ms2-ML) <= S and np.abs(ms2+mj)<= J:
                    s= s+ fL * float(CG(l, ML, s1, mj-ML, j, mj).doit()) * Q(l,L,ML,R,f_nlj) \
                        * float(CG(s1, mj-ML, s2, ms2, S, mj - ML + ms2).doit()) \
                            * float(CG(L,ML,S,mj-ML+ms2,J, mj + ms2).doit())


            A_matrix[p,q,:] = s.copy()
            
        

    return A_matrix


             



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



def dipole_moment(state,V_dipole):
    """
    d(R): dipole moment calculator of ULRM
    Parameters:
        state (np.ndarray): Electronic state vector in the Rydberg basis (|alpha>).
        V_dipole (np.ndarray): Dipole transition matrix in the Rydberg basis (|alpha>).
    Returns:    
        np.ndarray: Dipole moment vector for the electronic state as a function of R.
    """
    
    sum_tot = np.einsum('nr,nm,mr-> r',state,V_dipole,state.conj())
    #Summation(m,n) C_m* C_n <phi_m|dipole|phi_n>
    return sum_tot
    


def phase_smoothening(states):
    """
    Smoothens the phase of the electronic states to ensure adiabatic continuity.
    """                         
    for i in range(len(states[0,0,:])-1):
        for j in range(len(states)):
            if np.vdot(states[:,j,i],states[:,j,i+1]) < 0.0:
                states[:,j,i+1] = -1*states[:,j,i+1]
    return states


def diagonalization(H_ryd,H_hfs,H_U,A_matrix,R):
    """
    Diagonalizes the Hamiltonian for the Rydberg molecule.
    Parameters:
        H_ryd (np.ndarray): Rydberg Hamiltonian matrix.
        H_hfs (np.ndarray): Hyperfine structure Hamiltonian matrix.
        H_U (np.ndarray): Scattering Hamiltonian matrix.
        A_matrix (np.ndarray): Transformation matrix from ground-state to Rydberg basis.
        R (np.ndarray): Radial grid points.
    Returns:    
        energy (np.ndarray): Eigenvalues of the Hamiltonian.
        states (np.ndarray): Eigenstates of the Hamiltonian.
    """
    
    energy=np.zeros([*H_ryd[0].shape, R.size])   
    states=np.zeros([*H_ryd.shape, R.size])
    
    for i,r in enumerate(R):
        H_V = np.dot(np.dot(A_matrix[:,:,i],H_U[:,:,i]),np.conjugate(A_matrix[:,:,i].transpose()))
        H = H_ryd + H_V + H_hfs 
        w,v = np.linalg.eigh(H)
        energy[:,i] = w.copy()
        states[:,:,i] = v.copy()
        
        #print(i,r)
        
        
    states = phase_smoothening(states)
    return energy,states





def phase_interpolation(phases):
    """
    Input phase-shift date are interpolated to form usable functions for H_U.
    Parameters:
        phases (np.ndarray): Array of phase shifts with each row [k, 1S0, 3S1, 1P1, 3P0, 3P1, 3P2].
    Returns:
        list: List of interpolated phase shift functions.
    """
    phase_shifts = []
    for i in range(1,phases[0].size):
        phase_shifts.append(interp1d(phases[:,0],phases[:,i],kind="cubic"))

    return phase_shifts



@np.vectorize
def kval(n,r,k_cut=0):
    """
    Calculates the semiclassical momentum of the Rydberg electron.
    Parameters:
        n (int or np.ndarray): Principal quantum number.
        r (float or np.ndarray): Distance from the Rydberg ionic core.
        k_cut (float): Minimum value for k below which it is forced to be zero, default is 0.

    Returns:        
        float or np.ndarray: The semiclassical wave number k.
    """

    k= np.emath.sqrt(2 * (1/r - 1/(2*n**2))).real
    if k < k_cut:
        k =k_cut
    return k


#Spherical harmonics Y_lm(theta,phi)
def Spher(l,m,theta,phi):
    """
    Calculates spherical harmonics Y_lm(theta,phi) for given angles.
    Parameters:
        l (int): Orbital quantum number.
        m (int): Magnetic quantum number.
        theta (np.ndarray): Polar angle in radians.
        phi (np.ndarray): Azimuthal angle in radians.
    Returns:    
        np.ndarray: Spherical harmonics evaluated at the specified angles.
    """
    c = legendre(m,l,np.cos(theta)) * np.exp(1j * m * phi)
    t = c * np.emath.sqrt(gamma(l-m+1)*(2*l+1)/(gamma(l+m+1)*4.0*np.pi))
    return t




    
def anion_wf(R,Rin, l=1.35094, r0 = 9.004786, n=0):
    """
    Calculates the electronic wavefunction for an Alkali anion
    Parameters:
        R (np.ndarray): Radial grid points.
        Rin (float): Anion position with respect to Rydberg core.
    """
    x = np.abs(R-Rin)
    l=1.35094
    r0 = 9.004786
    n=0
    z= np.sinh(x/r0)
    F12 = sp.special.hyp2f1(-n,-2*l + n +1,1.5,-z**2)
    wf = (np.cosh(x/r0))**(-2*l)*z*F12
    return wf