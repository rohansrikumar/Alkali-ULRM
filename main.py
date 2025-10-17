from src import *
from arc import *
import matplotlib.pyplot as plt


def main():
    
    print("Initializing variables, loading parameters")

    #Load parameters for Rubidium from ARC package
    #Rydberg atom
    R_atom = Rubidium87()

    #Ground state atom
    GSatom = Rubidium87()

    phase = np.loadtxt('./data/phase.3016')
    #phase.1002 for markus, 1001 for Frederic, 3001 for Fabrikant etc.
    phase_shifts = phase_interpolation(phase)

    #Defining the relevant n,E windows. n0 under investigation
    n_min,n0,n_max=17,18,19
    E_min,E_max = R_atom.getEnergy(n_min+1, 2, 1.5)/eV, R_atom.getEnergy(n_max,n_max-1,n_max-0.5)/eV #eV -> a.u.

    #R_out should be large enough to calculate accurate Rydberg wavefunction
    #using numerov scheme
    R_in,R_out = 250, 2.6 * n0 * n0  
    R = np.linspace(R_in,R_out,200)
    """R1 = np.linspace(R_in,319,50)
    R2 = np.linspace(320,325,50)
    R3 = np.linspace(326,345,50)
    R4 = np.linspace(346,351,50)
    R5 = np.linspace(352,382,50)
    R6 = np.linspace(383,388,50)
    R7 = np.linspace(389,419,50)
    R8 = np.linspace(420,425,50)
    R9 = np.linspace(426,460,50)
    R10 = np.linspace(461,467,50)
    R11 = np.linspace(468,R_out,100)
    R = np.concatenate((R1,R2,R3,R4,R5,R6,R7,R8,R9,R10,R11))"""

    #electron k in n0 manifold,all k < k_cut assigned 0
    k_cut = 0.005
    k =np.array([kval(n0,r,k_cut) for r in R])
    
    #Omega = mj + ms2 + mi is only good quantum number
    Omega = 1.5
    
    print("Calculating R-independent terms; Rydberg and Ground-state atom")
    #|alpha> basis centered around the rydberg core
    alpha = RydbergBasis(R_atom, n_range = (n_min,n_max), E_range = (E_min,E_max),\
                          Omega  =Omega,GSatom = GSatom).get_basis_states()
    
    #|beta> basis centered around the perturber core
    beta = ScatteringBasis(Omega,GSatom,L_list=1,S_list=0).get_basis_states()

    H_U = H_Scattering(beta,k,phase_shifts)
    H_ryd = H_Rydberg(alpha,R_atom)
    H_hfs = H_Hyperfine(alpha,GSatom)

    A_ab = A_transform(alpha,beta,R)
    print(f"Frame transformation calculated, size: {A_ab.nbytes/1024**2} MB")

    A,B = GSatom.getHFSCoefficients(5,0,1/2)
    Ezero = GSatom.getHFSEnergyShift(0.5,1,A/10**9)
    Eryd = R_atom.getEnergy(n0,3,2.5)/eV #eV -> a.u.

    energy,states = diagonalization(H_ryd,H_hfs,H_U,A_ab,R)
    
    energy = GHz* (energy - Eryd) - Ezero

    print(states.shape)

    #rates = photo_excitation(alpha, states, R_atom, (5, 2, 2.5), F=1)

    initial_state = {'n':5,'l':2,'j':2.5,'mj':0.5,'F':1,'MF':0}
    fig,ax = plt.subplots()
    for j in range(230,290):
        rates = np.zeros(len(R))
        for mj,MF in product(np.arange(-2.5,3.5,1),np.arange(-1,2,1)):
            initial_state = {'n':5,'l':2,'j':2.5, 'mj':mj, 'F':1, 'MF':MF}
            rates += photo_excitation(alpha, states[:,j,:], R_atom, initial_state)
            
        ax.scatter(R, energy[j],cmap='viridis',c=rates,s=0.8)
        print(np.max(rates))

    ax.set_xlabel('R (a.u.)')
    ax.set_ylabel('Energy (a.u.)')
    ax.set_ylim(-45, 40)  # Set y-axis limits
    ax.set_xlim(250, 800)  # Set y-axis limits


    plt.savefig("pec.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()

