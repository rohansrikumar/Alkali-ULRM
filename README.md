# Alkali-ULRM
The Alkali-ULRM package calculates the electronic structure of Ultralong-range Rydberg molecules, composed of an Alkali Rydberg atom bound to a ground state atom via quantum scattering. The package provides modular, object oriented python code that allows for the calculation of electronic structure, spin couplings and radiative properties of these exotic molecules via Hamiltonian modelling (see reference for more details).

The class structure and data flow is illustrated below.

The routines depend on the independent ARC (Alkali.ne Rydberg Calculator) package for necessary data and attributes of the constituent Rydberg atom.
The mathematical modelling the molecule, and the associated Hamiltonian is obtained from the reference Matt Eiles et.al.
