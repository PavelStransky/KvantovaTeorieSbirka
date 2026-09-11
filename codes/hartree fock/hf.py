import numpy as np
import csv

class HartreeFock:
    """
    Class for solving the Hartree Fock equation for simple atoms.
    """
    def __init__(this, Z, maxN):
        """
        Z -- number of particles in the system 
        maxN -- maximum quantum number
        """

        this.basis = []
        for n in range(maxN):
            this.basis.append([n + 1, -1])
            this.basis.append([n + 1, 1])
        """
        basis -- the single-particle basis of the system in the form [a, s],
                 where a is the principal quantum number and s is the spin projection
        """

        """ Reading precalculated matrix elements """
        this.radialIntegrals = np.zeros(shape=(maxN, maxN, maxN, maxN))
        with open(r"radial.txt") as csvFile:
            csvReader = csv.reader(csvFile, delimiter=",")

            for row in csvReader:
                this.radialIntegrals[int(row[0]) - 1,int(row[1]) - 1,int(row[2]) - 1,int(row[3]) - 1] = float(row[4])       

        this.Z = Z                      # number of particles in the system
        this.n = len(this.basis)        # number of single-particle basis states
        this.epsilon = np.zeros(this.n) # single-particle energies
        this.e0 = 0                     # ground-state energy

        this.C = np.identity(this.n)    # Matrix with coefficients
        this.h = np.zeros((this.n, this.n))

        print(f"Creating Hartree-Fock object for atom with Z = {Z}. Number of basis states: {this.n}.")

        this.FillHFMatrix()             # h matrix must always be pre-calculated

    def H0(this, a, c):
        """
        One-particle states
        """
        n1, s1 = this.basis[a]
        n2, s2 = this.basis[c]

        if n1 != n2 or s1 != s2:            # States are othogonal
            return 0
        else:
            return -this.Z**2 / (2 * n1**2) # Interaction electron - nucleon

    def Radial(this, n1, n2, n3, n4):
        """
        Radial integral <n1, n2|v|n3, n4>.
        """
        return this.Z * this.radialIntegrals[n1 - 1, n2 - 1, n3 - 1, n4 - 1]

    def VHF(this, a, b, c, d):
        """
        Returns the antisymmetrized two-body matrix element <ab|v|cd> (taking the spin states into account)
        """
        n1, s1 = this.basis[a]
        n2, s2 = this.basis[b]
        n3, s3 = this.basis[c]
        n4, s4 = this.basis[d]

        if s1 == s2 == s3 == s4:
            return this.Radial(n1, n2, n3, n4) - this.Radial(n2, n1, n3, n4)
        if s1 == s3 and s2 == s4:
            return this.Radial(n1, n2, n3, n4)
        if s1 == s4 and s2 == s3:
            return -this.Radial(n2, n1, n3, n4)
        else:
            return 0

    def FillHFMatrix(this):
        """
        Fill the h matrix
        """
        for a in range(this.n):
            for c in range(this.n):
                x = this.H0(a, c)
                for k in range(this.Z):
                    for b in range(this.n):
                        for d in range(this.n):
                            x += this.C[k, b] * this.C[k, d] * this.VHF(a, b, c, d)

                this.h[a, c] = x

    def GroundStateEnergy(this):
        """
        Calculates the ground state energy from the matrix C.
        """
        e = 0
        for k in range(this.Z):
            for a in range(this.n):
                for c in range(this.n):
                    e += this.C[k, a] * this.C[k, c] * this.H0(a, c)
                    for l in range(this.Z):
                        for b in range(this.n):
                            for d in range(this.n):
                                e += 0.5 * this.C[k, a] * this.C[l, b] * this.C[k, c] * this.C[l, d] * this.VHF(a, b, c, d)
        return e

    def Iteration(this):
        """
        Perform one Hartree-Fock iteration.
        Returns absolute error of the lowest single-particle energy.
        """
        # Find eigenvalues and eigenvector of HF matrix
        this.epsilon, this.C = np.linalg.eigh(this.h)
        this.C = this.C.T               # We must take the transpose
        
        this.FillHFMatrix()             # Fill the new HF matrix

    def Solve(this, tolerance = 1e-6, maxIterations = 100):
        iteration = 0
        e0previous = this.GroundStateEnergy()

        while iteration < maxIterations:
            iteration += 1

            this.Iteration()
            e0 = this.GroundStateEnergy()

            error = abs(e0 - e0previous)
            print(f"Interation: {iteration}, Ground-state energy: {e0}, Error = {error}.")

            if error < tolerance:
                print(f"Calculation converged after {iteration} iterations.")
                return e0

            e0previous = e0

        print(f"Calculation failed to converge in {iteration} iterations.")
        return

Z = 2           # Helium
Z = 4           # Beryllium
Z = 6           # Carbon

maxN = 4        # Basis size

for Z in range(2, 10, 2):
    hf = HartreeFock(Z, maxN)
    print(f"Ground-state energy: {hf.Solve()}")
