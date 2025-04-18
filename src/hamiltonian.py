import numpy as np
import scipy as sp

# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
Id = np.eye(2)

def random_dense(d):
    H = np.random.rand(d,d) + 1.j * np.random.rand(d,d)
    return H, _

def kron(ops):
    op = 1
    for o in ops:
        op = np.kron(op, o)
    return op

def TFI(L, Jz, hx, hz):
    """
    H = sum_{i=1}^{i=L} J_z Z_i Z_{i+1} + h_x X_i + h_z Z_i
    Site L + 1 == Site 1, so periodic boundary conditions
    """
    # Build full 2^L x 2^L Hamiltonian
    # Also return translationally invariant 2 site term as a matrix.
    
    H = np.zeros((2**L, 2**L), dtype=np.float64)

    for i in range(L):
        H += hx * kron([Id]*i + [X] + [Id]*(L-i-1))
        H += hz * kron([Id]*i + [Z] + [Id]*(L-i-1))

    Ising = [Z, Z] + [Id] * (L-2)
    for i in range(L):
        H += Jz * kron(Ising)
        # Shift the first element to the last; periodic BCs!
        Ising.append(Ising.pop(0))
    return H, Jz * np.kron(Z, Z) + hx/2 * (np.kron(X, Id) + np.kron(Id, X)) + hz/2 * (np.kron(Z, Id) + np.kron(Id, Z))
    
def cluster_state(L):
    """
    H = sum_{i=1}^{i=L} -Z_i X_{i+1} Z_{i+2}
    Site L + 1 == Site 1 and L + 2 == Site 2, so periodic boundary conditions
    """
    # Build full 2^L x 2^L Hamiltonian

    H = np.zeros((2**L, 2**L), dtype=np.float64)

    Cluster = [Z, X, Z] + [Id] * (L-3)
    for i in range(L):
        H += -1 * kron(Cluster)
        # Shift the first element to the last; periodic BCs!
        Cluster.append(Cluster.pop(0))
    return H, -1 * kron([Z, X, Z])

def AKLT(L):
    X = 1/np.sqrt(2) * np.array([[0,1,0],[1,0,1],[0,1,0]])
    Y = 1/np.sqrt(2)/1.j * np.array([[0,1,0],[-1,0,1],[0,-1,0]])
    Z = np.array([[1,0,0],[0,0,0],[0,0,-1]])

    H = np.zeros((3**L, 3**L), dtype=np.float64)
    
    Xs = [X,X] + [Id] * (L-2)
    Ys = [Y,Y] + [Id] * (L-2)
    Zs = [Z,Z] + [Id] * (L-2)

    for i in range(L):
        SS = kron(Xs) + kron(Ys) + kron(Zs)
        H += 1/2 * SS + 1/6 * SS @ SS + 1/3 * np.eye(3**L)
        # Shift the first element to the last; periodic BCs!
        Xs.append(Xs.pop(0))
        Ys.append(Ys.pop(0))
        Zs.append(Zs.pop(0))

    SS = np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z)
    h = 1/2 * SS + 1/6 * SS @ SS + 1/3 * np.eye(3**2)
    assert np.isclose(np.linalg.norm(h - h.conj().T), 0.0)
    L = np.linalg.eigh(h)[0]
    assert np.isclose(np.sum(L), 5)
    assert np.isclose(np.sum(L**2), 5)
    return H, h