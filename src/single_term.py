import numpy as np
import scipy as sp
from scipy.linalg import expm, norm
from scipy.linalg import cosm, sinm

# Define the measurement operators
def mbdite_operators(epsilon, H):
    M0 = (cosm(epsilon * H) - sinm(epsilon * H)) / np.sqrt(2)
    M1 = (cosm(epsilon * H) + sinm(epsilon * H)) / np.sqrt(2)
    assert np.allclose(M0.T.conj() @ M0 + M1.T.conj() @ M1, np.eye(H.shape[0]))
    return M0, M1

def mbdite_algorithm(psi0, H, epsilon, E_th, num_measurements, U_C, seed=0):
    if seed >= 0:
        np.random.seed(seed)
    d = H.shape[0]
    L, U = np.linalg.eigh(H) # Energy eigenbasis
    ground_state = U[:,0]
    
    psi = psi0.copy()
    k0, k1 = 0, 0
    energy_history = [psi.conj().T @ H @ psi]
    fidelity_history = [np.abs(np.vdot(ground_state, psi))**2]
    x_max_history = [np.nan]
    M0, M1 = mbdite_operators(epsilon, H)
    
    for t in range(num_measurements):
        
        psi /= np.linalg.norm(psi)
        
        # Calculate probabilities
        p0 = np.vdot(psi, M0.T.conj() @ M0 @ psi).real
        p1 = np.vdot(psi, M1.T.conj() @ M1 @ psi).real
        p0, p1 = p0/(p0+p1), p1/(p0+p1)
        
        # Perform measurement
        outcome = np.random.choice([0, 1], p=[p0, p1])
        
        # Calculate hypothetical new counts
        new_k0 = k0 + (1 if outcome == 0 else 0)
        new_k1 = k1 + (1 if outcome == 1 else 0)
        
        # Calculate x_max for hypothetical state (handle division by zero)
        if (new_k0 + new_k1) > 0:
            x_max_hypothetical = 0.5 * np.arcsin((new_k1 - new_k0) / (new_k0 + new_k1))
        else:
            x_max_hypothetical = 0
        
        # Convert x_max to energy estimate: E_estimate = x_max / epsilon
        E_estimate = x_max_hypothetical / epsilon if epsilon != 0 else 0
        
        # Check threshold condition (compare energy estimate with E_th)
        if E_estimate < E_th:
            # Accept measurement: update counts and state
            k0, k1 = new_k0, new_k1
            psi = M0 @ psi if outcome == 0 else M1 @ psi
        else:
            # Reject: apply U_C and reset counts
            psi = U_C @ psi
            k0, k1 = 0, 0
        
        psi /= np.linalg.norm(psi)
        
        # Store fidelity with ground state (|1⟩ for H = Z)
        fidelity_history.append(np.abs(np.vdot(ground_state, psi))**2)

        # Energy
        energy_history.append(psi.conj().T @ H @ psi)
        
        # Initialize previous_x_max
        previous_x_max = 0

        # Inside the measurement loop:
        if (k0 + k1) > 0:
            current_x_max = 0.5 * np.arcsin((k1 - k0) / (k0 + k1))
            previous_x_max = current_x_max  # Store the last valid value
        else:
            current_x_max = previous_x_max  # Use last valid value when counts are zero
        x_max_history.append(current_x_max)
    
    return energy_history, fidelity_history, x_max_history, psi

# Function that does many steps
def hite_algorithm(psi, H, beta, max_iter, adaptive=True, force_fail=False, verbose=False, seed=0):
    if seed >= 0:
        np.random.seed(seed)
    d = H.shape[0]
    L, U = np.linalg.eigh(H) # Energy eigenbasis
    GS = U[:,0]

    # Shift spectrum of H to be non-negative with GS energy being 0
    H_diag = np.diag(L)
    H_diag -= np.min(L) * np.eye(d)

    if verbose:
        print(f"Shifted Spectrum: {np.diag(H_diag)}.")
    assert np.isclose(GS.conj().T @ H @ GS, np.min(L))
    
    # Make gamma = e^{-beta H_diag)
    gamma = sp.linalg.expm(-beta * H_diag)
    U_gamma = np.zeros((2*d,2*d), dtype=gamma.dtype)
    U_gamma[:d,:d] = gamma
    U_gamma[d:,:d] = U_gamma[:d,d:] = np.sqrt(np.eye(d) - gamma**2)
    U_gamma[d:,d:] = -gamma
    assert np.isclose(np.linalg.norm(U_gamma @ U_gamma.conj().T - np.eye(2*d)), 0.0)
    
    psi = U.conj().T @ psi # Move to energy eigenbasis
    GS_E = U.conj().T @ GS # GS in energy eigenbasis. Should just be [1,0,0,...]
    assert np.isclose(GS_E[0], 1)
    
    Es = [psi.conj().T @ H_diag @ psi] # Initial energy
    Fs = [np.abs(psi[0])**2]
    if verbose:
        print(f"Energy at iteration {0}: {Es[-1]}.")
        print(f"Fidelity at iteration {0}: {Fs[-1]}.")
        
    mos = []
    fail_count = 0
    def permutation(fail_count, d):
        # Define one particular choice of permutation matrix
        P = np.eye(d)
        if fail_count == 0:
            P = np.roll(np.eye(d), 1, axis=1)
        else:
            P[:-fail_count,:-fail_count] = np.roll(P[:-fail_count,:-fail_count], 1, axis=1)
        return P
    
    for i in range(1, max_iter+1):
        psi_E = np.kron(np.array([1,0]), psi)
        psi_E = U_gamma @ psi_E
        p0 = np.linalg.norm(psi_E[:d])**2
        p1 = np.linalg.norm(psi_E[d:])**2
        assert np.isclose(p0 + p1, 1)

        # Choose measurement outcome according to probabilities
        mo = np.random.choice([0,1], size=1, p=[p0,p1])[0]
        if p1 > 0 and force_fail:
            mo = 1
        mos.append(mo)
        if verbose:
            print(f"Measurement outcome and probability at iteration {i}: {mo}, {[p0,p1][mo]}.")
        
        if mo == 0:
            psi = psi_E[:d]
        else:
            psi = psi_E[d:]
        psi /= np.linalg.norm(psi)

        if mo == 1 and adaptive:
            # Made measurement error; try to fix
            nP = permutation(0, d)
            assert np.isclose(np.linalg.norm(nP @ nP.conj().T - np.eye(d)), 0.0)
            psi = nP @ psi
            fail_count += 1
            if verbose:
                print(f"-------fail counter: {fail_count}; iteration: {i}-------")
        Es.append(psi.conj().T @ H_diag @ psi)
        Fs.append(np.abs(psi[0])**2)
        if verbose:
            print(f"Energy at iteration {i}: {Es[-1]}.")
            print(f"Fidelity at iteration {i}: {Fs[-1]}.")
        #if len(Es) > 1 and np.abs(Es[-1] - Es[-2]) < 1.e-8:
        #    break
    if np.isclose(Es[-1], 0):
        assert np.isclose(np.abs(GS.conj().T @ (U @ psi)), 1.0)
        assert np.isclose((U @ psi).conj().T @ H @ (U @ psi), np.min(L))
    return Es, Fs, mos, U @ psi