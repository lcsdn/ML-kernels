import numpy as np
from time import time

class Kernel:
    """General class of kernels."""
    @staticmethod
    def _precomputations(set1, set2, same_sets):
        return set1, set2
    
    def _lazy_kernel(self, precomputations, i, j):
        set1, set2 = precomputations
        return self(set1[i], set2[j])
    
    def _pairwise_matrix(self, set1, set2, same_sets):
        precomputations = self._precomputations(set1, set2, same_sets)
        n1, n2 = len(set1), len(set2)
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                # If the sets are  identical, then the matrix is symmetric.
                if i > j and same_sets:
                    K[i, j] = K[j, i]
                else:
                    K[i, j] = self._lazy_kernel(precomputations, i, j)
        return K

    def pairwise_matrix(self, set1, set2=None, verbose=True):
        """
        Compute the matrix of pairwise kernel evaluations for two sets of points.
        If set2 is not specified, then the two sets are set to be identical
        and the resulting matrix is symmetric.
        If verbose is set to True, display the computation time.
        """
        if verbose:
            t0 = time()
        same_sets = set2 is None
        if same_sets:
            set2 = set1
        
        K = self._pairwise_matrix(set1, set2, same_sets)

        if verbose:
            print("Pairwise matrix computation time:", time() - t0)
        return K

## Basic kernels
class LinearKernel(Kernel):
    @staticmethod
    def __call__(x, y):
        if isinstance(x, (int, float)):
            return x*y
        return x.dot(y)
    
    @staticmethod
    def _pairwise_matrix(X1, X2, same_sets):
        if isinstance(X1[0], (int, float)):
            return np.outer(X1, X2)
        return X1 @ X2.T

class RBFKernel(Kernel):
    def __init__(self, radius=1):
        self.radius = radius
    
    def __call__(self, x, y):
        return np.exp(- self.radius * np.linalg.norm(x-y)**2)

class CosineKernel(Kernel):
    def __init__(self, radius=1):
        self.radius = radius
    
    def __call__(self, x, y):
        return np.cos(self.radius * (x-y))

class SequenceDotProductKernel(Kernel):
    """Compute the number of identical characters in a sequence."""
    @staticmethod
    def __call__(seq1, seq2):
        n = len(seq1)
        dotproduct = 0
        for i in range(n):
            if seq1[i] == seq2[i]:
                dotproduct += 1
        return dotproduct
    
## Spectrum kernel
def sparse_product(d1, d2):
    """Compute the dot product of two sparse vectors represented as dictionaries."""
    dotproduct = 0
    for key in d1:
        if key in d2:
            dotproduct += d1[key] * d2[key]
    return dotproduct

def compute_spectrum(sequence, size):
    """For a given sequence, compute a dictionary containing the number of times
    each subsequence of the given size appears in the sequence."""
    spectrum = {}
    n = len(sequence)
    for i in range(n - size + 1):
        subseq = sequence[i:i+size]
        if subseq in spectrum:
            spectrum[subseq] += 1
        else:
            spectrum[subseq] = 1
    return spectrum

class SpectrumKernel(Kernel):
    """Represent sequences as sparse vectors counting the number the number of times
    each subsequence of the given size appears in the sequence, and compute their
    dot product."""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, seq1, seq2):
        sp1 = compute_spectrum(seq1, self.size)
        sp2 = compute_spectrum(seq2, self.size)
        return sparse_product(sp1, sp2)
    
    def _precomputations(self, set1, set2, same_sets):
        spectrums1 = [compute_spectrum(seq, self.size) for seq in set1]
        if same_sets:
            spectrums2 = spectrums1
        else:
            spectrums2 = [compute_spectrum(seq, self.size) for seq in set2]
        return spectrums1, spectrums2
    
    @staticmethod
    def _lazy_kernel(precomputations, i, j):
        spectrums1, spectrums2 = precomputations
        return sparse_product(spectrums1[i], spectrums2[j])

## Local sequence similarity
class LocalSimilarity(Kernel):
    """
    Count the number of identical subsequences at the same position.
    Sequences are assumed to be of identical length.
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, seq1, seq2):
        n = len(seq1)
        output = 0
        current_streak = 0
        for i in range(n - self.size + 1):
            if seq1[i] == seq2[i]:
                current_streak += 1
            else:
                current_streak = 0
            if current_streak >= self.size:
                output += 1
        return output

## Substring kernel
class _BWrapper:
    def __init__(self, dim, decay, seq1, seq2, parent=None):
        self.array = - np.ones((dim, dim))
        self.decay = decay
        self.parent = parent
        self.seq1 = seq1
        self.seq2 = seq2
    
    def __call__(self, i, j):
        if i == -1 or j == -1:
            return 0
        
        if self.array[i, j] != -1:
            return self.array[i, j]
        
        value = self(i-1, j) + self(i, j-1) - self.decay * self(i-1, j-1)
        value *= self.decay
        if self.seq1[i] == self.seq2[j]:
            if self.parent is None:
                value += self.decay ** 2 # B_0 is constant 1
            else:
                value += self.decay ** 2 * self.parent(i-1, j-1)
        self.array[i, j] = value
        return value


class _KWrapper:
    def __init__(self, B, decay, seq1, seq2):
        self.B = B
        self.decay = decay
        self.seq1 = seq1
        self.seq2 = seq2
    
    def __call__(self, i, j):
        if i == -1 or j == -1:
            return 0
        
        value = 0
        for l, character in enumerate(self.seq2[:j+1]):
            if character == self.seq1[i]:
                value += self.B(i-1, l-1)
        value *= self.decay ** 2
        value += self(i-1, j)
        return value

class SubstringKernel(Kernel):
    """
    Count the number of identical subsequences of given size.
    Sequences are assumed to be of identical length.
    """
    def __init__(self, size, decay=0.9):
        self.size = size
        self.decay = decay
        
    def init_B(self, seq1, seq2):
        n = len(seq1)
        B = None
        for k in range(1, self.size):
            B = _BWrapper(n - self.size + k, self.decay, seq1, seq2, parent=B)
        return B
    
    def __call__(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        n = len(seq1)
        B = self.init_B(seq1, seq2)
        K = _KWrapper(B, self.decay, seq1, seq2)
        return K(n-1, n-1)

## Sum Kernel
class SumKernel(Kernel):
    """Compute the sum of given kernels."""
    def __init__(self, *kernels):
        self.kernels = kernels
    
    def __call__(self, x, y):
        value = 0
        for kernel in self.kernels:
            value += kernel(x, y)
        return value
         
    def _pairwise_matrix(self, set1, set2, same_sets):
        n1, n2 = len(set1), len(set2)
        K = np.zeros((n1, n2))
        for kernel in self.kernels:
            K += kernel._pairwise_matrix(set1, set2, same_sets)
        return K