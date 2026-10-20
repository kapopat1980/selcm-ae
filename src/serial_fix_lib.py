import numpy as np, os, hashlib, json
from scipy.special import gammaincc
from scipy.stats import chisquare
import nist
def serial2(e, m=16):
    p0, p1, p2 = nist._psi(e, m), nist._psi(e, m - 1), nist._psi(e, m - 2)
    return gammaincc(2 ** (m - 2), (p0 - p1) / 2), gammaincc(2 ** (m - 3), (p0 - 2 * p1 + p2) / 2)
