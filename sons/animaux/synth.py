"""Synthèse de 12 cris d'animaux (source + filtres de formants). Sortie : out/S01.wav …"""
import numpy as np, os, sys
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

FS = 44100
rng = np.random.default_rng(7)

def t_(d): return np.arange(int(FS * d)) / FS

def contour(points, n):
    """points = [(position 0..1, valeur), …] -> courbe interpolée de n échantillons"""
    xs, ys = zip(*points)
    return np.interp(np.linspace(0, 1, n), xs, ys)

def source(f0, rough=0.0, shape=1.0):
    """train d'impulsions glottiques (dent de scie adoucie) suivant f0 (tableau)"""
    jitter = 1 + rough * 0.03 * rng.standard_normal(len(f0))
    phase = np.cumsum(f0 * jitter) / FS
    saw = 2 * (phase % 1) - 1
    s = np.sign(saw) * np.abs(saw) ** shape
    return s - np.mean(s)

def resonateurs(x, formants, n):
    """formants = liste de (contourF, contourBW, gain) ; filtres variables dans le temps"""
    out = np.zeros(n)
    for F, BW, g in formants:
        F = F if np.ndim(F) else np.full(n, F)
        BW = BW if np.ndim(BW) else np.full(n, BW)
        r = np.exp(-np.pi * BW / FS)
        b1 = 2 * r * np.cos(2 * np.pi * F / FS)
        b2 = -r * r
        a0 = (1 - r) * 2
        y = np.zeros(n); y1 = y2 = 0.0
        for i in range(n):
            v = a0[i] * x[i] + b1[i] * y1 + b2[i] * y2
            y[i] = v; y2 = y1; y1 = v
        out += g * y
    return out

def env(n, attaque, relache, forme=None):
    e = np.ones(n)
    a = max(1, int(attaque * FS)); r = max(1, int(relache * FS))
    e[:a] = np.linspace(0, 1, a)
    e[-r:] *= np.linspace(1, 0, r) ** 1.5
    if forme is not None: e *= forme
    return e

def bruit(n): return rng.standard_normal(n)

def bande(x, lo, hi, ordre=4):
    return sosfilt(butter(ordre, [lo, hi], btype='band', fs=FS, output='sos'), x)

def passe_haut(x, f):
    return sosfilt(butter(2, f, btype='high', fs=FS, output='sos'), x)

def norm(x, pic=0.89):
    x = passe_haut(x, 60)
    return x / (np.max(np.abs(x)) + 1e-9) * pic

def silence(d): return np.zeros(int(FS * d))

# ---------------------------------------------------------------- animaux
def vache():
    d = 0.85; n = len(t_(d))
    f0 = contour([(0, 120), (0.15, 128), (0.6, 112), (1, 92)], n) * (1 + 0.01 * np.sin(2*np.pi*5*t_(d)))
    src = source(f0, rough=0.6, shape=1.3)
    F1 = contour([(0, 260), (0.2, 420), (1, 380)], n)
    F2 = contour([(0, 900), (0.2, 820), (1, 760)], n)
    y = resonateurs(src, [(F1, 90, 1.0), (F2, 120, 0.6), (2400, 250, 0.12)], n)
    y += 0.04 * bande(bruit(n), 300, 1500) * contour([(0,1),(1,1)], n)
    return norm(y * env(n, 0.12, 0.25))

def mouton():
    d = 0.8; n = len(t_(d)); t = t_(d)
    trem = 1 + 0.06 * np.sin(2 * np.pi * 7 * t)
    f0 = contour([(0, 240), (0.2, 270), (1, 230)], n) * trem
    src = source(f0, rough=0.8)
    y = resonateurs(src, [(600, 90, 1.0), (1750, 130, 0.7), (2600, 200, 0.3)], n)
    am = 0.65 + 0.35 * np.sin(2 * np.pi * 7 * t)
    return norm(y * am * env(n, 0.05, 0.2))

def chat():
    d = 0.8; n = len(t_(d))
    f0 = contour([(0, 520), (0.35, 720), (0.75, 600), (1, 430)], n)
    src = source(f0, rough=0.3)
    F1 = contour([(0, 320), (0.35, 850), (0.7, 700), (1, 380)], n)
    F2 = contour([(0, 2300), (0.35, 1500), (0.7, 1100), (1, 800)], n)
    y = resonateurs(src, [(F1, 110, 1.0), (F2, 160, 0.7), (3200, 300, 0.15)], n)
    return norm(y * env(n, 0.06, 0.22))

def aboiement(d, f0a, f0b):
    n = len(t_(d))
    f0 = contour([(0, f0a), (0.3, f0a * 1.05), (1, f0b)], n)
    src = source(f0, rough=2.0) + 0.6 * bruit(n)
    F1 = contour([(0, 750), (1, 450)], n); F2 = contour([(0, 1300), (1, 900)], n)
    y = resonateurs(src, [(F1, 160, 1.0), (F2, 200, 0.7), (2600, 400, 0.25)], n)
    return y * env(n, 0.008, d * 0.7)

def chien():
    return norm(np.concatenate([aboiement(0.2, 420, 260), silence(0.12), aboiement(0.22, 400, 240)]))

def quack(d, f):
    n = len(t_(d))
    f0 = contour([(0, f), (0.4, f * 1.08), (1, f * 0.8)], n)
    src = np.sign(source(f0, rough=1.5)) * 0.8 + 0.2 * bruit(n)
    y = resonateurs(src, [(1000, 180, 1.0), (1700, 220, 0.8), (2900, 350, 0.3)], n)
    return y * env(n, 0.01, d * 0.5)

def canard():
    return norm(np.concatenate([quack(0.17, 300), silence(0.08), quack(0.17, 290), silence(0.08), quack(0.2, 270)]))

def note_douce(d, f, souffle=0.05, gliss=0.0):
    n = len(t_(d)); t = t_(d)
    fr = f * (1 + gliss * t / d)
    ph = 2 * np.pi * np.cumsum(fr) / FS
    y = np.sin(ph) + 0.12 * np.sin(2 * ph) + 0.04 * np.sin(3 * ph)
    y += souffle * bande(bruit(n), f * 0.8, f * 3)
    return y * env(n, 0.03, d * 0.45)

def coucou():
    return norm(np.concatenate([note_douce(0.28, 680, gliss=-0.02), silence(0.08), note_douce(0.4, 560, gliss=-0.03)]))

def hibou():
    return norm(np.concatenate([note_douce(0.18, 400, 0.15), silence(0.12), note_douce(0.45, 390, 0.15, -0.05) * (1 + 0.15 * np.sin(2*np.pi*9*t_(0.45)))]))

def chirp(d, fa, fb, amp=1.0):
    n = len(t_(d))
    f = contour([(0, fa), (1, fb)], n)
    ph = 2 * np.pi * np.cumsum(f) / FS
    return amp * (np.sin(ph) + 0.1 * np.sin(2 * ph)) * env(n, 0.005, d * 0.6)

def poussin():
    p = [chirp(0.13, 4200, 2800), silence(0.09), chirp(0.13, 4300, 2900), silence(0.09), chirp(0.15, 4100, 2700)]
    return norm(np.concatenate(p))

def oiseau():  # cui-cui : montées rapides et trilles
    p = []
    for k in range(5):
        p += [chirp(0.06, 2600, 4800, 0.9), silence(0.035)]
    p += [chirp(0.18, 5200, 3600)]
    return norm(np.concatenate(p))

def grogne(d):
    n = len(t_(d)); t = t_(d)
    f0 = contour([(0, 95), (0.5, 120), (1, 85)], n)
    src = source(f0, rough=3.0, shape=0.7) + 0.9 * bruit(n)
    am = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 26 * t)) * 0.8
    y = resonateurs(src * am, [(450, 160, 1.0), (1100, 220, 0.6), (2300, 300, 0.2)], n)
    return y * env(n, 0.02, d * 0.4)

def cochon():
    return norm(np.concatenate([grogne(0.28), silence(0.1), grogne(0.34)]))

def croa(d):
    n = len(t_(d)); t = t_(d)
    pulses = np.zeros(n)
    per = int(FS / 38)
    for i in range(0, n, per):
        k = min(per, n - i)
        tt = np.arange(k) / FS
        pulses[i:i+k] += np.sin(2 * np.pi * 180 * tt) * np.exp(-tt * 260)
    src = pulses + 0.15 * bruit(n)
    y = resonateurs(src, [(700, 150, 1.0), (1300, 200, 0.6)], n)
    return y * env(n, 0.01, d * 0.3)

def grenouille():
    return norm(np.concatenate([croa(0.3), silence(0.1), croa(0.3)]))

def loup():
    d = 0.88; n = len(t_(d)); t = t_(d)
    f0 = contour([(0, 330), (0.3, 520), (0.7, 560), (1, 420)], n) * (1 + 0.012 * np.sin(2*np.pi*5.5*t))
    src = source(f0, rough=0.4, shape=1.6)
    F1 = contour([(0, 650), (0.4, 450), (1, 340)], n)
    F2 = contour([(0, 1100), (0.4, 900), (1, 750)], n)
    y = resonateurs(src, [(F1, 120, 1.0), (F2, 160, 0.4)], n)
    y += 0.05 * bande(bruit(n), 500, 2500)
    return norm(y * env(n, 0.15, 0.3))

ANIMAUX = [
    ('S01', 'Vache', vache), ('S02', 'Mouton', mouton), ('S03', 'Chat', chat),
    ('S04', 'Chien', chien), ('S05', 'Canard', canard), ('S06', 'Coucou', coucou),
    ('S07', 'Poussin', poussin), ('S08', 'Cochon', cochon), ('S09', 'Loup', loup),
    ('S10', 'Grenouille', grenouille), ('S11', 'Hibou', hibou), ('S12', 'Oiseau', oiseau),
]

if __name__ == '__main__':
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for code, nom, f in ANIMAUX:
        y = f()
        wavfile.write(f'{out}/{code}.wav', FS, (y * 32767).astype(np.int16))
        print(code, nom, round(len(y) / FS, 2), 's')
