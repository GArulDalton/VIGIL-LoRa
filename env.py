"""VIGIL-LoRa simulation environment: batteryless energy-harvesting LoRa sensors,
imperfect SF orthogonality, capture effect, duty cycle, remote Kalman/AR(1) estimation."""
import numpy as np

SFS = np.arange(7, 13)
PWR_DBM = np.array([2.0, 8.0, 14.0])
PWR_mA = np.array([20.0, 28.0, 44.0])       # representative SX127x-class Tx currents
V_SUP = 3.3
SNR_TH = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0])  # demod floor per SF
NOISE_DBM = -174 + 10*np.log10(125e3) + 6.0                   # -117 dBm
# Goursaud-Gorce SIR thresholds M[sf_desired, sf_interferer] (dB)
SIR_M = np.array([[  6, -16, -18, -19, -19, -20],
                  [-24,   6, -20, -22, -22, -22],
                  [-27, -27,   6, -23, -25, -25],
                  [-30, -30, -30,   6, -26, -28],
                  [-33, -33, -33, -33,   6, -29],
                  [-36, -36, -36, -36, -36,   6]], dtype=float)
N_ACT = 1 + 6*3

def toa(sf, pl=20, bw=125e3, cr=1, npre=8, H=0, crc=1):
    de = 1 if sf >= 11 else 0
    ts = (2.0**sf)/bw
    tpre = (npre + 4.25)*ts
    num = 8*pl - 4*sf + 28 + 16*crc - 20*H
    npay = 8 + max(np.ceil(num/(4*(sf - 2*de)))*(cr + 4), 0)
    return tpre + npay*ts
TOA = np.array([toa(s) for s in SFS])

def act_decode(a):
    a = np.asarray(a); tx = a > 0
    sfi = np.where(tx, (a-1)//3, 0); pi = np.where(tx, (a-1) % 3, 0)
    return tx, sfi, pi

E_TX_TABLE = np.zeros(N_ACT)
for a in range(1, N_ACT):
    s, p = (a-1)//3, (a-1) % 3
    E_TX_TABLE[a] = V_SUP*PWR_mA[p]*1e-3*TOA[s] + V_SUP*11.5e-3*0.5*TOA[s] + 1e-3  # tx + ACK rx + MCU
AIR_TABLE = np.zeros(N_ACT)
for a in range(1, N_ACT):
    AIR_TABLE[a] = TOA[(a-1)//3]

class LoRaEHEnv:
    def __init__(self, N=200, G=1, R=5000.0, dt=120.0, seed=0, ideal_orth=False,
                 p_peak=0.3e-3, E_max=0.8, anomaly_rate_per_day=1.0,
                 kappa=0.5, beta=0.8, start_hour=0.0, shield=True, shield_margin=1.2):
        self.N, self.G, self.R, self.dt = N, G, R, dt
        self.rng = np.random.default_rng(seed)
        self.ideal = ideal_orth
        self.shield, self.sm = shield, shield_margin
        self.M = SIR_M.copy()
        if ideal_orth:
            self.M[~np.eye(6, dtype=bool)] = -1e9
        self.p_peak, self.E_max = p_peak, E_max
        self.E_min, self.E_on = 0.02*E_max, 0.10*E_max
        self.base = 5e-6*dt + 0.2e-3          # sleep + sensing per slot (J)
        self.kappa = np.ones(N)               # true/model ratio of baseline drain (1 = model correct)
        self.knoise = 0.0                     # per-night multiplicative drain noise (uniform +-)
        self.rng2 = np.random.default_rng(seed + 9999)
        self.kap_t = np.ones(N)
        self.kappa, self.beta = kappa, beta
        self.slots_day = int(86400/dt)
        self.an_p = anomaly_rate_per_day/self.slots_day
        self.rho, self.sw = 0.98, 0.15
        self.sig2 = self.sw**2/(1-self.rho**2)
        self.t = int(start_hour*3600/dt)
        self._place()
        self.reset_state()

    def _place(self):
        r = self.R*np.sqrt(self.rng.random(self.N)); th = 2*np.pi*self.rng.random(self.N)
        self.pos = np.c_[r*np.cos(th), r*np.sin(th)]
        if self.G == 1:
            gw = np.zeros((1, 2))
        else:
            ang = 2*np.pi*np.arange(self.G)/self.G
            gw = 0.5*self.R*np.c_[np.cos(ang), np.sin(ang)]
        self.gw = gw
        d = np.linalg.norm(self.pos[:, None, :]-gw[None], axis=2)/1000.0
        d = np.maximum(d, 0.05)
        shadow = self.rng.normal(0, 7.8, d.shape)
        self.PL = 128.95 + 23.2*np.log10(d) + shadow      # Petajajarvi et al. model
        self.bestPL = self.PL.min(1)
        # ADR: smallest SF with >=10 dB margin at 14 dBm, then reduce power if margin allows
        margin = 14 - self.bestPL[:, None] - NOISE_DBM - SNR_TH[None, :]    # N x 6
        ok = margin >= 10
        sfi = np.where(ok.any(1), ok.argmax(1), 5)
        m = margin[np.arange(self.N), sfi]
        pi = np.full(self.N, 2); pi[m >= 22] = 0; pi[(m >= 16) & (m < 22)] = 1
        self.adr_act = 1 + sfi*3 + pi
        self.eta = self.rng.uniform(0.4, 1.2, self.N)
        self.amp = self.rng.uniform(2, 5, self.N); self.phase = self.rng.normal(0, 0.3, self.N)
        self.mu = self.rng.uniform(20, 30, self.N)

    def reset_state(self):
        N = self.N
        self.E = self.rng.uniform(0.3, 1.0, N)*self.E_max
        self.alive = np.ones(N, bool)
        self.s = self.rng.normal(0, np.sqrt(self.sig2), N)
        self.shat = np.zeros(N); self.P = np.full(N, self.sig2)
        self.anom = np.zeros(N); self.an_left = np.zeros(N, int)
        self.bucket = np.full(N, 36.0)
        self.Z = np.zeros(N); self.Hbar = np.full(N, 1e-3)
        self.aoi = np.zeros(N); self.lastok = np.zeros(N)
        self.cloud = 0; self.last_sf_frac = np.zeros(6); self.harv = np.zeros(N)
        self.an_onset_pending = np.full(N, -1)  # slot index of undetected anomaly onset
        self.events = []  # (delay_slots or -1 for missed)

    # ----- helpers -----
    def hour(self):
        return (self.t % self.slots_day)*self.dt/3600.0

    def seasonal(self):
        h = self.hour()
        return self.mu + self.amp*np.sin(2*np.pi*(h-9)/24 + self.phase)

    def x_true(self):
        return self.seasonal() + self.s + self.anom

    def xhat(self):
        return self.seasonal() + self.shat

    def err2(self):
        return (self.x_true()-self.xhat())**2/self.sig2

    def solar(self):
        h = self.hour()
        clear = max(0.0, np.sin(np.pi*(h-6)/12)) if 6 < h < 18 else 0.0
        cf = [1.0, 0.5, 0.15][self.cloud]
        return self.p_peak*clear*cf

    def reserve(self):
        # energy needed to survive until next sunrise (06:30) once harvesting window is closing
        h = self.hour()
        if 6.5 <= h < 14:
            return self.E_min
        hrs = (30.5 - h) if h >= 14 else (6.5 - h)
        hrs_dark = hrs - max(0.0, 18 - h) if h >= 14 else hrs
        return self.E_min + self.sm*self.base*hrs_dark*3600/self.dt
    def mask(self):
        m = np.zeros((self.N, N_ACT), bool); m[:, 0] = True
        floor = self.reserve() if self.shield else self.E_min
        can = (self.E[:, None]-floor >= E_TX_TABLE[None, :]) & (self.bucket[:, None] >= AIR_TABLE[None, :])
        can &= self.alive[:, None]
        m[:, 1:] = can[:, 1:]
        return m

    def obs(self):
        h = self.hour()
        e2 = self.err2()
        o = np.stack([
            np.log1p(np.minimum(e2, 1e3)),
            self.E/self.E_max,
            np.minimum(self.Hbar/(self.p_peak*self.dt*0.5), 3.0),
            (14 - self.bestPL - NOISE_DBM + 7.5)/20.0,
            self.bucket/36.0,
            np.full(self.N, np.sin(2*np.pi*h/24)), np.full(self.N, np.cos(2*np.pi*h/24)),
            np.minimum(self.Z/(0.2*self.E_max), 5.0),
            np.log1p(self.aoi)/5.0,
            self.lastok,
            self.P/self.sig2,
            np.clip((self.E - self.reserve())/self.E_max, -1, 1),
        ], 1)
        o[~self.alive, 0] = 0.0
        return o.astype(np.float32)

    def gobs(self):
        g = np.r_[self.last_sf_frac*10, self.E.mean()/self.E_max, self.alive.mean()]
        return np.tile(g, (self.N, 1)).astype(np.float32)

    # ----- dynamics -----
    def _collide(self, idx, sfi, pwr):
        k = len(idx)
        if k == 0:
            return np.zeros(0, bool)
        air = TOA[sfi]
        st = self.rng.uniform(0, self.dt - air)
        en = st + air
        ov = (st[:, None] < en[None, :]) & (st[None, :] < en[:, None]); np.fill_diagonal(ov, False)
        fad = 10*np.log10(self.rng.exponential(1.0, (k, self.G)) + 1e-12)
        Prx = pwr[:, None] - self.PL[idx] + fad                     # k x G dBm
        ok_any = np.zeros(k, bool)
        lin = 10**(Prx/10)
        for g in range(self.G):
            snr_ok = Prx[:, g] - NOISE_DBM >= SNR_TH[sfi]
            ok = snr_ok.copy()
            for sfp in range(6):
                sel = (sfi == sfp)
                if not sel.any():
                    continue
                I = (ov[:, sel]*lin[sel, g][None, :]).sum(1)
                has = I > 0
                sir = np.where(has, Prx[:, g] - 10*np.log10(np.maximum(I, 1e-30)), 1e9)
                ok &= sir >= self.M[sfi, sfp]
            ok_any |= ok
        return ok_any

    def step(self, a):
        N = self.N
        a = np.asarray(a).copy()
        m = self.mask(); a[~m[np.arange(N), a]] = 0
        tx, sfi, pi = act_decode(a)
        idx = np.where(tx)[0]
        succ = np.zeros(N, bool)
        if len(idx):
            succ[idx] = self._collide(idx, sfi[idx], PWR_DBM[pi[idx]])
        frac = np.zeros(6)
        for s in range(6):
            frac[s] = (tx & (sfi == s)).sum()/N
        self.last_sf_frac = frac
        etx = E_TX_TABLE[a]
        # delivery updates gateway estimator (and replica at sensor)
        xt_dev = self.s + self.anom
        self.shat = np.where(succ, xt_dev, self.shat); self.P = np.where(succ, 0.0, self.P)
        self.aoi = np.where(succ, 0, self.aoi+1)
        self.lastok = np.where(tx, succ.astype(float), self.lastok)
        # anomaly detection events
        det = succ & (self.an_onset_pending >= 0)
        for i in np.where(det)[0]:
            self.events.append(self.t - self.an_onset_pending[i]); self.an_onset_pending[i] = -1
        # energy
        H = self.eta*self.solar()*self.dt*self.rng.uniform(0.9, 1.1, N)
        self.harv = H
        self.Hbar = 0.995*self.Hbar + 0.005*H
        if self.knoise > 0 and (self.t % self.slots_day) == int(18*3600/self.dt):
            self.kap_t = self.kappa*self.rng2.uniform(1-self.knoise, 1+self.knoise, N)
        elif self.knoise == 0:
            self.kap_t = self.kappa
        cons = etx + np.where(self.alive, self.base*self.kap_t, 5e-6*self.dt)
        self.E = np.clip(self.E + H - cons, 0, self.E_max)
        self.bucket = np.minimum(self.bucket - AIR_TABLE[a] + 0.01*self.dt, 36.0)
        self.Z = np.maximum(self.Z + etx - self.beta*np.maximum(self.Hbar - self.base, 0), 0)
        was = self.alive.copy()
        self.alive = np.where(self.alive, self.E > self.E_min, self.E > self.E_on)
        outage = ~self.alive
        # process evolution
        self.t += 1
        if self.rng.random() < 0.01:
            self.cloud = self.rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
        self.s = self.rho*self.s + self.sw*self.rng.normal(0, 1, N)
        self.shat = self.rho*self.shat; self.P = self.rho**2*self.P + self.sw**2
        self.an_left = np.maximum(self.an_left-1, 0)
        ended = (self.an_left == 0) & (self.anom != 0)
        for i in np.where(ended & (self.an_onset_pending >= 0))[0]:
            self.events.append(-1); self.an_onset_pending[i] = -1
        self.anom = np.where(self.an_left > 0, self.anom, 0.0)
        new = (self.anom == 0) & (self.rng.random(N) < self.an_p)
        if new.any():
            k = new.sum()
            self.anom[new] = self.rng.choice([-1, 1], k)*self.rng.uniform(3, 6, k)*np.sqrt(self.sig2)
            self.an_left[new] = self.rng.integers(5, 31, k)
            self.an_onset_pending[new] = self.t
        e2 = np.minimum(self.err2(), 25.0)
        info = dict(tx=tx, succ=succ, sfi=sfi, pi=pi, etx=etx, outage=outage, e2=e2)
        return e2, outage, info
