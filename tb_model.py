import numpy as np
from scipy.integrate import solve_ivp

class TB_Model_Article_Normalized:
    """
    Modelo normalizado para 4 fármacos: INH, RIF, SRT, PZA.
    Ecuaciones farmacocinéticas: dc/dt = mu*(u - c)
    Parámetros basados en la Tabla 2 del artículo.
    """
    def __init__(self):
        # Crecimiento
        #self.beta_s = 0.40
        #self.beta_r = 0.28

        # Muerte natural
        #self.delta_s = 0.312
        #self.delta_r = 0.28

        # Generación de resistencia
        #self.q_H = 1e-5
        #self.q_R = 1e-6
        #self.q_S = 1e-6
        #self.q_Z = 1e-4

        # Efecto farmacológico
        #self.alpha_H = 0.03
        #self.alpha_R = 0.03
        #self.alpha_S = 0.022
        #self.alpha_Z = 0.010

        # Farmacocinética
        #self.mu_H = 0.08
        #self.mu_R = 0.08
        #self.mu_S = 0.07
        #self.mu_Z = 0.06
         
        #self.beta_s = 0.38
        # self.beta_r = 0.26

        # self.delta_s = 0.312
        # self.delta_r = 0.28

        # self.q_H = 1e-5
        # self.q_R = 1e-6
        # self.q_S = 1e-6
        # self.q_Z = 1e-4

        # self.alpha_H = 0.04
        # self.alpha_R = 0.04
        # self.alpha_S = 0.03
        # self.alpha_Z = 0.014

        # self.mu_H = 0.09
        # self.mu_R = 0.09
        # self.mu_S = 0.08
        # self.mu_Z = 0.07

        self.beta_s = 0.40
        self.beta_r = 0.22

        self.delta_s = 0.312
        self.delta_r = 0.34

        self.q_H = 1.2e-5
        self.q_R = 1.2e-6
        self.q_S = 1.2e-6
        self.q_Z = 1.2e-4

        self.alpha_H = 0.03
        self.alpha_R = 0.03
        self.alpha_S = 0.022
        self.alpha_Z = 0.010

        self.mu_H = 0.08
        self.mu_R = 0.08
        self.mu_S = 0.07
        self.mu_Z = 0.06
        
    def rhs(self, t, y, uH_func, uR_func, uS_func, uZ_func):
        s, r, cH, cR, cS, cZ = y
        uH = 1.0 if uH_func(t) else 0.0
        uR = 1.0 if uR_func(t) else 0.0
        uS = 1.0 if uS_func(t) else 0.0
        uZ = 1.0 if uZ_func(t) else 0.0

        ds = (
            self.beta_s * s * (1.0 - (s + r))
            - ((self.q_H + self.alpha_H) * cH
               + (self.q_R + self.alpha_R) * cR
               + (self.q_S + self.alpha_S) * cS
               + (self.q_Z + self.alpha_Z) * cZ) * s
            - self.delta_s * s
        )
        dr = (
            self.beta_r * r * (1.0 - (s + r))
            + (self.q_H * cH + self.q_R * cR + self.q_S * cS + self.q_Z * cZ) * s
            - self.delta_r * r
        )
        dcH = self.mu_H * (uH - cH)
        dcR = self.mu_R * (uR - cR)
        dcS = self.mu_S * (uS - cS)
        dcZ = self.mu_Z * (uZ - cZ)
        return [ds, dr, dcH, dcR, dcS, dcZ]

    # def simulate(self, treatment_code, days=180, s0=1e-2, r0=1e-10):
    def simulate(self, treatment_code, days=180, s0=1e-2, r0=5e-7):
        """
        treatment_code: lista de enteros de 0 a 15, donde cada bit representa un fármaco.
        bits: [INH, RIF, SRT, PZA] (0=ninguno, 1=INH, 2=RIF, 3=INH+RIF, ...)
        """
        y0 = [s0, r0, 0.0, 0.0, 0.0, 0.0]

        def uH(t):
            d = int(np.floor(t))
            if d < 0 or d >= days:
                return False
            code = treatment_code[d]
            return (code & 1) != 0   # bit 0 = INH

        def uR(t):
            d = int(np.floor(t))
            if d < 0 or d >= days:
                return False
            code = treatment_code[d]
            return (code & 2) != 0   # bit 1 = RIF

        def uS(t):
            d = int(np.floor(t))
            if d < 0 or d >= days:
                return False
            code = treatment_code[d]
            return (code & 4) != 0   # bit 2 = SRT

        def uZ(t):
            d = int(np.floor(t))
            if d < 0 or d >= days:
                return False
            code = treatment_code[d]
            return (code & 8) != 0   # bit 3 = PZA

        t_eval = np.arange(0, days, 1)
        sol = solve_ivp(
            lambda t, y: self.rhs(t, y, uH, uR, uS, uZ),
            (0, days), y0, t_eval=t_eval, method='RK45', max_step=0.25,
            rtol=1e-6, atol=1e-9
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.t, sol.y