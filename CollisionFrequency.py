import math
class CollisionKernal:
    def __init__(self,T,P,db1,db2,Rho_particle,CMD,Nconc_tot_0):
        self.T=T
        self.P=P 
        self.db1=db1
        self.db2=db2
        self.Rho_particle=Rho_particle
        self.CMD=CMD
        self.Nconc_tot_0=Nconc_tot_0
    def Fuchs(self):
        A_mu = 0.000001966 #Sutherland's constants for carrier gas in the form A*T^1.5/(B+T)
        B_mu = 147.47
        R  = 8.314          #Universal gas constant Joule/(K.mol)
        Kmin = 0.000000001
        Kmax = 0
        T=self.T
        P=self.P
        Rho_particle=self.Rho_particle
        db1=self.db1
        db2=self.db2
        kb= 1.38e-23        #Boltzmann constant
        pi=3.1415926
        mb1=(pi/6)*pow(db1,3)*Rho_particle
        mb2=(pi/6)*pow(db2,3)*Rho_particle
        mu = A_mu * (T**1.5) / (B_mu + T)
        #print("sqrt=",math.sqrt(64))
        #lam=20
        #lam = (mu / (P)) * math.sqrt(3.14 * R * T / (2 * 0.04))
        # Molar mass of air changed from 0.04 to 0.02896 kg/mol
        lam = (mu / (P)) * math.sqrt( 3.14 * R * T / (2 * 0.02896) )
        # print("lam=",lam)
        kn1 = (2.0 * lam) / db1
        kn2 = (2.0 * lam) / db2
        #D1 = (kb * T) / (3.0 * pi * mu * db1) * ((5.0 + 4.0 * kn1 + 6.0 * kn1 * kn1 + 18.0 * kn1 * kn1 * kn1) / (5.0 - kn1 + (8.0 + pi) * kn1 * kn1))
        #D2 = (kb * T) / (3.0 * pi * mu * db2) * ((5.0 + 4.0 * kn2 + 6.0 * kn2 * kn2 + 18.0 * kn2 * kn2 * kn2) / (5.0 - kn2 + (8.0 + pi) * kn2 * kn2))
        Cc1 = 1.0 + kn1 * (1.257 + 0.40 * math.exp( -1.1 / kn1 ))
        Cc2 = 1.0 + kn2 * (1.257 + 0.40 * math.exp( -1.1 / kn2 ))
        D1 = (kb * T) / (3.0 * pi * mu * db1)*Cc1
        D2 = (kb * T) / (3.0 * pi * mu * db2)*Cc2
        c1 = math.sqrt((8.0 * kb * T) / (pi * mb1))
        c2 = math.sqrt((8.0 * kb * T) / (pi * mb2))
        l1 = (8.0 * D1) / (pi * c1)
        l2 = (8.0 * D2) / (pi * c2)
        g1 = (pow((db1 + l1), 3) - pow((db1 * db1 + l1 * l1), 1.5)) / (3.0 * db1 * l1) - db1
        g2 = (pow((db2 + l2), 3) - pow((db2 * db2 + l2 * l2), 1.5)) / (3.0 * db2 * l2) - db2
        KK = 2.0 * pi * (D1 + D2) * (db1 + db2) / ((db1 + db2) / (db1 + db2 + 2.0 * math.sqrt(g1 * g1 + g2 * g2)) + (8.0 * (D1 + D2)) / (math.sqrt(c1 * c1 + c2 * c2) * (db1 + db2)))
        if KK <Kmin:
            Kmin=KK
        if KK>Kmax:
            Kmax=KK
        # print("KK=",KK)
        return KK
    def Char_coagT(self):
        A_mu = 0.000001966 #Sutherland's constants for carrier gas in the form A*T^1.5/(B+T)
        B_mu = 147.47
        R  = 8.314          #Universal gas constant Joule/(K.mol)
        Kmin = 0.000000001
        Kmax = 0
        T=self.T
        P=self.P
        Rho_particle=self.Rho_particle
        db1=self.CMD
        db2=self.CMD
        Nconc_tot_0=self.Nconc_tot_0
        kb= 1.38e-23        #Boltzmann constant
        pi=3.1415926
        mb1=(pi/6)*pow(db1,3)*Rho_particle
        mb2=(pi/6)*pow(db2,3)*Rho_particle
        mu = A_mu * (T**1.5) / (B_mu + T)
        #print("sqrt=",math.sqrt(64))
        #lam=20
        lam = (mu / (P)) * math.sqrt(3.14 * R * T / (2 * 0.04))
        # print("lam=",lam)
        kn1 = (2.0 * lam) / db1
        kn2 = (2.0 * lam) / db2
        D1 = (kb * T) / (3.0 * pi * mu * db1) * ((5.0 + 4.0 * kn1 + 6.0 * kn1 * kn1 + 18.0 * kn1 * kn1 * kn1) / (5.0 - kn1 + (8.0 + pi) * kn1 * kn1))
        D2 = (kb * T) / (3.0 * pi * mu * db2) * ((5.0 + 4.0 * kn2 + 6.0 * kn2 * kn2 + 18.0 * kn2 * kn2 * kn2) / (5.0 - kn2 + (8.0 + pi) * kn2 * kn2))
        c1 = math.sqrt((8.0 * kb * T) / (pi * mb1))
        c2 = math.sqrt((8.0 * kb * T) / (pi * mb2))
        l1 = (8.0 * D1) / (pi * c1)
        l2 = (8.0 * D2) / (pi * c2)
        g1 = (pow((db1 + l1), 3) - pow((db1 * db1 + l1 * l1), 1.5)) / (3.0 * db1 * l1) - db1
        g2 = (pow((db2 + l2), 3) - pow((db2 * db2 + l2 * l2), 1.5)) / (3.0 * db2 * l2) - db2
        KK = 2.0 * pi * (D1 + D2) * (db1 + db2) / ((db1 + db2) / (db1 + db2 + 2.0 * math.sqrt(g1 * g1 + g2 * g2)) + (8.0 * (D1 + D2)) / (math.sqrt(c1 * c1 + c2 * c2) * (db1 + db2)))
        if KK <Kmin:
            Kmin=KK
        if KK>Kmax:
            Kmax=KK
        coaT = 1 /(KK * Nconc_tot_0)
        # print("KK=",KK)   
        return coaT



