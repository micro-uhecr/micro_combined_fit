import sys
import numpy as np
                        
from crpropa import *

def zbins(zmin = 0, zmax = 2.5, dz_min = 1E-4, dz_max = 0.5):
    """Function extracted from ShapeToTensor.py that defines the 
    redshift bin edges.
    """
    dz = lambda z: dz_min + dz_max*(z-zmin)/(zmax-zmin)
    list_z = [zmin]
    while list_z[-1]<=zmax-dz_max: list_z.append(list_z[-1]+dz(list_z[-1]))
    list_z.append(zmax)
    return np.array(list_z)

class Output_as_Simprop(Module):
    """Outputs parent particle information.
    """
    def __init__(self, fname, Injected_Species, Dlim=2.5 * Gpc):
        Module.__init__(self)

        self.fout = open(fname, 'w')
        self.Dlim = Dlim
        self.sn_list = []
        self.injspec = Injected_Species

    def process(self, c):
        pid = c.current.getId()
        sn = c.getCreatedSerialNumber()
        z = c.getRedshift()
        
        Z = chargeNumber(pid)
        A = massNumber(pid)
        
        e = c.current.getEnergy() / eV
        R = c.current.getPosition().getR()
        
        if R >= self.Dlim:
            if (pid == self.injspec) and (sn not in self.sn_list): 
                self.fout.write(f'\n{sn:d} {Z:d} {A:d} {e:+5.4e} {z:7.6f} 0')
                self.sn_list.append(sn)
        elif R <= 1 * pc:
            self.fout.write(f' {Z:d} {A:d} {e:+5.4e} 1.0')
        
    def close(self):
        self.fout.close()

class Output_as_Simprop(Module):
    """Outputs parent particle information.
    """
    def __init__(self, fname, Injected_Species, Dlim=2.5 * Gpc):
        Module.__init__(self)

        self.fout = open(fname, 'w')
        self.Dlim = Dlim
        self.sn_list = []
        self.injspec = Injected_Species

    def process(self, c):
        R = c.current.getPosition().getR()
        
        if R <= 1 * pc:
            pid = c.current.getId()
            sn = c.getCreatedSerialNumber()
            z = c.getRedshift()
            
            Z = chargeNumber(pid)
            A = massNumber(pid)
            
            e = c.current.getEnergy() / eV
            if R <= self.Dlim:
                if (pid == self.injspec) and (sn not in self.sn_list): 
                    self.fout.write(f'\n{sn:d} {Z:d} {A:d} {e:+5.4e} {z:7.6f} 0')
                    self.sn_list.append(sn)
            self.fout.write(f' {Z:d} {A:d} {e:+5.4e} 1.0')
        
    def close(self):
        self.fout.close()

def run_1Dpropagation(Z, A, Nprim=10, zmin=0, zmax=2.5):
    """Simulation to obtain the propagation tensor.
    """
    # Defining the cosmological parameters: Ho=67.3 km/s/Mpc, OmegaM=0.3
    setCosmologyParameters(0.673, 0.3)
    
    Injected_Species = nucleusId(A, Z)
    Dmin, Dmax = redshift2ComovingDistance(zmin), redshift2ComovingDistance(zmax)
    
    if Dmin == 0:
        Dmin = 1 * pc
    
    # simulation setup
    sim = ModuleList()

    # propagation settings
    sim.add( SimplePropagation(100*kpc, 1000*kpc) )
    sim.add( Redshift() )

    # photopion production CMB+IRB
    sim.add( PhotoPionProduction(CMB()) )
    sim.add( PhotoPionProduction(IRB_Gilmore12()) ) 

    # photodisintegration production CMB+IRB
    sim.add( PhotoDisintegration(CMB()) )
    sim.add( PhotoDisintegration(IRB_Gilmore12()) )

    # other interaction losses
    sim.add( NuclearDecay() )
    sim.add( ElectronPairProduction(CMB()) )
    sim.add( ElectronPairProduction(IRB_Gilmore12()) )
    sim.add( MinimumEnergy(EeV) )
    
    myoutput = Output_as_Simprop(f'CRPropa_{A:d}_{zmin:5.4f}_{zmax:5.4f}.txt', Injected_Species, Dmin-100*kpc) # use extension .gz
    
    # observer to stop particles at origin
    obs = Observer()
    obs.add( ObserverPoint() )
    sim.add( obs )
    
    sim.add( myoutput )
    
    # source
    Emin = Z * 0.1 * EeV
    Rmax = 1000 * EeV
    source = Source()
    source.add( SourceUniform1D(Dmin, Dmax) )
    source.add( SourceRedshift1D() )
    composition = SourceComposition(Emin, Rmax, -1) # flat spectrum with Emin=0.1*Z EeV and charge dependent maximum energy Z*1000 EeV
    composition.add( A, Z, 1 ) 
    source.add( composition )


    # run simulation
    sim.setShowProgress(True)
    sim.run(source, Nprim, True)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        # run_1Dpropagation(26, 56, 100, 0.5, 2.5)
        Z, A, Nprim, zmin, zmax = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        run_1Dpropagation(Z, A, Nprim, zmin, zmax)

    elif sys.argv[1] == 'local':
        A = [1, 4, 12, 14, 28, 56] 
        Z = [1, 2,  6,  7, 16, 26]

        rs = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.5]
        
        for z, a in zip(Z, A):
            for z1, z2 in zip(rs[:-1], rs[1:]):
                run_1Dpropagation(z, a, 100, z1, z2)

    else:
        A = [1, 4, 12, 14, 16, 20, 24, 28, 32, 56] 
        Z = [1, 2,  6,  7,  8, 10, 12, 14, 16, 26]

        # rs = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.5]
        rs = zbins()

        with open('parameter_set.sh', 'w') as sfile:
            for z, a in zip(Z, A):
                for z1, z2 in zip(rs[:-1], rs[1:]):
                    sfile.write(f' {z:>2d} {a:>2d} 500000 {z1:f} {z2:f} \n')