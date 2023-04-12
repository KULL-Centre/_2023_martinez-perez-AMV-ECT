import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations

def initProteins():
    proteins = pd.DataFrame(columns=['eps_factor','pH','ionic','fasta'])
    fasta_ECT2N = """MATVAPPADQATDLLQKLSLDSPAKASEIPEPNKKTAVYQYGGVDVHGQVPSYDRSLTPMLPSDAADPSVCYVPNPYN
PYQYYNVYGSGQEWTDYPAYTNPEGVDMNSGIYGENGTVVYPQGYGYAAYPYSPATSPAPQLGGEGQLYGAQQYQYPNYFPNSGPYASSVATPTQPDLSANKPA
GVKTLPADSNNVASAAGITKGSNGSAPVKPTNQATLNTSSNLYGMGAPGGGLAAGYQDPRYAYEGYYAPVPWHDGSKYSDVQRPVSGSGVASSYSKSSTVPSSR
NQNYRSNSHYTSVHQPSSVTGYGTAQGYYNRMYQNKLYGQYGSTGRSALGYGSSGYDSRTNGRGWAATDNKYRSWGRGNSYYYGNENNVDGLNELNRGPRAKGT
KNQKGNLDDSLEVKEQTGESNVTEVGEADNTCVVPDREQYNKEDFPVDYA""".replace('\n', '')
    fasta_ECT2N_del2 = """MATVAPPADQATDLLQKLSLDSPAKASEIPEPNKKTAVYQYGGVDVHGQVPSYDRSLTPMLPSDAADPSVCYV
PNPYNPYQYYNVYGSGQEWTDYPAYTNPEGVDMNSGIYGENGYASSVATPTQPDLSANKPAGVKTLPADSNNVASAAGITKGSNGSAPVKPTNQATLNTSSNLY
GMGAPGGGLAAGYQDPRYAYEGYYAPVPWHDGSKYSDVQRPVSGSGVASSYSKSSTVPSSRNQNYRSNSHYTSVHQPSSVTGYGTAQGYYNRMYQNKLYGQYGS
TGRSALGYGSSGYDSRTNGRGWAATDNKYRSWGRGNSYYYGNENNVDGLNELNRGPRAKGTKNQKGNLDDSLEVKEQTGESNVTEVGEADNTCVVPDREQYNKE
DFPVDYA""".replace('\n', '')
    fasta_ECT2N_r2_9xYA = """MATVAPPADQATDLLQKLSLDSPAKASEIPEPNKKTAVYQYGGVDVHGQVPSYDRSLTPMLPSDAADPSVCYVPNPYN
PYQYYNVYGSGQEWTDYPAYTNPEGVDMNSGIYGENGTVVAPQGAGAAAAPASPATSPAPQLGGEGQLAGAQQAQAPNAFPNSGPYASSVATPTQPDLSANKPA
GVKTLPADSNNVASAAGITKGSNGSAPVKPTNQATLNTSSNLYGMGAPGGGLAAGYQDPRYAYEGYYAPVPWHDGSKYSDVQRPVSGSGVASSYSKSSTVPSSR
NQNYRSNSHYTSVHQPSSVTGYGTAQGYYNRMYQNKLYGQYGSTGRSALGYGSSGYDSRTNGRGWAATDNKYRSWGRGNSYYYGNENNVDGLNELNRGPRAKGT
KNQKGNLDDSLEVKEQTGESNVTEVGEADNTCVVPDREQYNKEDFPVDYA""".replace('\n', '')
    A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    FUS = """MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYG
QSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQS
SSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')
    
    proteins.loc['ECT2N'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_ECT2N),ionic=0.15)
    proteins.loc['ECT2N_del2'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_ECT2N_del2),ionic=0.15)
    proteins.loc['ECT2N_r2_9xYA'] = dict(eps_factor=0.2,pH=7.0,fasta=list(fasta_ECT2N_r2_9xYA),ionic=0.15)
    proteins.loc['A1'] = dict(eps_factor=0.2,pH=7.0,fasta=list(A1),ionic=0.15)
    proteins.loc['FUS'] = dict(eps_factor=0.2,pH=7.0,fasta=list(FUS),ionic=0.15)
    return proteins

def genParamsLJ(df,name,prot):
    fasta = prot.fasta.copy()
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = prot.eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,name,prot,temp):
    kT = 8.3145*temp*1e-3
    fasta = prot.fasta.copy()
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[prot.fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[prot.fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

def genDCD(residues,name,prot,temp,n_chains):
    path = '{:s}/{:d}'.format(name,temp)
    top = md.Topology()
    for _ in range(n_chains):
        chain = top.add_chain()
        for resname in prot.fasta:
            residue = top.add_residue(residues.loc[resname,'three'], chain)
            top.add_atom(residues.loc[resname,'three'], 
                         element=md.element.carbon, residue=residue)
        for i in range(chain.n_atoms-1):
            top.add_bond(chain.atom(i),chain.atom(i+1))

    t = md.load(path+'/{:s}.dcd'.format(name),top=top)
    t.xyz *= 10
    t.unitcell_lengths *= 10
    lz = t.unitcell_lengths[0,2]
    edges = np.arange(-lz/2.,lz/2.,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a.argmax()], 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)
    t[0].save_pdb(path+'/top.pdb')
    t.save_dcd(path+'/traj4.dcd')

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj4.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj3.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices): 
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj3.dcd',top=path+'/top.pdb')
    edges = np.arange(0,lz,1)
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[:120],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj3.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj2.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)  
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj2.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    zmid = np.apply_along_axis(lambda a: z[a>np.quantile(a,.98)].mean(), 1, h)
    indices = np.argmin(np.abs(t.xyz[:,:,2]-zmid[:,np.newaxis]),axis=1)

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj2.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj1.dcd', ag.n_atoms) as W:
        for ts,ndx in zip(u.trajectory,indices): 
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.center_in_box(
                u.select_atoms('index {:d}'.format(ndx)), center='geometry')(ts)
            ts = transformations.wrap(ag)(ts)
            W.write(ag)

    t = md.load(path+'/traj1.dcd',top=path+'/top.pdb')
    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    h = np.mean(h[120:],axis=0)
    maxoverlap = np.apply_along_axis(lambda a: np.correlate(h,np.histogram(a,
                bins=edges)[0], 'full').argmax()-h.size+dz, 1, t.xyz[:,:,2])

    u = MDAnalysis.Universe(path+'/top.pdb',path+'/traj1.dcd')
    ag = u.atoms
    with MDAnalysis.Writer(path+'/traj.dcd', ag.n_atoms) as W:
        for ts,mo in zip(u.trajectory,maxoverlap):
            ts = transformations.unwrap(ag)(ts)
            ts = transformations.translate([0,0,mo*10])(ts)  
            ts = transformations.wrap(ag)(ts)
            W.write(ag)
   
    t = md.load(path+'/traj.dcd',top=path+'/top.pdb')

    h = np.apply_along_axis(lambda a: np.histogram(a,bins=edges)[0], 1, t.xyz[:,:,2])
    np.save('{:s}_{:d}.npy'.format(name,temp),h,allow_pickle=False)
    os.remove(path+'/traj1.dcd')
    os.remove(path+'/traj2.dcd')
    os.remove(path+'/traj3.dcd')
    os.remove(path+'/traj4.dcd')
    t.xyz /= 10
    t.unitcell_lengths /= 10
    t[0].save_pdb(path+'/top.pdb')
    t.save_dcd(path+'/traj.dcd')
