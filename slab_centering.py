
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
import progressbar

def calc_zpatch(z,h,cutoff=0):
    ct = 0.
    ct_max = 0.
    zwindow = []
    hwindow = []
    zpatch = [] 
    hpatch = []
    for ix, x in enumerate(h):
        if x > cutoff:
            ct += x
            zwindow.append(z[ix])
            hwindow.append(x)
        else:
            if ct > ct_max:
                ct_max = ct
                zpatch = zwindow
                hpatch = hwindow
            ct = 0.
            zwindow = []
            hwindow = []
    if ct > ct_max: # edge case (slab at side of box)
        zpatch = zwindow
        hpatch = hwindow
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def center_slab(path,name,temp,start=None,end=None,step=1,input_pdb='top.pdb'):
    u = mda.Universe(f'{path}/{input_pdb}',path+f'/{name}.dcd',in_memory=True)

    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms
    
    print(u.dimensions)
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    print(len(u.trajectory[start:end:step]))
    with mda.Writer(path+'/traj.dcd',n_atoms) as W:
        for t,ts in progressbar.progressbar(enumerate(u.trajectory[start:end:step]),min_poll_interval=1):
            # shift max density to center
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch) # center of mass of slab
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)        
            hs[t] = h
            W.write(ag)
    np.save(path+f'/{name:s}_{temp:d}.npy',hs,allow_pickle=False)
    return hs, z