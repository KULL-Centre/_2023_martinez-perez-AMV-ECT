import pandas as pd
import MDAnalysis as mda
from Bio import SeqIO, SeqUtils
import yaml

def initProteins(eps_factor=0.2,pH=7.0,ionic=0.15,temp=300,fname=None):
    """ Initialize protein dataframe with default values """
    df = pd.DataFrame(columns=['eps_factor','pH','ionic','temp','fasta'],dtype=object)
    return df

def fasta_from_pdb(pdb,selection='all',fmt='string'):
    """ Generate fasta from pdb entries """
    u = mda.Universe(pdb)
    ag = u.select_atoms(selection)
    res3 = ag.residues.resnames
    if fmt == 'string':
        fastapdb = ""
    elif fmt == 'list':
        fastapdb = []
    else:
        raise
    for res in res3:
        res1 = SeqUtils.seq1(res)
        if res1 == "":
            res1 = "X"
        fastapdb += res1
    return fastapdb

def addProtein(df,name,use_pdb=False,pdb=None,ffasta=None,eps_factor=0.2,pH=7.0,ionic=0.15,temp=300):
    if use_pdb:
        fasta = fasta_from_pdb(pdb)
    else:
        records = read_fasta(ffasta)
        fasta = records[name].seq
    df.loc[name] = dict(pH=pH,ionic=ionic,temp=temp,eps_factor=eps_factor,fasta=list(fasta))
    return df

def modProtein(df,name,**kwargs):
    for key,val in kwargs.items():
        print(key,val)
        if key not in df.columns:
            df[key] = None # initialize property, does not work with lists
        df.loc[name,key] = val
    return df

def delProtein(df,name):
    df = df.drop(name)
    return df

def subset(df,names):
    df2 = df.loc[names]
    return df2

def read_fasta(ffasta):
    records = SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))
    return records

def get_ssdomains(name,fdomains):
    with open(f'{fdomains}','r') as f:
        stream = f.read()
        domainbib = yaml.safe_load(stream)

    domains = domainbib[name]
    print(f'Using domains {domains}')
    ssdomains = []

    for domain in domains:
        ssdomains.append(list(range(domain[0],domain[1])))
    
    return ssdomains