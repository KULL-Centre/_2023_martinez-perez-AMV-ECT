from analyse import *
import os
import subprocess
from jinja2 import Template
from argparse import ArgumentParser

subprocess.run('mkdir -p logs',shell=True)

submission = Template("""#!/bin/sh
#PBS -W group_list=ku_10001 -A ku_10001
#PBS -N {{name}}_{{temp}}
### Only send mail when job is aborted or terminates abnormally
#PBS -m n
### Number of nodes
#PBS -l nodes=1:ppn=40:gpus=1
### Memory
#PBS -l mem=20gb
#PBS -l walltime=24:00:00
#PBS -o logs/{{name}}_{{temp}}.out
#PBS -e logs/{{name}}_{{temp}}.err

cd $PBS_O_WORKDIR

source /home/people/sorbul/.bashrc
conda activate calvados
module purge
# module load openmpi/gcc/64/4.0.2
# module load cuda
# module load cuda10.1
python ./simulate.py --name {{name}} --temp {{temp}} --cutoff {{cutoff}}""")

proteins = initProteins()
proteins.to_pickle('proteins.pkl')

r = pd.read_csv('residues.csv').set_index('three')
r.lambdas = r['CALVADOS2'] # select CALVADOS1 or CALVADOS2 stickiness parameters
r.to_csv('residues.csv')
cutoff = 2.0 # set the cutoff for the nonionic interactions

for name,prot in proteins.loc[['ECT2N','ECT2N_del2','ECT2N_r2_9xYA','A1','FUS']].iterrows():
    if not os.path.isdir(name):
        os.mkdir(name)
    for temp in [293]:
        if not os.path.isdir(name+'/{:d}'.format(temp)):
            os.mkdir(name+'/{:d}'.format(temp))
        with open('{:s}_{:d}.sh'.format(name,temp), 'w') as submit:
            submit.write(submission.render(name=name,temp='{:d}'.format(temp),cutoff='{:.1f}'.format(cutoff)))
        subprocess.run(['qsub','{:s}_{:d}.sh'.format(name,temp)])
