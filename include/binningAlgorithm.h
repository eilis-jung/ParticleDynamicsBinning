#ifndef BINNING_ALGORITHM_H
#define BINNING_ALGORITHM_H

__global__
void locateBin(int *, double *, double *);

__global__
void assignParticleToBin1(int *, int *);

__global__
void assignParticleToBin2(int *, int *);

__global__
void computeXitParticle(int * , double * , double * , double * , double * , double * , double *);

__global__
void detectCollisionBtwnParticles(int *, int *, double *, double *);

__global__
void updateXitParticle(int * , double * , int * , double * );

#endif