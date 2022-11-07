#ifndef FORCE_COMPUTATION_H
#define FORCE_COMPUTATION_H

__global__
void updateForceFromParticleCollision(int *, double *, double *, double *, double *, double *, double *, double *, double *);

__global__
void updateForceFromWallCollision(int * , double * , double * , double * , double * , double * , double * , double * , double *, double *, double * );

#endif