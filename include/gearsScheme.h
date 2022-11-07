#ifndef GEAR_SCHEME_H
#define GEAR_SCHEME_H

__global__
void gearTranslationSolver(double *, double *, double *, double *,double *, double *, double *, double *, double *, double *);
	
__global__
void gearRotationSolver(double *, double *, double *, double *, double *);

#endif