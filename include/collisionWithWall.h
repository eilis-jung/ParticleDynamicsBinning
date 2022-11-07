#ifndef COLLISION_WITH_WALL_H
#define COLLISION_WITH_WALL_H

__global__
void detectCollisionWithWall(double * , double * , double * , double * , int *);

__global__
void computeXitWall(int * , double * , double * , double * , double * , double * , double * , double *, double *);

__global__
void updateXitWall(int * , double * , int * , double *);

#endif