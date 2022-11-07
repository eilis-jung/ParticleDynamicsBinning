#include "../include/constants.h"
#include "../include/gearsScheme.h"
#include "stdio.h"
#include "../include/vs.h"

/*
	rx : x component of position vector at time t
	ry : y component of position vector at time t
	vx : x component of velocity vector at time t
	vy : y component of velocity vector at time t
	ax : x component of acceleration vector at time t
	ay : y component of acceleration vector at time t
	jx : x component of jerk vector at time t 
	jy : y component of jerk vector at time t
	dt : step size
	m : mass of the particle
	Fx : Force at time (t+dt)
	Fy : Force at time (t+dt)
*/

__global__
void gearTranslationSolver(double *rx, double *ry, double *vx, double *vy,
	double *ax, double *ay, double *jx, double *jy, double * Fx, double * Fy){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i<N){

		// predictor step
		double rxP = rx[i] + dt * vx[i] + dt * dt * ax[i] / 2. + dt * dt * dt * jx[i] / 6.;
		double ryP = ry[i] + dt * vy[i] + dt * dt * ay[i] / 2. + dt * dt * dt * jy[i] / 6.;
		
		double vxP = vx[i] + dt * ax[i] + dt * dt * jx[i] / 2.;
		double vyP = vy[i] + dt * ay[i] + dt * dt * jy[i] / 2.;
		
		double axP = ax[i] + dt * jx[i];
		double ayP = ay[i] + dt * jy[i];
		
		double jxP = jx[i];
		double jyP = jy[i];		
		
		// Corrector Step:
		
		// computation/evaluation step
		double axC = (Fx[i] + m * gx) / m;
		double ayC = (Fy[i] + m * gy) / m;
		
		// correction term
		double delaX = (axC - axP)*dt*dt/2.;
		double delaY = (ayC - ayP)*dt*dt/2;
		
		// corrector step update rx, ry, vx, vy, ax, ay, jx, jy
		rx[i] = rxP + c0 * delaX;
		ry[i] = ryP + c0 * delaY;
		
		vx[i] = vxP + c1 * delaX/dt;
		vy[i] = vyP + c1 * delaY/dt;
		
		ax[i] = axP + 2.*c2 * delaX/(dt*dt);
		ay[i] = ayP + 2.*c2 * delaY/(dt*dt);
		
		jx[i] = jxP + 6.*c3 * delaX/(dt*dt*dt);
		jy[i] = jyP + 6.*c3 * delaY/(dt*dt*dt);
		
	}

}

/*
	phi : angular position
	omega : angular velocity
	alpha : angular acceleration
	zeta : angular jerk
	I : moment of inertia
*/

__global__
void gearRotationSolver(double *phi, double *omega, double *alpha, double *zeta, double *rDotF){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i >0 && i < N){
		// predictor 
				
		double phiP = phi[i] + dt * omega[i] + dt * dt * alpha[i] / 2. + dt * dt * dt * zeta[i] / 6.;
		
		double omegaP = omega[i] + dt * alpha[i] + dt * dt * zeta[i] / 2.;
		
		double alphaP = alpha[i] + dt * zeta[i];
		
		double zetaP = zeta[i];
		
		// Correction Step:
		
		// computation/evaluation step
		double alphaC = rDotF[i]/I;
		
		// correction term
		double delAlpha = (alphaC - alphaP)*dt*dt/2.;
		
		// update values
		phi[i] = phiP + c0 * delAlpha;
		omega[i] = omegaP + c1 * delAlpha/dt;
		alpha[i] = alphaP + 2*c2 * delAlpha/(dt*dt);
		zeta[i] = zetaP + 6*c3 * delAlpha/(dt*dt*dt);
		
	}
}