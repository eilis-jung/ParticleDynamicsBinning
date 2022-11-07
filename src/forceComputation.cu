#include "../include/forceComputation.h"
#include "../include/constants.h"
#include "../include/vs.h"

#include <stdio.h>
#include <cmath>
/* 
	Compute the total force from collision information
*/
__global__
void updateForceFromParticleCollision(int *pID2, double *xit, double * rX,
	double *rY, double * vX, double *vY, double * Fx, double * Fy, double * rDotF){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>5 && i < 6*N){
		
		int index2=pID2[i];
		
		int index1=i/6;
		
		if(index2!=0){		// Do the computations only if there was collision
			// distance
			double d = sqrt( (rY[index2] - rY[index1]) * (rY[index2] - rY[index1]) + (rX[index2] - rX[index1]) * (rX[index2] - rX[index1]) );
			
			// normal Vector
			double enX = (rX[index2] - rX[index1])/d;
			double enY = (rY[index2] - rY[index1])/d;
			
			// tangential vector
			double etX = -enY;
			double etY = enX;
			
			// compute xin
			double Xin = 2. * R - d;
			double sqrtXin = sqrt(Xin);
			
			// compute ks and kn
			double ks = 8 * G * sqrtR * sqrtXin;
			double kn = (4. / 3.) * E * sqrtR * sqrtXin;
			
			// relative normal velocity:
			double rNV = (vX[index1]*enX +vY[index1]*enY)-(vX[index2]*enX +vY[index2]*enY); // Vi.Eij gives magnitude of Vi in direction of Eij, and Vj.Eij
			
			// compute normalForce
			double damping = sqrt(3*m*kn);
			double coefN = (-kn * Xin -  damping* rNV * sqrtXin);
			
			double FnX = coefN * enX;
			double FnY = coefN * enY;
			
			// compute tangentialForce
			double coefT = min((2. / 3.) * ks * xit[i], mu * sqrt(FnX *FnX + FnY * FnY));
			
			double FtX = coefT * etX;
			double FtY = coefT * etY;
			
			
			// Force update
			Fx[index1] += FtX+FnX;
			Fy[index1] += FtY+FnY;
			
			Fx[index2] += -(FtX+FnX);
			Fy[index2] += -(FtY+FnY);
			
			// Dot product update
			rDotF[index1] += (rX[index2]-rX[index1])*(FtX+FnX) + (rY[index2]-rY[index1])*(FtY+FnY);
			rDotF[index2] += rDotF[index1];
		
		}
		
	}
}


__global__
void updateForceFromWallCollision(int* wallID, double * wallX, double * wallY, double * xit, 
double * rX, double * rY, double * vX, double * vY, double * Fx, double * Fy,double * rDotF){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>1 && i < 2*N){
		// compute properties of ghost particle
		// get vertices of the wall
		
		int wId = wallID[i];
		if(wId!=0){		// if there is no collision with wall
		
			int pId = i/2;
			// get two ends of the wall
			double x1 = wallX[wId-1];
			double y1 = wallY[wId-1];
			double x2 = wallX[wId%4];
			double y2 = wallY[wId%4];
			double a = y2-y1;
			double b = x1-x2;
			double c = -a*x1-b*y1;
			double m1 = sqrt(a*a+b*b);

			double a1 = a/m1;
			double b1 = b/m1;
			double c1n = c/m1;
			
			double d = a1*rX[pId]+b1*rY[pId]+c1n;
			
			double gX = rX[pId] - 2.*a1*d;
			double gY = rY[pId] - 2.*b1*d;
			// velocity of ghost particle

			// use a constant here
			double normWall = sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1));	// length of wall
			double normV = sqrt(vX[pId]*vX[pId]+vY[pId]*vY[pId]);	// length of velocity vector
			
			// projection of velocity vector along the wall
			double pX = 1 / (normWall * normV) * (x2 - x1) * ((x2 - x1) * vX[pId] + (y2 - y1) * vY[pId]);
			double pY = 1 / (normWall * normV) * (y2 - y1) * ((x2 - x1) * vX[pId] + (y2 - y1) * vY[pId]);
			
			// vector addition to compute the values // reflection = 2*projection-velocity
			double gvX = 2*pX-vX[pId];
			double gvY = 2*pY-vY[pId];
			// Repeat everything from previous kernel
			double d2 = sqrt( (gY - rY[pId]) * (gY - rY[pId]) + (gX - rX[pId]) * (gX - rX[pId]) );
			
			// normal Vector
			double enX = (gX - rX[pId])/d2;
			double enY = (gY - rY[pId])/d2;
			
			// tangential vector
			double etX = -enY;
			double etY = enX;
			
			// TODO define both sqrt R and R // makes computationally efficient
			// compute xin
			double Xin = 2. * R - d2;
			double sqrtXin = sqrt(Xin);
			
			// compute ks and kn
			// TODO store these two coefficients rather than compute every time
			double ks = 8. * G * sqrtR * sqrtXin;
			double kn = (4. / 3.) * E * sqrtR * sqrtXin;
			
			// relative normal velocity:
			double rNV = (vX[pId]*enX +vY[pId]*enY)-(gvX*enX +gvY*enY); // Vi.Eij gives magnitude of Vi in direction of Eij, and Vj.Eij
			
			// compute normalForce
			double coefN = (-kn * Xin - Gamma * rNV * sqrtXin);
			
			double FnX = coefN * enX;
			double FnY = coefN * enY;
			
			// compute tangentialForce
			double coefT = min((2. / 3.) * ks * xit[pId], mu * sqrt(FnX * FnX + FnY * FnY));
			
			double FtX = coefT * etX;
			double FtY = coefT * etY;
			
			Fy[pId] += FtY+FnY;
			Fx[pId] += FtX+FnX;
			rDotF[pId] += (gX-rX[pId])*(FtX+FnX) + (gY-rY[pId])*(FtY+FnY);	// due to collision with wall
		}
	}
}