#include "../include/collisionWithWall.h"
#include "../include/constants.h"
#include "../include/vs.h"

#include<stdio.h>
/*
	collision information of particle with position x[i],y[i] are stored in wallID index 2*i, and 2*i+1 since a particle can collide with at most two walls:
	double X, Y: position of particle
	double wallX, wallY: coordinates of corners of squares
*/
__global__
void detectCollisionWithWall(double * rX, double * rY, double * wallX, double * wallY, int * wallID){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i < N){	// particle 0 is ghost particle that need not collide with any wall
	
		int collisionNo = 0; // track collision
		
		double rx = rX[i];
		double ry = rY[i];
		
		for(int j =0; j<4; j++){
		
			double x1 = wallX[j];
			double y1 = wallY[j];
			double x2 = wallX[(j+1)%4];
			double y2 = wallY[(j+1)%4];

			// ************* check if there is collision
			
			// vector in direction of wall
			double dX = x2 - x1;
			double dY = y2 - y1;
			
			/* vector from center of circle to start of wall*/
			double fX = x1 - rx;
			double fY = y1 - ry;
			
			/* a, b, c are coefficients of quadratic equation */ 
			double a = dX * dX + dY * dY;
			double b = 2. * (dX * fX + dY * fY);
			double coef = fX * fX + fY * fY - R * R;
			double discriminant = b * b - 4. * a * coef;
			int collided = (discriminant + EPS >= 0);
			// ******** Update information
			int aa = (j+1)*collided;
			wallID[2*i+collisionNo] = aa;
			collisionNo += collided;
			
		}
	}
}

__global__
void computeXitWall(int * wallID, double * rX, double * rY, double * vX, double * vY, double * omega, double * xit, double *wallX, double *wallY){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>1 && i<2*N){		// Do not check collisions corresponding to particle 0.
			int wId = wallID[i];
		if(wId!=0){
			int pId = i/2;
			
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
			double normWall = sqrt((y2-y1)*(y2-y1)+(x2-x1)*(x2-x1));	// length of wall
			double normV = sqrt(vX[pId]*vX[pId]+vY[pId]*vY[pId]);	// length of velocity vector
			
			// projection of velocity vector along the wall
			double pX = normV*(x2-x1)/normWall;
			double pY = normV*(y2-y1)/normWall;
			
			// vector addition to compute the values // reflection = 2*projection-velocity
			double gvX = 2.*pX-vX[pId];
			double gvY = 2.*pY-vY[pId];

			// computation of Xit
			
			double d1 = sqrt( (gY - rY[pId]) * (gY - rY[pId]) + (gX - rX[pId]) * (gX - rX[pId]) );
			double etijX = -(rY[pId] - rY[pId])/d1;
			double etijY = (rX[pId] - rX[pId])/d1;
			double rTV = etijX * (gvX + vX[pId])+ etijY * (gvY + vY[pId]) + R * omega[pId];
			
			xit[i] = rTV*dt;
		}
		
	}
	
}

__global__
void updateXitWall(int * wallIDOld, double * xitOld, int * wallIDNew, double * xitNew){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i < N){
		for(int j =0; j<2; j++){
			for(int k =0; k<2; k++){
				int match = wallIDOld[2*i+j]==wallIDNew[2*i+k];
				xitNew[2*i+k] += match*xitOld[2*i+j];
			}
		}
	}
}