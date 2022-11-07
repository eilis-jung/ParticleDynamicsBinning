#include "../include/binningAlgorithm.h"
#include "../include/constants.h"
#include "../include/vs.h"

/*
binLocation: index: 0 1 2 3 4 5 6 7 8 9   ... 
			 value: 9 2 4 1 3 2 8 1 12 33 ... index indicates the particle and value indicates which bin it is located in. 

rY, rX:	these are standard variables defined in box class in the beginning.
*/
__global__
void locateBin(int *binLocation, double *rX, double *rY){	// binLocation[0] will be zero and shall be taken into account later.
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i<N){
		int x = (int)((rX[i] - binXMin) / gridSizeX);
		int y = (int)((rY[i] - binYMin) / gridSizeY);
		binLocation[i]=y*nBinAlongX+x;
		
	}
}


/*
binLocation: index: 0 1 2 3 4 5 6 7 8  9 ...
			 value: 0 2 4 1 3 2 8 1 12 33... index indicates particle ID and value indicates the bin No. the particle is located in.
pInBin:		 index : 0 1 2 3 4 5 6 7 8 9 ...
             binNo : 0 0 1 1 2 2 3 3 4 4 ... // this information is not stored anywhere but calculated from index
			 value : 0 0 3 7 1 5 4 0  
*/



__global__
void assignParticleToBin1(int *binLocation, int *pInBin){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i<N){
		int k = binLocation[i];
		if((k>0 && k<nBinAlongX*nBinAlongY)){
			pInBin[2*k] = i;
		}
	}
}

__global__
void assignParticleToBin2(int *binLocation, int *pInBin){
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i>0 && i<N){
		int k = binLocation[i];
		if((k>0 && k<nBinAlongX*nBinAlongY)){
		if(pInBin[2*k]!=i){
			pInBin[2*k+1] = i;
		}
		}
	}
}





/*
pInBin: same as above.
pID : index : 					0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20...
	  corresponding particle :	0 0 0 0 0 0 1 1 1 1 1  1  2  2  2  2  2  2  3  3  3 ... // this is not stored but this information is deduced from index of pID
	  particle collided with	0 0 0 0 0 0 3 4 0 0 0  0  4  9  12 34 0  0  0  0  0 ... // this is the particle with which corresponding particle above row collides with.
																						// 0 is supposed to indicate no collision with any particle.

*/
__global__
void detectCollisionBtwnParticles(int * pInBin, int *pID, double *rX, double *rY){
	
	
	int i = threadIdx.x + blockIdx.x * blockDim.x + 681;	// Note because of L shape, we start from 2nd position of second row which is index 681, because of this we need to create 680*680 - 681 threads 
	// or generate more threads and disregard.
	// We can generate 680*680 threads and disregard other threads.
	
	if(i < nBinAlongX*nBinAlongY){
		
		// ________________________________________________________
		// Collision between particles within each bin
		// ________________________________________________________
		int collided = 0;
		int collisionNo = 0;
		
		// two particles in current bin
		int p1 = pInBin[2*i];		// particle IDs start with 1, so to get index we substract 1.
		int p2 = pInBin[2*i+1];

		// detect if they collide and store information in particle1's collision information
		collided = (2*R-sqrt((rX[p1]-rX[p2])*(rX[p1]-rX[p2]) + (rY[p1]-rY[p2])*((rY[p1]-rY[p2]) )))+ EPS>=0.;
		
		// Notice that if pInBin[i] is 0, then collided will be false. so no collision information will be stored
		pID[6*p1] = p2*collided;	// stores p2 if collided and 0 else
		collisionNo+=collided;

		
		// ________________________________________________________
		// Collision across bin - neighbors
		// ________________________________________________________
		// get ID of bins in L-shaped neighborhood:
		int bW = i-1;
		int bSW = i-nBinAlongX-1;
		int bS = i-nBinAlongX;
		int bSE = i-nBinAlongX+1;


		// for each of those particles check neighbors
		for(int j =0;j<2;j++){
			
			// detect collisionNo
			int pIdI = (j==0) ? p1 : p2;	// this is basically check collision with particle p1 if j=0 and check collision with particle p2 if iteration is 2
			
			double rXI = rX[pIdI];
			double rYI = rY[pIdI];
			
			// particles in western bin
			for (int k = 0; k < 2; k++) {
				int pIdJ = pInBin[2*bW+k];
				collided = (2*R-sqrt((rXI-rX[pIdJ])*(rXI-rX[pIdJ]) + (rYI-rY[pIdJ])*((rYI-rY[pIdJ]) )))+EPS>=0.;
				pID[6*pIdI+collisionNo] = pIdJ*collided;
				collisionNo+=collided;
			}
			
			// particles in south western bin
			for (int k = 0; k < 2; k++) {
				int pIdJ = pInBin[2*bSW+k];
				collided = (2*R-sqrt((rXI-rX[pIdJ])*(rXI-rX[pIdJ]) + (rYI-rY[pIdJ])*((rYI-rY[pIdJ]) )))+EPS>=0.;
				pID[6*pIdI+collisionNo] = pIdJ*collided;
				collisionNo+=collided;
			}
			
			// particles in southern bin
			for (int k = 0; k < 2; k++) {
				int pIdJ = pInBin[2*bS+k];
				collided = (2*R-sqrt((rXI-rX[pIdJ])*(rXI-rX[pIdJ]) + (rYI-rY[pIdJ])*((rYI-rY[pIdJ]) )))+EPS>=0.;
				pID[6*pIdI+collisionNo] = pIdJ*collided;
				collisionNo+=collided;
			}			
			
			// particles in south eastern bin
			for (int k = 0; k < 2; k++) {
				int pIdJ = pInBin[2*bSE+k];
				collided = (2*R-sqrt((rXI-rX[pIdJ])*(rXI-rX[pIdJ]) + (rYI-rY[pIdJ])*((rYI-rY[pIdJ]) )))+EPS>=0.;
				pID[6*pIdI+collisionNo] = pIdJ*collided;
				collisionNo+=collided;
			}
			
			// reset collision ID as information will now be stored in particle p2's side
			collisionNo = 0;
		}
	}
}

/*
pId : stores information about particle collided with.
so, xit is not integrated value but relative tangential velocity * dt.
*/

__global__
void computeXitParticle(int * pId, double * rX, double * rY, double * vX, double * vY, double * omega, double * xit){

	int i = threadIdx.x + blockIdx.x * blockDim.x;
		
	if(i>5 && i < 6 * N){	// First six correspond to particle collision, so we disregard them.
	
		
		int idxJ = pId[i];
		if(idxJ!=0){		// TODO consider warp divergence
			int idxI = i/6;
			double d = sqrt( (rY[idxJ] - rY[idxI]) * (rY[idxJ] - rY[idxI]) + (rX[idxJ] - rX[idxI]) * (rX[idxJ] - rX[idxI]) );
			double etijX = -(rY[idxJ] - rY[idxI])/d;
			double etijY = (rX[idxJ] - rX[idxI])/d;
			double rTV = etijX * (vX[idxJ] + vX[idxJ])+ etijY * (vY[idxJ] + vY[idxJ]) + R * (omega[idxJ] + omega[idxI]);
			xit[i] = rTV*dt;
		}
		
	}
}

/*
index					:	0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
corresponding particle  :	0 0 0 0 0 0 1 1 1 1 1  1  2  2  2  2  2  2  3  3  3 ...
pIdOld					:	0 0 0 0 0 0 2 3 0 0 0  0  3  5  13  0  0  0  19 21 0 ...
xitOld value			:   0 0 0 0 0 0 a b 0 0 0  0  c  d  e  0  0  0  f  g  0 ...

index					:	0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
corresponding particle  :	0 0 0 0 0 0 1 1 1 1 1  1  2  2  2  2  2  2  3  3  3 ...
pIdNew					:	0 0 0 0 0 0 5 3 6 0 0  0  5  9  12 14 0  0  19 21 0 ...
xitNewvalue				:   0 0 0 0 0 0 m n 0 0 0  0  o  p  q  r  0  0  s  t  0 ...

So, for each particle, this function compares all pIdNew and pIdOld corresponding to a specific particle and if they are same adds value from 
e.g.
after this function, xitNewvalue will look like:


index					:	0 1 2 3 4 5 6 7   8 9 10 11 12  13 14 15 16 17 18  19  20
corresponding particle  :	0 0 0 0 0 0 1 1   1 1 1  1  2   2  2  2  2  2  3   3   3 ...
pIdNew					:	0 0 0 0 0 0 5 3   6 0 0  0  5   9  12 14 0  0  19  21  0 ...
xitNewvalue				:	0 0 0 0 0 0 m n+b 0 0 0  0  o+d p  q  r  0  0  f+s t+g 0 ...

pIdNew is not changed and index and corresponding particles are just for understanding, they are not actually there in the array

Notice that we get n+b because for specific corresponding particle 1, pIdNew and pIdOld have id 3 in common. so only that corresponding xitOld needs to be added to xitNew.

*/


__global__
void updateXitParticle(int * pIdOld, double * xitOld, int * pIdNew, double * xitNew){
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(i < N){
	
		for (int j = 0; j < maxCollisionPerParticle; j++) {			// For each collision corresponding to particle i in current iteration 
			for (int k = 0; k < maxCollisionPerParticle; k++) {			// loop through all collisions corresponding to particle i in previous iteration
			
				int id1 = i*6+j;
				int id2 = i*6+k;
				int repeated = pIdOld[id2] == pIdNew[id1];
				xitNew[id1] += repeated * (xitOld[id2] + xitNew[id1]);
				
			}
		}
	}
}