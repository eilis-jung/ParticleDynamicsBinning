// external libraries
#include<iostream>
#include<sstream>
#include<math.h>
#include<fstream>
#include<cuda.h>
#include<string>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include<thrust/extrema.h>
#include<thrust/copy.h>
#include "../include/vs.h"

// internal includes
#include "../include/binningAlgorithm.h"
#include "../include/gearsScheme.h"
#include "../include/collisionWithWall.h"
#include "../include/forceComputation.h"
#include "../include/constants.h"

#define ERR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


using namespace std;

class Box{
	public:
	//double dt;			// dt is simulation time step size	-- defined in header file
	
	// host variables:
	double *wallX, *wallY;		// each has length 4 and stores x and y coordinates of the walls of the bin
	double *rX, *rY, *vX, *vY, *aX, *aY, *jX, *jY;		// translational parameters: have length N, rX, rY are
	// positions, vX, vY are velocities, aX, aY are accelerations, jX, jY are jerks (third derivative of position), col is color of the particle
	// 1 indicates blue and 0 indicates red. All have length N.
	int *col;
	double *phi, *omega, *alpha, *zeta;		// rotational parameters: phi is angular position (rotation), omega is angular velocity, alpha is 
	// angular acceleration, zeta is third derivative of angular position. All have length N.
	
	double *Fx, *Fy;	// Total force on particle, has length N
	double *rDotF;		// This stores the dot product of r and F as in equations of motion explained in the project description.
	double * xitParticleOld, *xitWallOld;	// xitOld stores the relative tangential velocity * dt for the collision.
	int * wallIDOld, * pIDOld;
	/* for particle consider, for particle
	0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ... <- this is index of the array
	0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4  ... <- corresponding particle indices, they are not stored as such, but computed from index of array
	0 0 0 0 0 0 a b 0 0 0 0 c d f g 0 0 k l 0 0 0 0 m 0 ...  <- value in xitParticleOld, (in beginning of simulation all the values are zero) 
	
	you will later see that pId has similar array structure. Assume pId is:
	0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 ... <- same as above
	0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3 4 4  ... <- same as above
	0 0 0 0 0 0 5 7 0 0 0 0 4 6 7 8 0 0 5 9 0 0 0 0 10 0 ...  <- particle collided with
	
	the two arrays together (pId and xitOld) tell us that particle 1 collides with 5 and 7 and their relative tangential velocity * dt is "a" and "b" respectively.
	
	xitNew will store that relative tangentialvelocity * dt in new iteration then update... method will check if the collision occured in previous iteration and then if did add
	that value to xitNew from xitOld. Then after this update, xitNew becomes xitOld, and in next iteration, xitNew is computed.
	
	*/
	
	
	// these are same variables but are for device. They are allocated in device.
	double *dev_wallX, *dev_wallY;
	double *dev_rX, *dev_rY, *dev_vX, *dev_vY, *dev_aX, *dev_aY, *dev_jX, *dev_jY;		// translational parameters
	double *dev_phi, *dev_omega, *dev_alpha, *dev_zeta, *dev_zetat;		// rotational parameters
	double *dev_Fx, *dev_Fy;
	double *dev_rDotF;
	double *dev_xitWallOld;
	double *dev_xitParticleOld;
	int *dev_wallIDOld;
	int *dev_pIDOld;

	
	
//	public:
	Box();
	void initialize();
	
	void runIteration();	
	void rotateBox();
	void computeForceFromParticleCollision();
	void computeForceFromWallCollision();
	void solveODE();
	
	void transferDataHostToDevice();
	void updateDataForVisualization();
	void transferDataDeviceToHost();
	
	void writeDataToFile(int);
	
	void freeAll();
};

Box::Box(){
	// nothing to do now
}

void Box::transferDataHostToDevice(){
		
	ERR_CHECK( cudaMemcpy(dev_rX, rX, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_rY, rY, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_vX, vX, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_vY, vY, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_aX, aX, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_aY, aY, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_jX, jX, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_jY, jY, N*sizeof(double),cudaMemcpyHostToDevice));
	
	ERR_CHECK( cudaMemcpy(dev_phi, phi, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_omega, omega, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_alpha, alpha, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_zeta, zeta, N*sizeof(double),cudaMemcpyHostToDevice));
	
	ERR_CHECK( cudaMemcpy(dev_Fx, Fx, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_Fy, Fy, N*sizeof(double),cudaMemcpyHostToDevice));
	
	ERR_CHECK( cudaMemcpy(dev_rDotF, rDotF, N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_xitWallOld, xitWallOld, 2*N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_xitParticleOld, xitParticleOld, 6*N*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_wallIDOld, wallIDOld, 2*N*sizeof(int),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_pIDOld, pIDOld, 6*N*sizeof(int),cudaMemcpyHostToDevice));
	
}

void Box::transferDataDeviceToHost(){	
	ERR_CHECK( cudaMemcpy( rX, dev_rX, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( rY, dev_rY, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( vX, dev_vX, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( vY, dev_vY, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( aX, dev_aX, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( aY, dev_aY, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( jX, dev_jX, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( jY, dev_jY, N*sizeof(double),cudaMemcpyDeviceToHost));
	
	ERR_CHECK( cudaMemcpy( phi, dev_phi, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( omega, dev_omega, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( alpha, dev_alpha, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( zeta, dev_zeta, N*sizeof(double),cudaMemcpyDeviceToHost));
	
	ERR_CHECK( cudaMemcpy( Fx, dev_Fx, N*sizeof(double),cudaMemcpyDeviceToHost));
	ERR_CHECK( cudaMemcpy( Fy, dev_Fy, N*sizeof(double),cudaMemcpyDeviceToHost));
	
	ERR_CHECK( cudaMemcpy(rDotF, dev_rDotF, N*sizeof(double),cudaMemcpyDeviceToHost));
	
	
}

void Box::updateDataForVisualization(){		// less transfer, less overhead
	cudaMemcpy( rX, dev_rX, N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy( rY, dev_rY, N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy( phi, dev_phi, N*sizeof(double),cudaMemcpyDeviceToHost);
}

void Box::freeAll(){
	
	ERR_CHECK( cudaFree(dev_wallX));
	ERR_CHECK( cudaFree(dev_wallY));
	
	ERR_CHECK( cudaFree(dev_rX));
	ERR_CHECK( cudaFree(dev_rY));
	ERR_CHECK( cudaFree(dev_vX));
	ERR_CHECK( cudaFree(dev_vY));
	ERR_CHECK( cudaFree(dev_aX));
	ERR_CHECK( cudaFree(dev_aY));
	ERR_CHECK( cudaFree(dev_jX));
	ERR_CHECK( cudaFree(dev_jY));
	
	ERR_CHECK( cudaFree(dev_omega));
	ERR_CHECK( cudaFree(dev_phi));
	ERR_CHECK( cudaFree(dev_alpha));
	ERR_CHECK( cudaFree(dev_zeta));
	
	ERR_CHECK( cudaFree(dev_zeta));
	ERR_CHECK( cudaFree(dev_zeta));
	
	ERR_CHECK( cudaFree(dev_rDotF));
	
	ERR_CHECK( cudaFree(dev_xitWallOld));
	ERR_CHECK( cudaFree(dev_xitParticleOld));
	
	ERR_CHECK( cudaFree(dev_wallIDOld));
	ERR_CHECK( cudaFree(dev_pIDOld));
	
}

void Box::initialize(){

	/* Corners of the Box */

	wallX = new double[4]();
	wallX[0] = -0.05;
	wallX[1] = -0.05;
	wallX[2] = 48.1;
	wallX[3] = 48.1;
	wallY = new double[4]();
	wallY[0] = -0.05;
	wallY[1] = 48.1;
	wallY[2] = 48.1;
	wallY[3] = -0.05;
	
	/* Positions of the Particles */
	rX = new double[N]();
	rY = new double[N]();
        std::cout<<"before radious";
	// initialize 115200 particles

	for(int i = 0; i < 240; i++){
		double y = 1 + i * ht;			// ht is defined in constants
		double shift = 0.05+(i%2)*0.05;		// the packing is hexagonal, so non trivial
		for(int j =0; j< 480;j++){			// see the file hexagonalPacking.png in 
			double x = shift+j*0.1;
			rX[j + i * 480] = x;
			rY[j + i * 480] = y;
		}
	}
	
	// place particle 0 outside the box so that it does not collide with any particles

	rX[0] = -20.;
	rY[0] = -20.;
	
	vX 		= new double[N]();
	vY 		= new double[N]();
	
	aX 		= new double[N]();
	aY 		= new double[N]();
	jX 		= new double[N]();
	jY 		= new double[N]();
	phi 		= new double[N]();
	omega 		= new double[N]();
	alpha 		= new double[N]();
	zeta 		= new double[N]();
	Fx 		= new double[N]();
	Fy 		= new double[N]();

	col		= new int[N]();
	thrust::fill(col,col+(N/2),1);

	rDotF 	= new double[N]();
	xitWallOld	= new double[2*N]();
	xitParticleOld	= new double[6*N]();
	wallIDOld	= new int[2*N]();
	pIDOld	= new int[6*N]();
	
	ERR_CHECK( cudaMalloc( (void**)&dev_wallX, 4 * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_wallY, 4 * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_rX, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_rY, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_vX, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_vY, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_aX, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_aY, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_jX, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_jY, N * sizeof(double) ));
	
	ERR_CHECK( cudaMalloc( (void**)&dev_phi, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_omega, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_alpha, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_zeta, N * sizeof(double) ));
	
	ERR_CHECK( cudaMalloc( (void**)&dev_Fx, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_Fy, N * sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_rDotF, N * sizeof(double) ));
	
	ERR_CHECK( cudaMalloc( (void**)&dev_wallIDOld, 2*N * sizeof(int) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_pIDOld, 6*N * sizeof(int) ));
	
	ERR_CHECK( cudaMalloc( (void**)&dev_xitWallOld, 2 *N* sizeof(double) ));
	ERR_CHECK( cudaMalloc( (void**)&dev_xitParticleOld, 6 *N* sizeof(double) ));

}

void Box::rotateBox(){
	
	double cTheta = cos(boxOmega * dt);
	double sTheta = sin(boxOmega * dt);
	
	for(int i = 0; i<4; i++){
	
		double xOld = wallX[i];
		double yOld = wallY[i];
		/* update corner values */
		double a = cTheta*(xOld - binCenterX) - sTheta*(yOld - binCenterY)+binCenterX;
		double b = sTheta*(xOld - binCenterX) + cTheta*(yOld - binCenterY)+binCenterY;
		wallX[i] = a;
		wallY[i] = b;
		
	}
	ERR_CHECK( cudaMemcpy(dev_wallX, wallX, 4*sizeof(double),cudaMemcpyHostToDevice));
	ERR_CHECK( cudaMemcpy(dev_wallY, wallY, 4*sizeof(double),cudaMemcpyHostToDevice));
}

void Box::computeForceFromWallCollision(){

	thrust::device_vector<int> dev_wallIDNewV(2*N);		// this is temporary vector that has to be initialized to zero, so it is declared locally. thrust::device is used because it automatically initializes all values to zero which means, I do not need to initialize in cpu and transfer to device.
	int * dev_wallIDNew = thrust::raw_pointer_cast(&dev_wallIDNewV[0]);	// pointer has to be obtained this way to send the pointer through kernel

	
	ERR_CHECK( cudaDeviceSynchronize() );

	detectCollisionWithWall<<<(N+127)/128,128>>>(dev_rX, dev_rY, dev_wallX, dev_wallY, dev_wallIDNew);	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
	thrust::device_vector<double> dev_xitWallNewV(2*N);
	double * dev_xitWallNew = thrust::raw_pointer_cast(&dev_xitWallNewV[0]);
	
	
	computeXitWall<<<(2*N+31)/32,32>>>(dev_wallIDNew, dev_rX, dev_rY, dev_vX, dev_vY, dev_omega, dev_xitWallNew, dev_wallX, dev_wallY);	// 2N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );

	
	updateXitWall<<<(N+31)/32,32>>>(dev_wallIDOld, dev_xitWallOld, dev_wallIDNew, dev_xitWallNew);	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );


	updateForceFromWallCollision<<<(2*N+31)/32,32>>>(dev_wallIDNew, dev_wallX, dev_wallY, dev_xitWallNew, dev_rX, dev_rY, dev_vX, dev_vY, dev_Fx, dev_Fy, dev_rDotF);	// 2N threads
	ERR_CHECK( cudaDeviceSynchronize() );
	ERR_CHECK( cudaPeekAtLastError() );


	thrust::device_ptr<double> dev_xitWptr = thrust::device_pointer_cast(dev_xitWallOld);
	thrust::copy(dev_xitWallNew,dev_xitWallNew+(2*N),dev_xitWptr);
	thrust::device_ptr<int> dev_wIDptr = thrust::device_pointer_cast(dev_wallIDOld);
	thrust::copy(dev_wallIDNew,dev_wallIDNew+(2*N),dev_wIDptr);

	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );

}

void Box::computeForceFromParticleCollision(){
	
	
	thrust::device_vector<int> dev_binLocationV(N);	// this part is same as in updateForceFromWallCollision. Understanding that part will make this part clear.
	int * dev_binLocation = thrust::raw_pointer_cast(&dev_binLocationV[0]);
	
	locateBin<<<(N+31)/32, 32>>>(dev_binLocation, dev_rX, dev_rY);	// see the source code and that will make it clear. // N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
	
	thrust::device_vector<int> dev_pInBinV(2*nBinAlongX*nBinAlongY);
	int * dev_pInBin = thrust::raw_pointer_cast(&dev_pInBinV[0]);
	
	assignParticleToBin1<<<(N+31)/32,32>>>(dev_binLocation, dev_pInBin);	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
	assignParticleToBin2<<<(N+31)/32, 32>>>(dev_binLocation, dev_pInBin);	// N threads

	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );

	thrust::device_vector<int> dev_pIDNewV(6*N);
	int * dev_pIDNew = thrust::raw_pointer_cast(&dev_pIDNewV[0]);
	detectCollisionBtwnParticles<<<((nBinAlongX-1)*nBinAlongY+63)/64, 64>>>(dev_pInBin, dev_pIDNew,dev_rX, dev_rY);		// N threads

	ERR_CHECK( cudaPeekAtLastError() );	
	ERR_CHECK( cudaDeviceSynchronize() );

	thrust::device_vector<double> dev_xitParticleNewV(6*N);
	double * dev_xitParticleNew = thrust::raw_pointer_cast(&dev_xitParticleNewV[0]);

	computeXitParticle<<<(6*N+31)/32,32>>>(dev_pIDNew, dev_rX, dev_rY, dev_vX, dev_vY, dev_omega, dev_xitParticleNew);	// 6N threads
	ERR_CHECK( cudaPeekAtLastError() );	
	ERR_CHECK( cudaDeviceSynchronize() );
	
	updateXitParticle<<<(N+31)/32,32>>>( dev_pIDOld, dev_xitParticleOld, dev_pIDNew, dev_xitParticleNew); 	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
	updateForceFromParticleCollision<<<(6*N+31)/32,32>>>(dev_pIDNew, dev_xitParticleNew, dev_rX, dev_rY, dev_vX, dev_vY, dev_Fx, dev_Fy, dev_rDotF);//6*N
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
	thrust::device_ptr<double> dev_xitPptr = thrust::device_pointer_cast(dev_xitParticleOld);
	thrust::copy(dev_xitParticleNew,dev_xitParticleNew+(6*N),dev_xitPptr);
	thrust::device_ptr<int> dev_pIDptr = thrust::device_pointer_cast(dev_pIDOld);
	thrust::copy(dev_pIDNew,dev_pIDNew+(6*N),dev_pIDptr);
	
}

// this may not need much changes
void Box::solveODE(){

	gearTranslationSolver<<<(N+31)/32,32>>>(dev_rX, dev_rY, dev_vX, dev_vY, dev_aX, dev_aY, dev_jX, dev_jY, dev_Fx, dev_Fy); 	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );

	gearRotationSolver<<<(N+31)/32,32>>>(dev_rX, dev_rY, dev_vX, dev_vY, dev_aX, dev_aY, dev_jX, dev_jY, dev_Fx, dev_Fy);	// N threads
	ERR_CHECK( cudaPeekAtLastError() );
	ERR_CHECK( cudaDeviceSynchronize() );
	
}

void Box::runIteration(){

	// reset force values
	thrust::device_ptr<double> dev_Fxptr=thrust::device_pointer_cast(dev_Fx);
	thrust::fill(dev_Fxptr, dev_Fxptr+N, 0.);
	thrust::device_ptr<double> dev_Fyptr=thrust::device_pointer_cast(dev_Fy);
	thrust::fill(dev_Fyptr, dev_Fyptr+N, 0.);

	rotateBox();

	computeForceFromWallCollision();

	computeForceFromParticleCollision();
	
	solveODE();

	ERR_CHECK( cudaDeviceSynchronize() );	
}


void Box::writeDataToFile(int i){

	updateDataForVisualization();

	std::stringstream sstrm1;	// generate file name
	sstrm1<<"output2/particleData"<<i<<".csv";
	string fileName1 = sstrm1.str();
	ofstream file1;
	file1.open(fileName1.c_str());
	file1<<"x,y,z,phi,col"<<std::endl;
	for(int j = 1; j< N; j++){		// 0th particle is ghost that we don't need.
		// avoid writing nans to file.
		if(rX[j]==rX[j] && rY[j] == rY[j] && phi[j]==phi[j]){
			file1<<rX[j]<<","<<rY[j]<<","<<0<<","<<phi[j]<<","<<col[j]<<std::endl;
		}
	}
	file1.close();

	std::stringstream sstrm2;	// generate file name
	sstrm2<<"output2/wallData"<<i<<".csv";
	string fileName2 = sstrm2.str();
	ofstream file2;

	file2.open(fileName2.c_str());
	file2<<"x,y,z"<<std::endl;
	for(int j = 0; j< 4; j++){
		file2<<wallX[j]<<","<<wallY[j]<<","<<0<<std::endl;
	}


	file2.close();
}

int main(){
	
	const int iter = 4000000;			// number of iterations
	
	Box bx;
	std::cout<<"b4 initial";
	bx.initialize();
	
	ofstream resMatlab("res.csv");

	ERR_CHECK( cudaDeviceSynchronize() );
	std::cout<<"after initialization";
	bx.transferDataHostToDevice();

	ERR_CHECK( cudaDeviceSynchronize() );

	for(int i = 0; i < iter; i++){
		
		std::cout<<i<<std::endl;
		bx.runIteration();
		double *nRX = new double[N]();
		double *nRY = new double[N]();
		
		ERR_CHECK( cudaMemcpy(nRX, bx.dev_rX, N*sizeof(double),cudaMemcpyDeviceToHost));
		ERR_CHECK( cudaMemcpy(nRY, bx.dev_rY, N*sizeof(double),cudaMemcpyDeviceToHost));
		
		
		
		if (i % 10000 == 0) {
			resMatlab<<i<<",";
			for (int j = 1; j<N; j+=50) {
				resMatlab << nRX[j] << "," << nRY[j] << ",";
			}
		}

		resMatlab <<std::endl;
		
		delete [] nRX;
		delete [] nRY;
		
		ERR_CHECK(cudaDeviceSynchronize());	
		if(i%10000==0){	// transfer only once every 1000 iterations
			bx.writeDataToFile(i/10000);
		}

	}
	
	bx.freeAll();
	ERR_CHECK( cudaDeviceSynchronize() );
	resMatlab.close();
	
	return 0;
}