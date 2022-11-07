#ifndef CONSTANTS_H
#define CONSTANTS_H
//#include<limits>

#define gx 0.	/* acceleration due to gravity */
#define gy -9.80665

/* properties of the box */
#define dt 0.000001		/* time step size */
#define boxOmega .2		/* angular velocity of the box */
#define N 115200		/* total number of particles */
//#define N 57601
//#define N 481
/* properties of the particles */
#define R .04		/* radius of the ball */
#define sqrtR 0.22360679775		/* sqrt(R): precomputed so the our code is faster */
#define m 4.2149701468		/* mass of the particle */
#define Gamma 0.001 /* damping coefficient */
#define G 793000000.0	/* shear modulus */
#define E 2000000000.0	/* Young's modulus */
#define I 0.00526871268	/* Moment due to intertia: 1/2mr^2 */
#define mu 0.4		/* coefficient of static friction */

/* constants for Gear Predictor Corrector Scheme for third order solver */
#define c0 .16666666666666666
#define c1 .83333333333333333
#define c2 1.
#define c3 .33333333333333333

/* bin constants */
/*
 __ __ __ __ __ __
|__|__|__|__|__|__|
|__|__|__|__|__|__|
|__|__|__|__|__|__|
|__|__|__|__|__|__|
|__|__|__|__|__|__|
|__|__|__|__|__|__|

so to compute which bin a particle lies in we need a grid like
above, binXMin and binYMax are absolute coordinates of the above grid.
nBinAlongX and nBinAlong Y indicate the number of squares along x and y axis: 6*6 in above case.
grid sizeX and gridsizeY indicate the length and width of the small squares in above grid. This is same as the size of the particle.
So, all the double values below are properties of such a bin/grid.
*/


#define binXMin -9.975
#define binYMin -9.975
#define binXMax 58.025
#define binYmax 58.025
#define nBinAlongX 680
#define nBinAlongY 680
#define binCenterX 24.025
#define binCenterY 24.025
#define gridSizeX 0.1
#define gridSizeY 0.1
#define ht 0.08660254037

/* other constants */
#define maxCollisionPerParticle 6
#define EPS 0.000000001
//#define eps std::numeric_limits<double>::epsilon();

#endif
