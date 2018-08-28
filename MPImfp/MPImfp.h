#ifndef _MPIMFP_H
#define _MPIMFP_H

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*--------------------------------------------------
*  Imfp functions
*/
void MP_ImfpMeasure(unsigned char *data, int step, int width, int height,
	unsigned char barrier, int nclass, unsigned int freq[], double dpix, int nsample, long *seed, int dflag);

/*--------------------------------------------------
*  Rand functions
*/
float MP_Rand(long *rand_seed);
float MP_RandGauss(long *rand_seed);

#ifdef __cplusplus
}
#endif

#endif /* _MPIMFP_H */
