#ifndef _MPLN23D_H
#define _MPLN23D_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef FALSE
#define FALSE 0
#endif
#ifndef TRUE
#define TRUE 1
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef _DEBUG
#ifndef Py_PYTHON_H
#include <Python.h>
#endif
#ifndef Py_STRUCTMEMBER_H
#include <structmember.h>
#endif
#endif
#ifndef _INC_STDIO
#include <stdio.h>
#endif
#ifndef _INC_STDLIB
#include <stdlib.h>
#endif
#ifndef _INC_STRING
#include <string.h>
#endif
#ifndef _INC_MATH
#include <math.h>
#endif

#define MP_OUT_OF_REGION -1
#define MP_NO_CELL -2
#define MP_NO_SECTION -3
#define MP_ALLOC_FAILED -99

/*--------------------------------------------------
* ln2d typedef and functions
*/
typedef struct MP_Ln2dSection {
	double size[2];
	int ngc;
	double *x, *y, *r;
	int ngc_step;
	int ngc_max;
} MP_Ln2dSection;

typedef struct MP_Ln2dData {
#ifndef _DEBUG
	PyObject_HEAD
#endif
	int nsec;
	MP_Ln2dSection *sec;
	int nsec_max;
	long seed;
} MP_Ln2dData;

int MP_Ln2dAlloc(MP_Ln2dData *data, int nsec_max);
void MP_Ln2dFree(MP_Ln2dData *data);
MP_Ln2dSection *MP_Ln2dAddSection(MP_Ln2dData *data, int ngc_step, double sx, double sy);
int MP_Ln2dAddGc(MP_Ln2dSection *sec, double x, double y, double r);
int MP_Ln2dAddGcRandom(MP_Ln2dSection *sec, int ngc, double sd, double r, long *seed);
void MP_Ln2dMeasureGc(MP_Ln2dData *data, int nclass, unsigned int freq[]);
void MP_Ln2dMeasureRandom(MP_Ln2dData *data, int nclass, unsigned int freq[], int nsample);
double MP_Ln2dAreaFraction(MP_Ln2dData *data);

/*--------------------------------------------------
* ln3d typedef and functions
*/
typedef struct MP_Ln3dCell {
	double size[3];
	int ngc;
	double *x, *y, *z, *r;
	int ngc_step;
	int ngc_max;
} MP_Ln3dCell;

typedef struct MP_Ln3dData {
#ifndef _DEBUG
	PyObject_HEAD
#endif
	int ncell;
	MP_Ln3dCell *cell;
	int ncell_max;
	long seed;
} MP_Ln3dData;

int MP_Ln3dAlloc(MP_Ln3dData *data, int ncell_max);
void MP_Ln3dFree(MP_Ln3dData *data);
MP_Ln3dCell *MP_Ln3dAddCell(MP_Ln3dData *data, int ngc_step, double sx, double sy, double sz);
int MP_Ln3dAddGc(MP_Ln3dCell *cell, double x, double y, double z, double r);
int MP_Ln3dAddGcRandom(MP_Ln3dCell *cell, int ngc, double sd, double r, long *seed);
void MP_Ln3dMeasureGc(MP_Ln3dData *data, int nclass, unsigned int freq[]);
void MP_Ln3dMeasureRandom(MP_Ln3dData *data, int nclass, unsigned int freq[], int nsample);
double MP_Ln3dVolumeFraction(MP_Ln3dData *data);

/*--------------------------------------------------
* cut functions
*/
int MP_Ln23dCut(MP_Ln2dData *ln2d, int ngc_step, MP_Ln3dCell *cell, int dir, double pos);
int MP_Ln23dCutRandom(MP_Ln2dData *ln2d, int nsec, int ngc_step, MP_Ln3dCell *cell, long *seed);

/*--------------------------------------------------
* stat functions
*/
int MP_StatCalc(int nclass, unsigned int freq[], double rfreq[], double *ave, double *var);

/*--------------------------------------------------
* random functions
*/
float MP_Rand(long *rand_seed);
float MP_RandGauss(long *rand_seed);

/*--------------------------------------------------
* neigh2d typedef and functions
*/
#define MP_NEIGH_LINKEND -99

typedef struct MP_Neigh2dData {
	int num_lc[2];
	double len_lc[2];
	int *link;
	int link_max;
	int *link_top;
	int link_top_max;
} MP_Neigh2dData;

void MP_Neigh2dInit(MP_Neigh2dData *data);
void MP_Neigh2dFree(MP_Neigh2dData *data);
int MP_Neigh2dDivide(MP_Neigh2dData *data, double rcut,
	int num, double size[], double x[], double y[]);
int MP_Neigh2dNumber(MP_Neigh2dData *data, double rcut, double size[],
	double cx, double cy, double x[], double y[]);

/*--------------------------------------------------
* neigh3d typedef and functions
*/
typedef struct MP_Neigh3dData {
	int num_lc[3];
	double len_lc[3];
	int *link;
	int link_max;
	int *link_top;
	int link_top_max;
} MP_Neigh3dData;

void MP_Neigh3dInit(MP_Neigh3dData *data);
void MP_Neigh3dFree(MP_Neigh3dData *data);
int MP_Neigh3dDivide(MP_Neigh3dData *data, double rcut,
	int num, double size[], double x[], double y[], double z[]);
int MP_Neigh3dNumber(MP_Neigh3dData *data, double rcut, double size[],
	double cx, double cy, double cz, double x[], double y[], double z[]);

#ifdef __cplusplus
}
#endif

#endif /* _MPLN23D_H */