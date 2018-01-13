#include "MPLn23d.h"
#include <time.h>

int MP_Ln2dAlloc(MP_Ln2dData *data, int nsec_max)
{
	data->sec = (MP_Ln2dSection *) malloc(nsec_max*sizeof(MP_Ln2dSection));
	if (data->sec == NULL) {
		fprintf(stderr, "Error : allocation failure (MP_Ln2dAlloc)\n");
		return FALSE;
	}
	data->nsec = 0;
	data->nsec_max = nsec_max;
	data->seed = (long)time(NULL);
	return TRUE;
}

void MP_Ln2dFree(MP_Ln2dData *data)
{
	int i;

	for (i = 0;i < data->nsec;i++) {
		if (data->sec[i].ngc_max > 0) {
			free(data->sec[i].x);
			free(data->sec[i].y);
			free(data->sec[i].r);
		}
	}
	free(data->sec);
}

MP_Ln2dSection *MP_Ln2dAddSection(MP_Ln2dData *data, int ngc_step, double sx, double sy)
{
	MP_Ln2dSection *sec;

	if (data->nsec >= data->nsec_max) {
		fprintf(stderr, "Error : can't add section (MP_Ln2dAddSection)\n");
		return NULL;
	}
	sec = &(data->sec[data->nsec++]);
	sec->ngc_max = sec->ngc_step = ngc_step;
	sec->x = (double *) malloc(sec->ngc_max*sizeof(double));
	sec->y = (double *) malloc(sec->ngc_max*sizeof(double));
	sec->r = (double *) malloc(sec->ngc_max*sizeof(double));
	if (sec->x == NULL || sec->y == NULL || sec->r == NULL) {
		fprintf(stderr, "Error : allocation failure (MP_Ln2dAddSection)\n");
		return NULL;
	}
	sec->size[0] = sx, sec->size[1] = sy;
	sec->ngc = 0;
	return sec;
}

int MP_Ln2dAddGc(MP_Ln2dSection *sec, double x, double y, double r)
{
	if (sec->ngc >= sec->ngc_max) {
		sec->ngc_max += sec->ngc_step;
		sec->x = (double *) realloc(sec->x, sec->ngc_max*sizeof(double));
		sec->y = (double *) realloc(sec->y, sec->ngc_max*sizeof(double));
		sec->r = (double *) realloc(sec->r, sec->ngc_max*sizeof(double));
		if (sec->x == NULL || sec->y == NULL || sec->r == NULL) {
			fprintf(stderr, "Error : allocation failure (MP_Ln2dAddGc)\n");
			return MP_ALLOC_FAILED;
		}
	}
	if (x < 0.0 || x >= sec->size[0]
		|| y < 0.0 || y >= sec->size[1]) return MP_OUT_OF_REGION;
	sec->x[sec->ngc] = x;
	sec->y[sec->ngc] = y;
	sec->r[sec->ngc] = r;
	return ++sec->ngc;
}

static double RandGaussSize(double size, double sd, long *seed)
{
	double v;

	while (TRUE) {
		v = size*(0.5 + sd*MP_RandGauss(seed));
		if (v >= 0.0 && v < size) return v;
	}
}

int MP_Ln2dAddGcRandom(MP_Ln2dSection *sec, int ngc, double sd, double r, long *seed)
{
	int count;
	double x, y;
	int ret;

	count = 0;
	while (count < ngc) {
		if (sd > 0.0) {
			x = RandGaussSize(sec->size[0], sd, seed);
			y = RandGaussSize(sec->size[1], sd, seed);
		}
		else {
			x = sec->size[0] * MP_Rand(seed);
			y = sec->size[1] * MP_Rand(seed);
		}
		ret = MP_Ln2dAddGc(sec, x, y, r);
		if (ret == MP_ALLOC_FAILED) return ret;
		else if (ret != MP_OUT_OF_REGION) count++;
	}
	return sec->ngc;
}

static double radius2d(MP_Ln2dData *data)
{
	int i;
	int tpnum = 0;
	double tarea = 0.0;
	double lama;

	for (i = 0;i < data->nsec;i++) {
		tarea += data->sec[i].size[0]*data->sec[i].size[1];
		tpnum += data->sec[i].ngc;
	}
	lama = (double)tpnum/tarea;
	return sqrt(7.0/M_PI/lama);
}

void MP_Ln2dMeasureGc(MP_Ln2dData *data, int nclass, unsigned int freq[])
{
	int i, j, nn;
	MP_Neigh2dData neigh2d;
	MP_Ln2dSection *sec;
	double r2d = radius2d(data);

	MP_Neigh2dInit(&neigh2d);
	for (i = 0; i < data->nsec; i++) {
		sec = &(data->sec[i]);
		if (!MP_Neigh2dDivide(&neigh2d, r2d, sec->ngc,
			sec->size, sec->x, sec->y)) return;
		for (j = 0; j < sec->ngc; j++) {
			nn = MP_Neigh2dNumber(&neigh2d, r2d, sec->size,
				sec->x[j], sec->y[j], sec->x, sec->y);
			if (nn < nclass) freq[nn]++;
		}
	}
	MP_Neigh2dFree(&neigh2d);
}

void MP_Ln2dMeasureRandom(MP_Ln2dData *data, int nclass, unsigned int freq[], int nsample)
{
	int i, n, nn, nmax;
	MP_Neigh2dData neigh2d;
	MP_Ln2dSection *sec;
	double tarea = 0.0;;
	double r2d = radius2d(data);
	double x, y;

	for (i = 0; i < data->nsec; i++) {
		tarea += data->sec[i].size[0] * data->sec[i].size[1];
	}
	MP_Neigh2dInit(&neigh2d);
	for (i = 0; i < data->nsec; i++) {
		sec = &(data->sec[i]);
		nmax = (int)(nsample * sec->size[0] * sec->size[1] / tarea);
		if (!MP_Neigh2dDivide(&neigh2d, r2d, sec->ngc,
			sec->size, sec->x, sec->y)) return;
		n = 0;
		while (n < nmax) {
			x = sec->size[0] * MP_Rand(&(data->seed));
			y = sec->size[1] * MP_Rand(&(data->seed));
			if (x >= 0.0 && x < sec->size[0]
				&& y >= 0.0 && y < sec->size[1]) {
				nn = MP_Neigh2dNumber(&neigh2d, r2d, sec->size,
					x, y, sec->x, sec->y);
				if (nn < nclass) freq[nn]++;
				n++;
			}
		}
	}
	MP_Neigh2dFree(&neigh2d);
}

double MP_Ln2dAreaFraction(MP_Ln2dData *data)
{
	int i, j;
	MP_Ln2dSection *sec;
	double total = 0.0;
	double area = 0.0;	

	if (data->nsec <= 0) return -1.0;
	for (i = 0; i < data->nsec; i++) {
		sec = &(data->sec[i]);
		total += sec->size[0] * sec->size[1];
		for (j = 0; j < sec->ngc; j++) {
			area += M_PI * sec->r[j] * sec->r[j];
		}
	}
	return area / total;
}
