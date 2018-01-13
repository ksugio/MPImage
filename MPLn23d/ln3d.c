#include "MPLn23d.h"
#include <time.h>

int MP_Ln3dAlloc(MP_Ln3dData *data, int ncell_max)
{
	data->cell = (MP_Ln3dCell *)malloc(ncell_max*sizeof(MP_Ln3dCell));
	if (data->cell == NULL) {
		fprintf(stderr, "Error : allocation failure (MP_Ln3dAlloc)\n");
		return FALSE;
	}
	data->ncell = 0;
	data->ncell_max = ncell_max;
	data->seed = (long)time(NULL);
	return TRUE;
}

void MP_Ln3dFree(MP_Ln3dData *data)
{
	int i;

	for (i = 0; i < data->ncell; i++) {
		if (data->cell[i].ngc_max > 0) {
			free(data->cell[i].x);
			free(data->cell[i].y);
			free(data->cell[i].z);
			free(data->cell[i].r);
		}
	}
	free(data->cell);
}

MP_Ln3dCell *MP_Ln3dAddCell(MP_Ln3dData *data, int ngc_step, double sx, double sy, double sz)
{
	MP_Ln3dCell *cell;

	if (data->ncell >= data->ncell_max) {
		fprintf(stderr, "Error : can't add cell (MP_Ln3dAddCell)\n");
		return NULL;
	}
	cell = &(data->cell[data->ncell++]);
	cell->ngc_max = cell->ngc_step = ngc_step;
	cell->x = (double *)malloc(cell->ngc_max*sizeof(double));
	cell->y = (double *)malloc(cell->ngc_max*sizeof(double));
	cell->z = (double *)malloc(cell->ngc_max*sizeof(double));
	cell->r = (double *)malloc(cell->ngc_max*sizeof(double));
	if (cell->x == NULL || cell->y == NULL || cell->z == NULL || cell->r == NULL) {
		fprintf(stderr, "Error : allocation failure (MP_Ln3dAddCell)\n");
		return NULL;
	}
	cell->size[0] = sx, cell->size[1] = sy, cell->size[2] = sz;
	cell->ngc = 0;
	return cell;
}

int MP_Ln3dAddGc(MP_Ln3dCell *cell, double x, double y, double z, double r)
{
	if (cell->ngc >= cell->ngc_max) {
		cell->ngc_max += cell->ngc_step;
		cell->x = (double *)realloc(cell->x, cell->ngc_max*sizeof(double));
		cell->y = (double *)realloc(cell->y, cell->ngc_max*sizeof(double));
		cell->z = (double *)realloc(cell->z, cell->ngc_max*sizeof(double));
		cell->r = (double *)realloc(cell->r, cell->ngc_max*sizeof(double));
		if (cell->x == NULL || cell->y == NULL || cell->z == NULL || cell->r == NULL)
		{
			fprintf(stderr, "Error : allocation failure (MD_Ln3dAddGc)\n");
			return MP_ALLOC_FAILED;
		}
	}
	if (x < 0.0 || x >= cell->size[0]
		|| y < 0.0 || y >= cell->size[1]
		|| z < 0.0 || z >= cell->size[2]) return MP_OUT_OF_REGION;
	cell->x[cell->ngc] = x;
	cell->y[cell->ngc] = y;
	cell->z[cell->ngc] = z;
	cell->r[cell->ngc] = r;
	return ++cell->ngc;
}

static double RandGaussSize(double size, double sd, long *seed)
{
	double v;

	while (TRUE) {
		v = size*(0.5+sd*MP_RandGauss(seed));
		if (v >= 0.0 && v < size) return v;
	}
}

int MP_Ln3dAddGcRandom(MP_Ln3dCell *cell, int ngc, double sd, double r, long *seed)
{
	int count;
	double x, y, z;
	int ret;

	count = 0;
	while (count < ngc) {
		if (sd > 0.0) {
			x = RandGaussSize(cell->size[0], sd, seed);
			y = RandGaussSize(cell->size[1], sd, seed);
			z = RandGaussSize(cell->size[2], sd, seed);
		}
		else {
			x = cell->size[0] * MP_Rand(seed);
			y = cell->size[1] * MP_Rand(seed);
			z = cell->size[2] * MP_Rand(seed);
		}
		ret = MP_Ln3dAddGc(cell, x, y, z, r);
		if (ret == MP_ALLOC_FAILED) return ret;
		else if (ret != MP_OUT_OF_REGION) count++;
	}
	return cell->ngc;
}

static double radius3d(MP_Ln3dData *data)
{
	int i;
	int tpnum = 0;
	double tvol = 0.0;
	double lamv;

	for (i = 0; i < data->ncell; i++) {
		tvol += data->cell[i].size[0] * data->cell[i].size[1] * data->cell[i].size[2];
		tpnum += data->cell[i].ngc;
	}
	lamv = (double)tpnum / tvol;
	return pow(9.75 / M_PI / lamv, 1.0 / 3.0);
}

void MP_Ln3dMeasureGc(MP_Ln3dData *data, int nclass, unsigned int freq[])
{
	int i, j, nn;
	MP_Neigh3dData neigh3d;
	MP_Ln3dCell *cell;
	double r3d = radius3d(data);

	MP_Neigh3dInit(&neigh3d);
	for (i = 0; i < data->ncell; i++) {
		cell = &(data->cell[i]);
		if (!MP_Neigh3dDivide(&neigh3d, r3d, cell->ngc,
			cell->size, cell->x, cell->y, cell->z)) return;
		for (j = 0; j < cell->ngc; j++) {
			nn = MP_Neigh3dNumber(&neigh3d, r3d, cell->size,
				cell->x[j], cell->y[j], cell->z[j], cell->x, cell->y, cell->z);
			if (nn < nclass) freq[nn]++;
		}
	}
	MP_Neigh3dFree(&neigh3d);
}

void MP_Ln3dMeasureRandom(MP_Ln3dData *data, int nclass, unsigned int freq[], int nsample)
{
	int i, n, nn, nmax;
	MP_Neigh3dData neigh3d;
	MP_Ln3dCell *cell;
	double tvol = 0.0;
	double r3d = radius3d(data);
	double x, y, z;

	for (i = 0; i < data->ncell; i++) {
		tvol += data->cell[i].size[0] * data->cell[i].size[1] * data->cell[i].size[2];
	}
	MP_Neigh3dInit(&neigh3d);
	for (i = 0; i < data->ncell; i++) {
		cell = &(data->cell[i]);
		nmax = (int)(nsample * cell->size[0] * cell->size[1] * cell->size[2] / tvol);
		if (!MP_Neigh3dDivide(&neigh3d, r3d, cell->ngc,
			cell->size, cell->x, cell->y, cell->z)) return;
		n = 0;
		while (n < nmax) {
			x = cell->size[0] * MP_Rand(&(data->seed));
			y = cell->size[1] * MP_Rand(&(data->seed));
			z = cell->size[2] * MP_Rand(&(data->seed));
			if (x >= 0.0 && x < cell->size[0]
				&& y >= 0.0 && y < cell->size[1]
				&& z >= 0.0 && z < cell->size[2]) {
				nn = MP_Neigh3dNumber(&neigh3d, r3d, cell->size,
					x, y, z, cell->x, cell->y, cell->z);
				if (nn < nclass) freq[nn]++;
				n++;
			}
		}
	}
	MP_Neigh3dFree(&neigh3d);
}

double MP_Ln3dVolumeFraction(MP_Ln3dData *data)
{
	int i, j;
	MP_Ln3dCell *cell;
	double total = 0.0;
	double vol = 0.0;

	if (data->ncell <= 0) return -1.0;
	for (i = 0; i < data->ncell; i++) {
		cell = &(data->cell[i]);
		total += cell->size[0] * cell->size[1] * cell->size[2];
		for (j = 0; j < cell->ngc; j++) {
			vol += 4.0 / 3.0 * M_PI * cell->r[j] * cell->r[j] * cell->r[j];
		}
	}
	return vol / total;
}
