#include "MPLn23d.h"

void MP_Neigh3dInit(MP_Neigh3dData *data)
{
	data->link_max = 0;
	data->link_top_max = 0;
}

void MP_Neigh3dFree(MP_Neigh3dData *data)
{
	free(data->link);
	free(data->link_top);
}

int MP_Neigh3dDivide(MP_Neigh3dData *data, double rcut,
	int num, double size[], double x[], double y[], double z[])
{
	int i, j;
	int c[3], cp;
	int link_top_num;

	if (size[0] == 0.0 || size[1] == 0.0 || size[2] == 0.0) {
		fprintf(stderr, "Error : The size is zero (MP_Neigh3dDivide)\n");
		return FALSE;
	}
	if (num >= data->link_max) {
		if (data->link_max == 0) {
			data->link = (int *)malloc(num*sizeof(int));
		}
		else {
			data->link = (int *)realloc(data->link, num*sizeof(int));
		}
		if (data->link == NULL) {
			fprintf(stderr, "Error : allocation failure (MP_Neigh3dDivide)\n");
			return FALSE;
		}
		data->link_max = num;
	}
	data->num_lc[0] = (int)(size[0] / rcut);
	data->num_lc[1] = (int)(size[1] / rcut);
	data->num_lc[2] = (int)(size[2] / rcut);
	if (data->num_lc[0] == 0) data->num_lc[0] = 1;
	if (data->num_lc[1] == 0) data->num_lc[1] = 1;
	if (data->num_lc[2] == 0) data->num_lc[2] = 1;
	data->len_lc[0] = size[0] / data->num_lc[0];
	data->len_lc[1] = size[1] / data->num_lc[1];
	data->len_lc[2] = size[2] / data->num_lc[2];
	link_top_num = data->num_lc[0] * data->num_lc[1] * data->num_lc[2];
	if (link_top_num >= data->link_top_max) {
		if (data->link_top_max == 0) {
			data->link_top = (int *)malloc(link_top_num*sizeof(int));
		}
		else {
			data->link_top = (int *)realloc(data->link_top, link_top_num*sizeof(int));
		}
		if (data->link_top == NULL) {
			fprintf(stderr, "Error : allocation failure (MD_Neigh3dDivide)\n");
			return FALSE;
		}
		data->link_top_max = link_top_num;
	}
	for (cp = 0;cp < link_top_num;cp++) {
		data->link_top[cp] = MP_NEIGH_LINKEND;
	}
	for (i = 0;i < num;i++) {
		c[0] = (int)(x[i]/data->len_lc[0]);
		c[1] = (int)(y[i]/data->len_lc[1]);
		c[2] = (int)(z[i]/data->len_lc[2]);
		cp = c[0] + c[1]*data->num_lc[0] + c[2]*data->num_lc[0]*data->num_lc[1];
		j = data->link_top[cp];
		data->link_top[cp] = i;
		data->link[i] = j;
	}
	return TRUE;
}

int MP_Neigh3dNumber(MP_Neigh3dData *data, double rcut, double size[],
	double cx, double cy, double cz, double x[], double y[], double z[])
{
	int nn = 0;
	int j;
	int c[3], cp;
	int nc[3], ncp;
	double bl[3];
	double rcut2, dx, dy, dz, dr2;
	int neigh[27][3] = {
		{0,0,0}, {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {1,1,0},
		{1,-1,0}, {-1,1,0}, {-1,-1,0}, {0,0,1}, {1,0,1}, {-1,0,1},
		{0,1,1}, {0,-1,1}, {1,1,1}, {1,-1,1}, {-1,1,1}, {-1,-1,1},
		{0,0,-1}, {1,0,-1}, {-1,0,-1}, {0,1,-1}, {0,-1,-1}, {1,1,-1},
		{1,-1,-1}, {-1,1,-1}, {-1,-1,-1}};
		
	rcut2 = rcut*rcut;
	c[0] = (int)(cx/data->len_lc[0]);
	c[1] = (int)(cy/data->len_lc[1]);
	c[2] = (int)(cz/data->len_lc[2]);
	for (cp = 0;cp < 27;cp++) {
		for (j = 0;j < 3;j++) {
			nc[j] = c[j] + neigh[cp][j];
			bl[j] = 0.0;
			if (c[j] == 0 && nc[j] < c[j]) {
				nc[j] = data->num_lc[j]-1;
				bl[j] = -size[j];
			}
			else if (c[j] == data->num_lc[j]-1 && nc[j] > c[j]) {
				nc[j] = 0;
				bl[j] = size[j];
			}
		}
		ncp = nc[0] + nc[1]*data->num_lc[0] + nc[2]*data->num_lc[0]*data->num_lc[1];
		j = data->link_top[ncp];
		while (j != MP_NEIGH_LINKEND) {
			dx = cx - (x[j] + bl[0]);
			dy = cy - (y[j] + bl[1]);
			dz = cz - (z[j] + bl[2]);
			dr2 = dx*dx + dy*dy + dz*dz;
			if (dr2 <= rcut2) {
				nn++;
			}
			j = data->link[j];
		}
	}
	return nn;
}
