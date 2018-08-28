#include <math.h>
#include "MPImfp.h"

#define abs(x)  (x>=0?(x):-(x))
#define sgn(x)  (x>=0?(1):(-1))

static double ImfpLength1(unsigned char *data, int step, int width, int height,
	unsigned char barrier, int x1, int y1, int idx, int idy)
{
	int ix, iy, ia, ib, ie;
	int bx, by;

    ia  = abs(idx);
    ib  = abs(idy);
    ix = bx = x1;
    iy = by = y1;
	if (ia >= ib) {
		ie = -abs(idy);
		while (TRUE) {
			if (data[step * iy + ix] == barrier) return sqrt((bx-x1)*(bx-x1)+(by-y1)*(by-y1));
			else if (ix <= 0 || ix >= width-1 || iy <= 0 || iy >= height-1) return -1.0;
			bx = ix, by = iy;
            ix += sgn(idx);
            ie += 2 * ib;
            if (ie >= 0) {
               iy += sgn(idy);
               ie -= 2 * ia;
			}
		}
	} else {
		ie = -abs(idx);
        while (TRUE) {
			if (data[step * iy + ix] == barrier) return sqrt((bx-x1)*(bx-x1)+(by-y1)*(by-y1));
			else if (ix <= 0 || ix >= width-1 || iy <= 0 || iy >= height-1) return -1.0;
			bx = ix, by = iy;
			iy += sgn(idy);
			ie += 2 * ia;
			if (ie >= 0) {
				ix += sgn(idx);
				ie -= 2 * ib;
			}
		}
	}
}

static double ImfpLength(unsigned char *data, int step, int width, int height, 
	unsigned char barrier, long *seed, int dflag)
{
	int x1, y1;
	int idx, idy;
	double th, len1, len2;

	x1 = (int)(width*MP_Rand(seed));
	y1 = (int)(height*MP_Rand(seed));
	if (data[step * y1 + x1] == barrier) return -1.0;
	th = 2.0*M_PI*MP_Rand(seed);
	if (width >= height) {
		idx = (int)(width*cos(th));
		idy = (int)(width*sin(th));
	}
	else {
		idx = (int)(height*cos(th));
		idy = (int)(height*sin(th));
	}
	len1 = ImfpLength1(data, step, width, height, barrier, x1, y1, idx, idy);
	if (len1 < 0.0 || !dflag) return len1;
	len2 = ImfpLength1(data, step, width, height, barrier, x1, y1, -idx, -idy);
	if (len2 < 0.0) return len2;
	else return len1+len2;
}

void MP_ImfpMeasure(unsigned char *data, int step, int width, int height,
	unsigned char barrier, int nclass, unsigned int freq[], double dpix, int nsample, long *seed, int dflag)
{
	int n = 0;
	int	cl;
	double len;

	while (n < nsample) {
		len = ImfpLength(data, step, width, height, barrier, seed, dflag);
		if (len >= 0.0) {
			cl = (int)((len + 0.5) / dpix);
			if (cl < nclass) (freq[cl])++;
			n++;
		}
	}
}
