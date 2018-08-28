#include "MPLn23d.h"

int MP_StatCalc(int nclass, unsigned int freq[], double rfreq[], double *ave, double *var)
{
	int i;
	int total;

	total = 0;
	for (i = 0; i < nclass; i++) {
		total += freq[i];
	}
	for (i = 0; i < nclass; i++) {
		rfreq[i] = (double)freq[i]/(double)total;
	}
	*ave = 0.0;
	for (i = 0; i < nclass; i++) {
		*ave += (double)i*rfreq[i];
	}
	*var = 0.0;
	for (i = 0; i < nclass; i++) {
		*var += pow((double)i-*ave, 2.0)*rfreq[i];
	}
	return total;
}