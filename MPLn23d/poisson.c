#include "MPLn23d.h"

static double Gammln(double xx)
{
	double x, y, tmp, ser;
	static double cof[6] = { 76.18009172947146, -86.50532032941677,
		24.01409824083091, -1.231739572450155,
		0.1208650973866179e-2, -0.5395239384953e-5 };
	int j;

	y = x = xx;
	tmp = x + 5.5;
	tmp -= (x + 0.5)*log(tmp);
	ser = 1.000000000190015;
	for (j = 0; j <= 5; j++) ser += cof[j] / ++y;
	return -tmp + log(2.5066282746310005*ser / x);
}

static double Digamma(double x)
{
	static	double	brn[] = { 1.6666666666666666e-01, 3.3333333333333333e-02,
		2.3809523809523809e-02, 3.3333333333333333e-02, 7.5757575757575757e-02,
		2.5311355311355311e-01, 1.1666666666666667e+00, 7.0921568627450980e+00,
		5.4971177944862155e+01, 5.2912424242424242e+02 };
	int	n;
	int	i;
	int	i2, isgn;
	double	s;
	double	y;
	double	x2;
	double	*bp;
	double	slv = 13.06;

	if (x >= slv) {
		s = 0.0;
		x2 = x*x;
		bp = &brn[9];
		isgn = 1;
		for (i = 9; i >= 0; i--){
			i2 = 2 * (i + 1);
			s += *bp-- / (double)i2 * isgn;
			s /= x2;
			isgn *= -1;
		}
		s += log(x) - 0.5 / x;
	}
	else {
		n = (int)(slv - x);
		y = (double)n + x + 1.0;
		s = Digamma(y);
		isgn = -1;
		for (i = 0; i <= n; i++){
			y -= 1.0;
			if (fabs(y) < 1.e-3){
				if (x > 0)
					y = x - (double)((int)(x + 0.5));
				else
					y = x - (double)((int)(x - 0.5));
			}
			s += isgn / y;
		}
	}
	return s;
}

double MP_Poisson(double a, double b, double x)
{
	return pow(a, x - b) * exp(-a - Gammln(x - b + 1));
}
