#include <math.h>
#include "MPImfp.h"

/*--------------------------------------------------
 * 一様乱数を発生させる。
 * rand_seed を任意の整数値に設定・再設定すれば乱数列が初期化される。
 * 同じ乱数列を使っている間は rand_seed を変えてはならない。
 * [引数]
 *  *rand_seed : 任意の整数値
 * [返り値]
 *  0.0 から 1.0 までの一様乱数
 */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876
float MP_Rand(long *rand_seed)
{
  long k;
  float ans;

  *rand_seed ^= MASK;
  k=(*rand_seed)/IQ;
  *rand_seed=IA*(*rand_seed-k*IQ)-IR*k;
  if (*rand_seed < 0) *rand_seed += IM;
  ans=(float)(AM*(*rand_seed));
  *rand_seed ^= MASK;
  return ans;
}

/*--------------------------------------------------
 * ガウス乱数を発生させる。
 * rand_seed を任意の整数値に設定・再設定すれば乱数列が初期化される。
 * 同じ乱数列を使っている間は rand_seed を変えてはならない。
 * [引数]
 *  *rand_seed : 0 以外の任意の整数値
 * [返り値]
 *  ガウス乱数
 */
float MP_RandGauss(long *rand_seed)
{
  float r1,r2;
  static float w1,w2;
  static int iset = 0;

  if (iset == 0) {
    r1 = MP_Rand(rand_seed);
    r2 = MP_Rand(rand_seed);
    w1 = (float)(sqrt(-2.0*log((double)r1))*cos(2.0*M_PI*(double)r2));
    w2 = (float)(sqrt(-2.0*log((double)r1))*sin(2.0*M_PI*(double)r2));
    iset = 1;
    return w1;
  }
  else {
    iset = 0;
    return w2;
  }
}
