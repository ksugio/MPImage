#include "MPLn23d.h"

int MP_Ln23dCut(MP_Ln2dData *ln2d, int ngc_step, MP_Ln3dCell *cell, int dir, double pos)
{
	int i;
	MP_Ln2dSection *sec;
	double sx, sy;
	double dis;
	double x, y, r;

	if (dir == 0) {
		sx = cell->size[1], sy = cell->size[2];
	}
	else if (dir == 1) {
		sx = cell->size[0], sy = cell->size[2];
	}
	else if (dir == 2) {
		sx = cell->size[0], sy = cell->size[1];
	}
	sec = MP_Ln2dAddSection(ln2d, ngc_step, sx, sy);
	if (sec == NULL) return MP_ALLOC_FAILED;
	for (i = 0; i < cell->ngc; i++) {
		if (dir == 0) dis = pos - cell->x[i];
		else if (dir == 1) dis = pos - cell->y[i];
		else if (dir == 2) dis = pos - cell->z[i];
		if (dis > cell->size[dir] / 2.0) {
			dis -= cell->size[dir];
		}
		else if (dis < -cell->size[dir] / 2.0) {
			dis += cell->size[dir];
		}
		if (fabs(dis) < cell->r[i]) {
			if (dir == 0) {
				x = cell->y[i], y = cell->z[i];
			}
			else if (dir == 1) {
				x = cell->x[i], y = cell->z[i];
			}
			else if (dir == 2) {
				x = cell->x[i], y = cell->y[i];
			}
			r = sqrt(cell->r[i]*cell->r[i] - dis*dis);
			MP_Ln2dAddGc(sec, x, y, r);
		}
	}
	return sec->ngc;
}

int MP_Ln23dCutRandom(MP_Ln2dData *ln2d, int nsec, int ngc_step, MP_Ln3dCell *cell, long *seed)
{
	int i;
	int dir = 0;
	double pos;
	int ngc;
	int total = 0;

	for (i = 0; i < nsec;i++) {
		pos = cell->size[dir]*MP_Rand(&(ln2d->seed));
		ngc = MP_Ln23dCut(ln2d, ngc_step, cell, dir, pos);
		if (ngc < 0) return ngc;
		total += ngc;
		if (++dir >= 3) dir = 0;
	}
	return total;
}

	//static int CutSlab(MPMD_LN2DData *data, MP_CellMember *member, int dir, double pos, double thickness)
//{
//	int i;
//	double dis, adis;
//	double x, y, r;
//	MPMD_LN2DSection *section;
//
//	section = MPMD_LN2DAddSection(data);
//	if (section == NULL) return -1;
//	section->ngc = 0;
//	for (i = 0;i < member->natom;i++) {
//		if (dir == 0) dis = pos - member->x[i];
//		else if (dir == 1) dis = pos - member->y[i];
//		else if (dir == 2) dis = pos - member->z[i];
//		if (dis > member->size[dir]/2.0) {
//			dis -= member->size[dir];
//		}
//		else if (dis < -member->size[dir]/2.0) {
//			dis += member->size[dir];
//		}
//		adis = fabs(dis);
//		if (adis < 0.5*thickness+member->r[i]) {
//			if (dir == 0) {
//				x = member->y[i], y = member->z[i];
//			}
//			else if (dir == 1) {
//				x = member->x[i], y = member->z[i];
//			}
//			else if (dir == 2) {
//				x = member->x[i], y = member->y[i];
//			}
//			if (adis < 0.5*thickness) {
//				r = member->r[i];
//			}
//			else {
//				r = sqrt(member->r[i]*member->r[i] - (adis-0.5*thickness)*(adis-0.5*thickness));
//			}
//			if (!MPMD_LN2DAddGC(data, section, x, y, r)) return -1;
//		}
//	}
//	if (dir == 0) {
//		section->size[0] = member->size[1], section->size[1] = member->size[2];
//	}
//	else if (dir == 1) {
//		section->size[0] = member->size[0], section->size[1] = member->size[2];
//	}
//	else if (dir == 2) {
//		section->size[0] = member->size[0], section->size[1] = member->size[1];
//	}
//	return section->ngc;
//	return 0;
//}

//static int CutWedgePointX(double a, double b, double c, double r, double *x, double *rs)
//{
//	double d1, d2;
//	double x1, x2;
//
//	d1 = (c*a-b)/sqrt(c*c+1.0);
//	d2 = (-c*a-b)/sqrt(c*c+1.0);
//	if (d1 > -r && d2 < r) {
//		if (d1 <= 0.0 || d2 >= 0.0) {
//			if (fabs(d1) < fabs(d2)) {
//				x1 = (a+b*c-sqrt(r*r-a*a*c*c+c*c*r*r-b*b+2.0*a*b*c))/(c*c+1.0);
//				x2 = (a+b*c+sqrt(r*r-a*a*c*c+c*c*r*r-b*b+2.0*a*b*c))/(c*c+1.0);
//			}
//			else {
//				x1 = (a-b*c-sqrt(r*r-a*a*c*c+c*c*r*r-b*b-2.0*a*b*c))/(c*c+1.0);
//				x2 = (a-b*c+sqrt(r*r-a*a*c*c+c*c*r*r-b*b-2.0*a*b*c))/(c*c+1.0);
//			}
//			*x = (x1+x2)/2.0;
//			*rs = (x2-x1)/2.0;
//			return TRUE;
//		}
//		else {
//			*x = a;
//			*rs = r;
//			return TRUE;
//		}
//	}
//	return FALSE; 
//}
//
//static int CutWedge(MPMD_LN2D *ln2d, MP_CellMember *member, int dir, double pos, double thickness, short group)
//{
//	int i;
//	double dis;
//	double x, rs;
//	MPMD_LN2DSection *section;
//
//	section = MPMD_LN2DAddSection(ln2d);
//	if (section == NULL) return -1;
//	section->ngc = 0;
//	for (i = 0;i < member->natom;i++) {
//		if (dir == 0) dis = pos - member->x[i];
//		else if (dir == 1) dis = pos - member->y[i];
//		else if (dir == 2) dis = pos - member->z[i];
//		if (dis > member->size[dir]/2.0) {
//			dis -= member->size[dir];
//		}
//		else if (dis < -member->size[dir]/2.0) {
//			dis += member->size[dir];
//		}
//		if (dir == 0) {
//			if (CutWedgePointX(member->y[i], dis, thickness/2.0/member->size[1], member->r[i], &x, &rs)) {
//				if (!MPMD_LN2DAddGC(section, x, member->z[i], rs)) return -1;
//				if (group >= 0) member->group[i] = group;
//			}
//		}
//		else if (dir == 1) {
//			if (CutWedgePointX(member->x[i], dis, thickness/2.0/member->size[0], member->r[i], &x, &rs)) {
//				if (!MPMD_LN2DAddGC(section, x, member->z[i], rs)) return -1;
//				if (group >= 0) member->group[i] = group;
//			}
//		}
//		else if (dir == 2) {
//			if (CutWedgePointX(member->y[i], dis, thickness/2.0/member->size[1], member->r[i], &x, &rs)) {
//				if (!MPMD_LN2DAddGC(section, x, member->x[i], rs)) return -1;
//				if (group >= 0) member->group[i] = group;
//			}
//		}
//	}
//	if (dir == 0) {
//		section->size[0] = member->size[1], section->size[1] = member->size[2];
//	}
//	else if (dir == 1) {
//		section->size[0] = member->size[0], section->size[1] = member->size[2];
//	}
//	else if (dir == 2) {
//		section->size[0] = member->size[1], section->size[1] = member->size[0];
//	}
//	return section->ngc;
//	return 0;
//}
//
//int MPMD_LN2DCut(MPMD_LN2D *ln2d, MP_CellMember *member, int type, int dir, double pos, double thickness, short group)
//{
//	if (type == MPMD_LN2D_SLAB) {
//		return CutSlab(ln2d, member, dir, pos, thickness, group);
//	}
//	else if (type == MPMD_LN2D_WEDGE) {
//		return CutWedge(ln2d, member, dir, pos, thickness, group);
//	}
//	else {
//		MP_Print("invalid cutting type %d (MPMD_LN2DCut)\n", type);
//		return -1;
//	}
//}

//void MPMD_LN2DCutRandom(MPMD_LN2DData *data, MP_CellMember *member, int nsample, int type, double thickness, long *rand_seed)
//{
//	int ngc, total;
//	int dir;
//	double pos;
//
//	if (member->natom == 0) {
//		MP_Print("no particle in the cell (MPMD_LN2DCutRandom)\n");
//		return;
//	}
//	total = 0;
//	dir = 0;	
//	MPMD_LN2DClearSection(data);
//	while (total < nsample) {
//		pos = member->size[dir]*MP_Rand(rand_seed);
//		if (type == MPMD_LN2D_SLAB) {
//			ngc = CutSlab(data, member, dir, pos, thickness);
//		}
//		else if (type == MPMD_LN2D_WEDGE) {
//			return CutWedge(ln2d, member, dir, pos, thickness, group);
//		}
//		ngc = MPMD_LN2DCut(ln2d, member, type, dir, pos, thickness, -99);
//		if (ngc < 0) return;
//		total += ngc;
//		if (++dir >= 3) dir = 0;
//	}
//	MP_Print("%d GCs in %d sections. (MPMD_LN2DCutRandom)\n", total, data->nsection);
//}
