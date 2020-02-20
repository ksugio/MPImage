#ifndef _DEBUG

#include "MPLn23d.h"
#include <numpy/arrayobject.h>

#if PY_MAJOR_VERSION >= 3
#define PY3
#endif

static void PyLn3dDealloc(MP_Ln3dData* self)
{
	MP_Ln3dFree(self);
#ifndef PY3
	self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *PyLn3dNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	int ncmax;
	static char *kwlist[] = { "ncmax", NULL };
	MP_Ln3dData *self;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &ncmax)) {
		return NULL;
	}
	self = (MP_Ln3dData *)type->tp_alloc(type, 0);
	if (self != NULL) {
		if (!MP_Ln3dAlloc(self, ncmax)) {
			Py_DECREF(self);
			return NULL;
		}
	}
	return (PyObject *)self;
}

static PyMemberDef PyLn3dMembers[] = {
	{ "ncell", T_INT, offsetof(MP_Ln3dData, ncell), 1, "number of cell" },
	{ "ncell_max", T_INT, offsetof(MP_Ln3dData, ncell_max), 1, "maximum number of cell" },
	{ "seed", T_LONG, offsetof(MP_Ln3dData, seed), 0, "seed of random number" },
	{ NULL }  /* Sentinel */
};

static PyObject *PyLn3dAddCell(MP_Ln3dData *self, PyObject *args, PyObject *kwds)
{
	int step;
	double sx, sy, sz;
	static char *kwlist[] = { "step", "sx", "sy", "sz", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iddd", kwlist, &step, &sx, &sy, &sz)) {
		return NULL;
	}
	if (MP_Ln3dAddCell(self, step, sx, sy, sz) == NULL) {
		PyErr_SetString(PyExc_ValueError, "can't add cell");
		return NULL;
	}
	else {
		return Py_BuildValue("i", self->ncell-1);
	}
}

static PyObject *PyLn3dAddGc(MP_Ln3dData *self, PyObject *args, PyObject *kwds)
{
	int cid;
	double x, y, z, r;
	static char *kwlist[] = { "cid", "x", "y", "z", "r", NULL };
	MP_Ln3dCell *cell;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "idddd", kwlist, &cid, &x, &y, &z, &r)) {
		return NULL;
	}
	if (cid < 0 || cid >= self->ncell) {
		PyErr_SetString(PyExc_ValueError, "no cell");
		return NULL;
	}
	else {
		cell = &(self->cell[cid]);
		return Py_BuildValue("i", MP_Ln3dAddGc(cell, x, y, z, r));
	}
}

static PyObject *PyLn3dAddGcRandom(MP_Ln3dData *self, PyObject *args, PyObject *kwds)
{
	int cid, ngc;
	double sd, r;
	static char *kwlist[] = { "cid", "ngc", "sd", "r", NULL };
	MP_Ln3dCell *cell;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iidd", kwlist, &cid, &ngc, &sd, &r)) {
		return NULL;
	}
	if (cid < 0 || cid >= self->ncell) {
		PyErr_SetString(PyExc_ValueError, "no cell");
		return NULL;
	}
	else {
		cell = &(self->cell[cid]);
		return Py_BuildValue("i", MP_Ln3dAddGcRandom(cell, ngc, sd, r, &(self->seed)));
	}
}

static PyObject *PyLn3dMeasureGc(MP_Ln3dData *self, PyObject *args, PyObject *kwds)
{
	PyObject *f_obj;
	PyArrayObject *f_arr;
	static char *kwlist[] = { "f", NULL };
	npy_intp f_shape;
	unsigned int *f_data;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &f_obj)) {
		return NULL;
	}
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) return NULL;
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	MP_Ln3dMeasureGc(self, f_shape, f_data);
	Py_RETURN_NONE;
}

static PyObject *PyLn3dMeasureRandom(MP_Ln3dData *self, PyObject *args, PyObject *kwds)
{
	PyObject *f_obj;
	int nsample;
	PyArrayObject *f_arr;
	static char *kwlist[] = { "f", "nsample", NULL };
	npy_intp f_shape;
	unsigned int *f_data;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!i", kwlist, &PyArray_Type, &f_obj, &nsample)) {
		return NULL;
	}
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) return NULL;
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	MP_Ln3dMeasureRandom(self, f_shape, f_data, nsample);
	Py_RETURN_NONE;
}

static PyObject *PyLn3dVolumeFraction(MP_Ln3dData *self, PyObject *args)
{
	return Py_BuildValue("d", MP_Ln3dVolumeFraction(self));
}

static PyMethodDef PyLn3dMethods[] = {
	{ "add_cell", (PyCFunction)PyLn3dAddCell, METH_VARARGS | METH_KEYWORDS,
	"add_cell(step, sx, sy, sz) : add cell" },
	{ "add_gc", (PyCFunction)PyLn3dAddGc, METH_VARARGS | METH_KEYWORDS,
	"add_gc(cid, x, y, z, r) : add gc" },
	{ "add_gc_random", (PyCFunction)PyLn3dAddGcRandom, METH_VARARGS | METH_KEYWORDS,
	"add_gc_random(cid, ngc, sd, r) : add gc random" },
	{ "measure_gc", (PyCFunction)PyLn3dMeasureGc, METH_VARARGS | METH_KEYWORDS,
	"measure_gc(f) : measure on gc" },
	{ "measure_random", (PyCFunction)PyLn3dMeasureRandom, METH_VARARGS | METH_KEYWORDS,
	"measure_random(f, nsample) : measure on random point" },
	{ "volume_fraction", (PyCFunction)PyLn3dVolumeFraction, METH_NOARGS,
	"volume_fraction() : calculate volume fraction" },
	{ NULL }  /* Sentinel */
};

static PyTypeObject PyLn3dNewType = {
	PyObject_HEAD_INIT(NULL)
#ifndef PY3
	0,							/*ob_size*/
#endif
	"MPLn23d.ln3d_new",			/*tp_name*/
	sizeof(MP_Ln3dData),		/*tp_basicsize*/
	0,							/*tp_itemsize*/
	(destructor)PyLn3dDealloc,	/*tp_dealloc*/
	0,							/*tp_print*/
	0,							/*tp_getattr*/
	0,							/*tp_setattr*/
	0,							/*tp_compare*/
	0,							/*tp_repr*/
	0,							/*tp_as_number*/
	0,							/*tp_as_sequence*/
	0,							/*tp_as_mapping*/
	0,							/*tp_hash */
	0,							/*tp_call*/
	0,							/*tp_str*/
	0,							/*tp_getattro*/
	0,							/*tp_setattro*/
	0,							/*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,	/*tp_flags*/
	"ln3d_new(ncmax)",				/* tp_doc */
	0,							/* tp_traverse */
	0,							/* tp_clear */
	0,							/* tp_richcompare */
	0,							/* tp_weaklistoffset */
	0,							/* tp_iter */
	0,							/* tp_iternext */
	PyLn3dMethods,				/* tp_methods */
	PyLn3dMembers,				/* tp_members */
	0,							/* tp_getset */
	0,							/* tp_base */
	0,							/* tp_dict */
	0,							/* tp_descr_get */
	0,							/* tp_descr_set */
	0,							/* tp_dictoffset */
	0,							/* tp_init */
	0,							/* tp_alloc */
	PyLn3dNew,					/* tp_new */
};

static void PyLn2dDealloc(MP_Ln2dData* self)
{
	MP_Ln2dFree(self);
#ifndef PY3
	self->ob_type->tp_free((PyObject*)self);
#endif
}

static PyObject *PyLn2dNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	int nsmax;
	static char *kwlist[] = { "nsmax", NULL };
	MP_Ln2dData *self;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &nsmax)) {
		return NULL;
	}
	self = (MP_Ln2dData *)type->tp_alloc(type, 0);
	if (self != NULL) {
		if (!MP_Ln2dAlloc(self, nsmax)) {
			Py_DECREF(self);
			return NULL;
		}
	}
	return (PyObject *)self;
}

static PyMemberDef PyLn2dMembers[] = {
	{ "nsec", T_INT, offsetof(MP_Ln2dData, nsec), 1, "number of section" },
	{ "nsec_max", T_INT, offsetof(MP_Ln2dData, nsec_max), 1, "maximum number of section" },
	{ "seed", T_LONG, offsetof(MP_Ln2dData, seed), 0, "seed of random number" },
	{ NULL }  /* Sentinel */
};

static PyObject *PyLn2dAddSection(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	int step;
	double sx, sy;
	static char *kwlist[] = { "step", "sx", "sy", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "idd", kwlist, &step, &sx, &sy)) {
		return NULL;
	}
	if (MP_Ln2dAddSection(self, step, sx, sy) == NULL) {
		PyErr_SetString(PyExc_ValueError, "can't add section");
		return NULL;
	}
	else {
		return Py_BuildValue("i", self->nsec - 1);
	}
}

static PyObject *PyLn2dAddGc(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	int sid;
	double x, y, r;
	static char *kwlist[] = { "sid", "x", "y", "r", NULL };
	MP_Ln2dSection *sec;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iddd", kwlist, &sid, &x, &y, &r)) {
		return NULL;
	}
	if (sid < 0 || sid >= self->nsec) {
		PyErr_SetString(PyExc_ValueError, "no section");
		return NULL;
	}
	else {
		sec = &(self->sec[sid]);
		return Py_BuildValue("i", MP_Ln2dAddGc(sec, x, y, r));
	}
}

static PyObject *PyLn2dAddGcRandom(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	int sid, ngc;
	double sd, r;
	static char *kwlist[] = { "sid", "ngc", "sd", "r", NULL };
	MP_Ln2dSection *sec;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iidd", kwlist, &sid, &ngc, &sd, &r)) {
		return NULL;
	}
	if (sid < 0 || sid >= self->nsec) {
		PyErr_SetString(PyExc_ValueError, "no section");
		return NULL;
	}
	else {
		sec = &(self->sec[sid]);
		return Py_BuildValue("i", MP_Ln2dAddGcRandom(sec, ngc, sd, r, &(self->seed)));
	}
}

static PyObject *PyLn2dMeasureGc(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	PyObject *f_obj;
	PyArrayObject *f_arr;
	static char *kwlist[] = { "f", NULL };
	npy_intp f_shape;
	unsigned int *f_data;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &f_obj)) {
		return NULL;
	}
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) return NULL;
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	MP_Ln2dMeasureGc(self, f_shape, f_data);
	Py_RETURN_NONE;
}

static PyObject *PyLn2dMeasureRandom(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	PyObject *f_obj;
	int nsample;
	PyArrayObject *f_arr;
	static char *kwlist[] = { "f", "nsample", NULL };
	npy_intp f_shape;
	unsigned int *f_data;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!i", kwlist, &PyArray_Type, &f_obj, &nsample)) {
		return NULL;
	}
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) return NULL;
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	MP_Ln2dMeasureRandom(self, f_shape, f_data, nsample);
	Py_RETURN_NONE;
}

static PyObject *PyLn2dAreaFraction(MP_Ln2dData *self, PyObject *args)
{
	return Py_BuildValue("d", MP_Ln2dAreaFraction(self));
}

static PyObject *PyLn23dCut(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	int step, cid, dir;
	MP_Ln3dData *ln3d;
	double pos;
	static char *kwlist[] = { "step", "ln3d", "cid", "dir", "pos", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!iid", kwlist, &step, &PyLn3dNewType, &ln3d, &cid, &dir, &pos)) {
		return NULL;
	}
	if (cid < 0 || cid >= ln3d->ncell) {
		PyErr_SetString(PyExc_ValueError, "no cell");
		return NULL;
	}
	if (dir < 0 || dir >= 3) {
		PyErr_SetString(PyExc_ValueError, "invalid direction");
		return NULL;
	}
	return Py_BuildValue("i", MP_Ln23dCut(self, step, &(ln3d->cell[cid]), dir, pos));
}

static PyObject *PyLn23dCutRandom(MP_Ln2dData *self, PyObject *args, PyObject *kwds)
{
	int nsec, step, cid;
	MP_Ln3dData *ln3d;
	static char *kwlist[] = { "nsec", "step", "ln3d", "cid", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiO!i", kwlist, &nsec, &step, &PyLn3dNewType, &ln3d, &cid)) {
		return NULL;
	}
	if (cid < 0 || cid >= ln3d->ncell) {
		PyErr_SetString(PyExc_ValueError, "no cell");
		return NULL;
	}
	return Py_BuildValue("i", MP_Ln23dCutRandom(self, nsec, step, &(ln3d->cell[cid]), &(self->seed)));
}

static PyMethodDef PyLn2dMethods[] = {
	{ "add_sec", (PyCFunction)PyLn2dAddSection, METH_VARARGS | METH_KEYWORDS,
	"add_sec(step, sx, sy) : add section" },
	{ "add_gc", (PyCFunction)PyLn2dAddGc, METH_VARARGS | METH_KEYWORDS,
	"add_gc(sid, x, y, r) : add gc" },
	{ "add_gc_random", (PyCFunction)PyLn2dAddGcRandom, METH_VARARGS | METH_KEYWORDS,
	"add_gc_random(sid, ngc, sd, r) : add gc random" },
	{ "measure_gc", (PyCFunction)PyLn2dMeasureGc, METH_VARARGS | METH_KEYWORDS,
	"measure_gc(f) : measure on gc" },
	{ "measure_random", (PyCFunction)PyLn2dMeasureRandom, METH_VARARGS | METH_KEYWORDS,
	"measure_random(f, nsample) : measure on random point" },
	{ "area_fraction", (PyCFunction)PyLn2dAreaFraction, METH_NOARGS,
	"area_fraction() : calculate area fraction" },
	{ "cut", (PyCFunction)PyLn23dCut, METH_VARARGS | METH_KEYWORDS,
	"cut(step, ln3d, cid, dir, pos) : cut ln3d cell" },
	{ "cut_random", (PyCFunction)PyLn23dCutRandom, METH_VARARGS | METH_KEYWORDS,
	"cut_random(nsec, step, ln3d, cid) : cut ln3d cell randomly" },
	{ NULL }  /* Sentinel */
};

static PyTypeObject PyLn2dNewType = {
	PyObject_HEAD_INIT(NULL)
#ifndef PY3
	0,							/*ob_size*/
#endif
	"MPLn23d.ln2d_new",			/*tp_name*/
	sizeof(MP_Ln2dData),		/*tp_basicsize*/
	0,							/*tp_itemsize*/
	(destructor)PyLn2dDealloc,	/*tp_dealloc*/
	0,							/*tp_print*/
	0,							/*tp_getattr*/
	0,							/*tp_setattr*/
	0,							/*tp_compare*/
	0,							/*tp_repr*/
	0,							/*tp_as_number*/
	0,							/*tp_as_sequence*/
	0,							/*tp_as_mapping*/
	0,							/*tp_hash */
	0,							/*tp_call*/
	0,							/*tp_str*/
	0,							/*tp_getattro*/
	0,							/*tp_setattro*/
	0,							/*tp_as_buffer*/
	Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,	/*tp_flags*/
	"ln2d_new(nsmax)",				/* tp_doc */
	0,							/* tp_traverse */
	0,							/* tp_clear */
	0,							/* tp_richcompare */
	0,							/* tp_weaklistoffset */
	0,							/* tp_iter */
	0,							/* tp_iternext */
	PyLn2dMethods,				/* tp_methods */
	PyLn2dMembers,				/* tp_members */
	0,							/* tp_getset */
	0,							/* tp_base */
	0,							/* tp_dict */
	0,							/* tp_descr_get */
	0,							/* tp_descr_set */
	0,							/* tp_dictoffset */
	0,							/* tp_init */
	0,							/* tp_alloc */
	PyLn2dNew,					/* tp_new */
};

static PyObject *PyStatCalc(PyObject *self, PyObject *args, PyObject *kwds)
{
	static char *kwlist[] = { "f", NULL };
	PyObject *f_obj, *rf_obj;
	PyArrayObject *f_arr;
	npy_intp f_shape, rf_dim[1];
	unsigned int *f_data;
	double *rf_data;
	double ave, var;
	int total;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyArray_Type, &f_obj)) {
		return NULL;
	}
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) return NULL;
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	rf_dim[0] = f_shape;
	rf_obj = PyArray_SimpleNew(1, rf_dim, NPY_FLOAT64);
	rf_data = (double *)PyArray_DATA((PyArrayObject *)rf_obj);
	total =	MP_StatCalc(f_shape, f_data, rf_data, &ave, &var);
	return Py_BuildValue("(idd)O", total, ave, var, rf_obj);
}

static PyMethodDef MPLn23dPyMethods[] = {
	{ "stat_calc", (PyCFunction)PyStatCalc, METH_VARARGS | METH_KEYWORDS,
	"stat_calc(f, rf) : return total number, average, variance and relative frequency" },
	{ NULL }  /* Sentinel */
};

#ifdef PY3
static struct PyModuleDef Ln23dModuleDef = {
	PyModuleDef_HEAD_INIT,
	"MPLn23d",
	NULL,
	-1,
	MPLn23dPyMethods,
};
#endif

#ifndef PY3
PyMODINIT_FUNC initMPLn23d(void)
#else
PyMODINIT_FUNC PyInit_MPLn23d(void)
#endif
{
	PyObject *m;

#ifndef PY3
	if (PyType_Ready(&PyLn3dNewType) < 0) return;
	if (PyType_Ready(&PyLn2dNewType) < 0) return;
	m = Py_InitModule3("MPLn23d", MPLn23dPyMethods, "MPLn23d extention");
	if (m == NULL) return;
#else
	if (PyType_Ready(&PyLn3dNewType) < 0) return NULL;
	if (PyType_Ready(&PyLn2dNewType) < 0) return NULL;
	m = PyModule_Create(&Ln23dModuleDef);
	if (m == NULL) return NULL;
#endif
	import_array();
	Py_INCREF(&PyLn3dNewType);
	PyModule_AddObject(m, "ln3d_new", (PyObject *)&PyLn3dNewType);
	Py_INCREF(&PyLn2dNewType);
	PyModule_AddObject(m, "ln2d_new", (PyObject *)&PyLn2dNewType);
#ifdef PY3
	return m;
#endif
}

#endif /* _DEBUG */
