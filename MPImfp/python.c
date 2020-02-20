#ifndef _DEBUG

#include <Python.h>
#include <numpy/arrayobject.h>
#include "MPImfp.h"

#if PY_MAJOR_VERSION >= 3
#define PY3
#endif

static PyObject *PyImfpMeasure(PyObject *self, PyObject *args, PyObject *kwds)
{	
	PyObject *img_obj, *f_obj;
	unsigned char barrier;
	double dpix;
	int nsample;
	long seed;
	int dflag;
	PyArrayObject *img_arr, *f_arr;
	npy_intp img_width, img_height, f_shape;
	npy_intp *img_strides;
	unsigned char *img_data;
	int *f_data;
	static char *kwlist[] = { "img", "barrier", "f", "dpix", "nsample", "seed", "dflag", NULL };

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "ObO!dili", kwlist, &img_obj, &barrier, &PyArray_Type, &f_obj, &dpix, &nsample, &seed, &dflag)) {
		return NULL;
	}
	img_arr = (PyArrayObject *)PyArray_FROM_OTF(img_obj, NPY_UINT8, NPY_IN_ARRAY);
	if (img_arr == NULL) return NULL;
	if (PyArray_NDIM(img_arr) != 2) {
		Py_XDECREF(img_arr);
		PyErr_SetString(PyExc_ValueError, "invalid image data, ndim must be 2");
		return NULL;
	}
	img_height = PyArray_DIM(img_arr, 0);
	img_width = PyArray_DIM(img_arr, 1);
	img_strides = PyArray_STRIDES(img_arr);
	img_data = (unsigned char *)PyArray_DATA(img_arr);
	f_arr = (PyArrayObject *)PyArray_FROM_OTF(f_obj, NPY_UINT, NPY_INOUT_ARRAY);
	if (f_arr == NULL) {
		Py_XDECREF(img_arr);
		return NULL;
	}
	if (PyArray_NDIM(f_arr) != 1) {
		Py_XDECREF(img_arr);
		Py_XDECREF(f_arr);
		PyErr_SetString(PyExc_ValueError, "invalid freq data, ndim must be 1");
		return NULL;
	}
	f_shape = PyArray_DIM(f_arr, 0);
	f_data = (unsigned int *)PyArray_DATA(f_arr);
	MP_ImfpMeasure(img_data, img_strides[0], img_width, img_height,
		barrier, f_shape, f_data, dpix, nsample, &seed, dflag);
	Py_DECREF(img_arr);
	Py_DECREF(f_arr);
	return Py_BuildValue("l", seed);
}

static PyMethodDef PyImfpMethods[] = {
	{ "measure", (PyCFunction)PyImfpMeasure, METH_VARARGS | METH_KEYWORDS,
	"measure(img, barrier, f, dpix, nsample, seed, dflag) : measure image mean free path" },
	{ NULL }  /* Sentinel */
};

#ifdef PY3
static struct PyModuleDef IMFPModuleDef = {
	PyModuleDef_HEAD_INIT,
	"MPImfp",
	NULL,
	-1,
	PyImfpMethods,
};
#endif

#ifndef PY3
PyMODINIT_FUNC initMPImfp(void)
{
	Py_InitModule3("MPImfp", PyImfpMethods, "MPImfp extention");
	import_array();
	return;
}
#else
PyMODINIT_FUNC PyInit_MPImfp(void)
{
	PyObject* m = PyModule_Create(&IMFPModuleDef);
	import_array();
	return m;
}
#endif

#endif /* _DEBUG */
