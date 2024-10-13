/** @file jlt_module.c
 *  @brief A Python C extension for the Johnson-Lindenstrauss transform.
 *  @author Sean Svihla
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define ONESIXTH 0.16666666666666
#define ONETHIRD 0.33333333333333

#include <Python.h>
#include <numpy/arrayobject.h>
#include <cblas.h>

/*****************************************************************************
 *                             helper functions                              *
 *****************************************************************************/

static inline double
rand_uniform(double a, double b)
{
    return a + (b-a)*rand()/(double)RAND_MAX; // Uniform(a,b)
}

static inline double 
rand_normal(double mu, double sigma2) 
{
    // Marsaglia polar method
    double u1, u2, s, m;
    do
    {
        u1 = rand_uniform(-1.0, 1.0);
        u2 = rand_uniform(-1.0, 1.0);
        s = u1 * u1 + u2 * u2;
    } while (s >= 1.0 || s == 0.0);

    m = sqrt(-2.0 * log(s)/s);
    return mu + sigma2 * u1 * m; // Normal(mu, sigma2)
}

/* convenience wrapper for cblas_dgemm */
static inline PyArrayObject*
_cblas_dgemm(const CBLAS_TRANSPOSE transA,
             const CBLAS_TRANSPOSE transX,
             double a, PyArrayObject *A_arr, PyArrayObject *X_arr, 
             double b, PyArrayObject *B_arr)
{
    double *A = (double *)PyArray_DATA(A_arr);
    double *X = (double *)PyArray_DATA(X_arr);
    double *B = (double *)PyArray_DATA(B_arr);

    int k = (int)PyArray_DIM(A_arr, 0);
    int d = (int)PyArray_DIM(A_arr, 1);
    int n = (int)PyArray_DIM(X_arr, 1); 

    cblas_dgemm(CblasRowMajor, transA, transX,
                k, n, d,         // m = k, n = n, k = d
                a,               // Scaling factor for A
                A, d,            // 'lda' = leading dimension of A (d)
                X, n,            // 'ldb' = leading dimension of X (n)
                b,               // Scaling factor for B (output)
                B, n);           // 'ldc' = leading dimension of B (n)

    return B_arr;
}

/* convenience wrapper for PyArray_ZEROS */
static inline PyArrayObject*
_PyArray_ZEROS(npy_intp dim1, npy_intp dim2)
{
    PyArrayObject *arr;
    npy_intp      *dims;

    dims = (npy_intp*)malloc(2 * sizeof(npy_intp));
    if(dims == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to allocate npy_intp *dims");
        return NULL;
    }
    dims[0] = dim1;
    dims[1] = dim2;
    arr = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_DOUBLE, 0);
    free(dims);
    if(arr == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Failed to initialize PyArrayObject arr");
        return NULL;
    }
    Py_INCREF(arr);
    return arr;
}

/*****************************************************************************
 *                     top-level method definitions                          *
 *****************************************************************************/

/**
 * @brief Computes the Johnson-Lindenstrauss transform for dimensionality 
 *        reduction.
 *
 * This function takes a 2D NumPy array and projects it onto a lower-
 * dimensional space using a random Gaussian matrix, ensuring that the distance 
 * between points in the high-dimensional space is approximately preserved in 
 * the lower-dimensional space. The reduced dimensionality is specified by the 
 * `new_dim` parameter.
 *
 * @param self Python module object (unused in this function).
 * @param args Python tuple containing:
 *             - `input_arr` (PyArrayObject*): 2D NumPy array to be transformed.
 *             - `new_dim` (npy_intp): Target number of dimensions for the 
 *                                     output array.
 *
 * @return PyObject* A 2D NumPy array containing the transformed data. 
 *         Returns NULL if any error occurs, with an appropriate Python 
 *         exception set.
 *
 * @exception RuntimeError If input tuple parsing fails.
 * @exception ValueError If the input array is not 2-dimensional.
 * @exception MemoryError If memory allocation for the transformation or output 
 *                        array fails.
 *
 * The function uses the following steps:
 * 1. Validates input parameters and checks that the input array is 
 *    2-dimensional.
 * 2. Allocates memory for the transformation matrix, filling it with random 
 *    Gaussian values.
 * 3. Allocates memory for the output array and performs the matrix 
 *    multiplication with the input array.
 * 4. Returns the output array, which is the result of the Johnson-
 *    Lindenstrauss transform.
 *
 * @note The random Gaussian matrix is generated with standard normal 
 *       distribution sampled using the Marsaglia polar method.
 * 
 * @see https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma for 
 *      general information on the Johnson-Lindenstrauss transform.
 * @see http://arxiv.org/abs/2103.00564 for an overview of the theory and use 
 *      of the Johnson-Lindenstrauss transform.
 */
static PyObject*
jl(PyObject *self, PyObject *args)
{
    PyArrayObject *input_arr, *output_arr, *transf_arr;
    npy_intp      *dims, new_dim, size, idx;
    double        *data;

    if(!PyArg_ParseTuple(args, "O!n", &PyArray_Type, &input_arr, &new_dim))
    {
        PyErr_SetString(PyExc_RuntimeError, 
                        "Failed to parse input tuple");
        return NULL;
    }

    if(PyArray_NDIM(input_arr) != 2)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Expected 2-dimensional np.ndarray");
        return NULL;
    }
    dims = PyArray_DIMS(input_arr);

    transf_arr = _PyArray_ZEROS(new_dim, dims[0]);
    if(transf_arr == NULL) 
    {
        // PyErr_SetString in _PyArray_ZEROS
        return NULL;
    }

    size = PyArray_SIZE(transf_arr);
    data = (double *)PyArray_DATA(transf_arr);
    for(idx=0; idx<size; idx++)
    {
        data[idx]=(double)rand_normal(0.0, 1.0);
    }

    // perform the transform
    output_arr = _PyArray_ZEROS(new_dim, dims[1]);
    if(output_arr == NULL) 
    {
        // PyErr_SetString in _PyArray_ZEROS
        Py_DECREF(transf_arr);
        return NULL;
    }
    output_arr = _cblas_dgemm(1.0/sqrt((double)new_dim), transf_arr, 
                              input_arr, 0.0, output_arr);

    Py_DECREF(transf_arr);
    return (PyObject *)output_arr;
}

/**
 * @brief Computes the Fast Johnson-Lindenstrauss transform using a sparse 
 *        random projection matrix.
 *
 * This function applies a faster variant of the Johnson-Lindenstrauss transform 
 * by leveraging a sparse random matrix where each entry is drawn from a 
 * distribution that produces either -1, 0, or 1, with probabilities 1/6, 2/3, 
 * and 1/6, respectively. The method is used for efficient dimensionality 
 * reduction with reduced computational cost compared to a dense Gaussian 
 * matrix.
 *
 * @param self Python module object (unused in this function).
 * @param args Python tuple containing:
 *             - `input_arr` (PyArrayObject*): 2D NumPy array to be transformed.
 *             - `new_dim` (npy_intp): Target number of dimensions for the 
 *                                     output array.
 *
 * @return PyObject* A 2D NumPy array containing the transformed data.
 *         Returns NULL if any error occurs, with an appropriate Python 
 *         exception set.
 *
 * @exception RuntimeError If input tuple parsing fails.
 * @exception ValueError If the input array is not 2-dimensional.
 * @exception MemoryError If memory allocation for the transformation or output 
 *                        array fails.
 *
 * The function uses the following steps:
 * 1. Validates input parameters and checks that the input array is 
 *    2-dimensional.
 * 2. Allocates memory for the transformation matrix, filling it with values 
 *    from a sparse distribution:
 *    - Each entry in the transformation matrix is randomly set to -1, 0, or 1.
 * 3. Allocates memory for the output array and performs the matrix 
 *    multiplication with the input array.
 * 4. Returns the output array, which is the result of the Fast Johnson-
 *    Lindenstrauss transform.
 *
 * @note The sparse transformation matrix uses entries drawn from {-1, 0, 1}, 
 *       which significantly reduces the computation time and memory 
 *       requirements compared to the standard Johnson-Lindenstrauss transform.
 *
 * @see https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma for 
 *      more information on the Johnson-Lindenstrauss transform.
 * @see http://arxiv.org/abs/2103.00564 for an overview of the theory and use 
 *      of the Johnson-Lindenstrauss transform.
 */
static PyObject*
fastjl(PyObject *self, PyObject *args)
{
    PyArrayObject *input_arr, *output_arr, *transf_arr;
    npy_intp      *dims, size, new_dim, idx;
    double        *data, draw;

    if(!PyArg_ParseTuple(args, "O!n", &PyArray_Type, &input_arr, &new_dim))
    {
        PyErr_SetString(PyExc_RuntimeError, 
                        "Failed to parse input tuple");
        return NULL;
    }

    if(PyArray_NDIM(input_arr) != 2)
    {
        PyErr_SetString(PyExc_ValueError,
                        "Expected 2-dimensional np.ndarray");
        return NULL;
    }
    
    dims = PyArray_DIMS(input_arr);
    transf_arr  = _PyArray_ZEROS(new_dim, dims[0]);
    if(transf_arr == NULL)
    {
        // PyErr_SetString in _PyArray_ZEROS
        return NULL;
    }

    size = PyArray_SIZE(transf_arr);
    data = (double *)PyArray_DATA(transf_arr);
    for(idx=0; idx<size; idx++)
    {
            draw = rand_uniform(0,1);
            data[idx] = (-1.0)*(draw < ONESIXTH) 
                       +( 1.0)*(draw >= ONESIXTH && draw < ONETHIRD);
    }

    // perform the transform
    output_arr = _PyArray_ZEROS(new_dim, dims[1]);
    if(output_arr == NULL) 
    {
        // PyErr_SetString in _PyArray_ZEROS
        Py_DECREF(transf_arr);
        return NULL;
    }
    output_arr = _cblas_dgemm(sqrt(3.0/(double)new_dim), transf_arr,
                              input_arr, 0.0, output_arr);

    Py_DECREF(transf_arr);
    return (PyObject *)output_arr;
}

static PyMethodDef JLTMethods[] = {
    {"jl", jl, METH_VARARGS, "Johnson-Lindenstrauss transform."},
    {"fastjl", fastjl, METH_VARARGS, "Fast Johnson-Lindenstrauss transform."},
    {NULL, NULL, 0, NULL}  /* Sentinel */
};

/*****************************************************************************
 *                            module definition                              *
 ******************************************************************************/

static PyModuleDef jlt_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "jlt",
    .m_doc = "Johnson-Lindenstrauss transform",
    .m_size = -1,
    .m_methods = JLTMethods
};

PyMODINIT_FUNC PyInit_jlt(void)
{
    PyObject *m;

    import_array(); // Initialize the NumPy C API

    m = PyModule_Create(&jlt_module);
    if(m == NULL)
    {
        return NULL;
    }

    return m;
}
