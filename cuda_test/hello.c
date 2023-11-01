#define PY_SSIZE_T_CLEAN
#include <Python.h>

extern void launch_hello_kernel();

// Define the method of the module
static PyObject* hello_say_hello(PyObject* self, PyObject* args) {
    // Print from CPU
    printf("Hello World from CPU!\n");
    launch_hello_kernel();
    Py_RETURN_NONE;
}

// Define the methods of the module
static PyMethodDef HelloMethods[] = {
    {"say_hello", hello_say_hello, METH_VARARGS, "Print 'Hello from C!'"},
    {NULL, NULL, 0, NULL}   /* Sentinel */
};

// Define the module
static struct PyModuleDef hellomodule = {
    PyModuleDef_HEAD_INIT,
    "hello",   /* name of module */
    NULL,     /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    HelloMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_hello(void) {
    return PyModule_Create(&hellomodule);
}
