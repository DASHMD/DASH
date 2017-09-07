#include "Python.h"
#undef _XOPEN_SOURCE
#undef _POSIX_C_SOURCE
#include <boost/python.hpp>
namespace PythonHelpers {
void printErrors() {
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);

    boost::python::handle<> hType(ptype);
    boost::python::object extype(hType);

    boost::python::handle<> hTraceback(ptraceback);
    boost::python::object traceback(hTraceback);

    std::string errorMsg = boost::python::extract<std::string>(pvalue);
    int lineNum = boost::python::extract<int>(traceback.attr("tb_lineno"));
    std::string funcname = boost::python::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_name"));
    std::string filename = boost::python::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
    std::cout << "Error in python script" << std::endl;
    std::cout << errorMsg << std::endl;
    std::cout << "Line: " << lineNum << std::endl;
    std::cout << "Function: " << funcname << std::endl;
    std::cout << "File: " << filename << std::endl;
    //char *err = PyString_AsString(pvalue);
    //std::cout << err << std::endl;
    exit(0);
}
}
