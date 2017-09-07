#include "PythonOperation.h"
#include "PythonHelpers.h"
using namespace std;
namespace py = boost::python;

PythonOperation::PythonOperation(string handle_, int operateEvery_, PyObject *operation_, bool synchronous_) {
    orderPreference = 0;//see header for comments
    operation = operation_;
    assert(PyCallable_Check(operation));
    operateEvery = operateEvery_;
    assert(operateEvery > 0);
    handle = handle_;
    synchronous = synchronous_;
}

bool PythonOperation::operate(int64_t turn) {
	try {
        py::object res = py::call<py::object>(operation, turn);

        py::extract<bool> resPy(res);
        if (resPy.check()) {
            return resPy;
        }
	} catch (boost::python::error_already_set &) {
		PythonHelpers::printErrors();
	}
    return false;
}

void export_PythonOperation() {
	py::class_<PythonOperation, SHARED(PythonOperation)> ("PythonOperation", py::init<string, int, PyObject*, py::optional<bool> >(py::args("handle", "operateEvery", "operation", "synchronous")) )
        .def_readwrite("operateEvery", &PythonOperation::operateEvery)
        .def_readwrite("operation", &PythonOperation::operation)
        .def_readonly("handle", &PythonOperation::handle)
        .def_readwrite("synchronous", &PythonOperation::synchronous)
        ;
}
