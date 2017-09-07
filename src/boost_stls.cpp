#include "boost_stls.h"
#include "Bounds.h"
#include "array_indexing_suite.hpp"
//#include "array_ref.hpp"

namespace py = boost::python;
using namespace std;

void export_stls() {
    py::class_<std::map<std::string, int> >("stringInt")
        .def(py::map_indexing_suite<std::map<std::string, int> >())
        ;
    py::class_<std::map<std::string, uint32_t> >("stringUInt")
        .def(py::map_indexing_suite<std::map<std::string, uint32_t> >())
        ;
    py::class_<std::vector<std::string> >("vecstring")
        .def(py::vector_indexing_suite<std::vector<std::string> >())
        ;
    py::class_<std::vector<double> >("vecdouble")
        .def(py::vector_indexing_suite<std::vector<double> >() )
        ;
    py::class_<std::vector<vector<double> > >("vecdouble")
        .def(py::vector_indexing_suite<std::vector< vector<double> > >() )
        ;
    py::class_<std::vector<int> >("vecInt")
        .def(py::vector_indexing_suite<std::vector<int> >() )
        ;

    py::class_<std::vector<int64_t> >("vecLong")
        .def(py::vector_indexing_suite<std::vector<int64_t> >() )
        ;
    py::class_<std::vector<Atom> >("vecAtom")
        .def(py::vector_indexing_suite<std::vector<Atom> >() )
        ;
    py::class_<std::vector<SHARED(WriteConfig) > >("vecWriteConfig")
        .def(py::vector_indexing_suite<std::vector<SHARED(WriteConfig) > >() )
        ;
    py::class_<std::vector<SHARED(Fix) > >("vecFix")
        .def(py::vector_indexing_suite<std::vector<SHARED(Fix) > >() )
        //	.def("remove", &vectorRemove<Fix>)
        //	.staticmethod("remove")
        ;

    py::class_<std::array<int, 2> >("arrayInt2")
        .def(array_indexing_suite<std::array<int, 2> >() )
        ;
    py::class_<std::array<int, 3> >("arrayInt3")
        .def(array_indexing_suite<std::array<int, 3> >() )
        ;
    py::class_<std::array<int, 4> >("arrayInt4")
        .def(array_indexing_suite<std::array<int, 4> >() )
        ;
    py::class_<std::array<double, 4> >("arrayDouble4")
        .def(array_indexing_suite<std::array<double, 4> >() )
        ;
}
