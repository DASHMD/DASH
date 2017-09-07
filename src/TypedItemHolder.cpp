#include "TypedItemHolder.h"
#include "boost_for_export.h"
void export_TypedItemHolder() {
    boost::python::class_<TypedItemHolder> ("TypedItemHolder", boost::python::no_init)
    .def("getTypeIds", &TypedItemHolder::getTypeIds)
    ;
}
