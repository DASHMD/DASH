#include "FixExternal.h"
#include "State.h"

namespace py = boost::python;

// export FixExternal()
void export_FixExternal() {
	py::class_<FixExternal, SHARED(FixExternal), py::bases<Fix> > (
		"FixExternal",
		py::no_init
	);
}

