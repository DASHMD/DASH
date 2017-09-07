#include "FixWall.h"
#include "State.h"

namespace py = boost::python;

// export FixWall()
void export_FixWall() {
	py::class_<FixWall, SHARED(FixWall), py::bases<Fix> > (
		"FixWall",
		py::no_init
	);

}

