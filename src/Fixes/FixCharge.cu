#include "FixCharge.h"
#include "State.h"

namespace py = boost::python;

FixCharge::FixCharge(boost::shared_ptr<State> state_,
                     std::string handle_, std::string groupHandle_,
                     std::string type_, bool forceSingle_)
  : Fix(state_, handle_, groupHandle_, type_, forceSingle_, false, true, 1)
{   }

bool FixCharge::prepareForRun() {
    //check for electo neutrality
    double sum = 0.0;
    for (int i=0; i<state->atoms.size(); i++) {
        sum += state->atoms[i].q;
    }
    if (sum != 0.0) {
        std::cout << "System is not electroneutral. Total charge is " << sum
                  << std::endl;
    }
    return true;
}

void export_FixCharge() {
    py::class_<FixCharge, SHARED(FixCharge), py::bases<Fix> > (
        "FixCharge",
        boost::python::no_init
    );
}

