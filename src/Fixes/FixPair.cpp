#include "FixPair.h"
#include "GPUArrayGlobal.h"
#include "State.h"

#include <cmath>
#include "xml_func.h"
#include "Logging.h"
namespace py = boost::python;

const std::string ARITHMETICTYPE = "arithmetic";
const std::string GEOMETRICTYPE = "geometric";

void FixPair::prepareParameters(std::string handle,
                                std::function<float (float, float)> fillFunction,
                                std::function<float (float)> processFunction,
                                bool fillDiag,
                                std::function<float ()> fillDiagFunction)
{
    std::vector<float> &preProc = *paramMap[handle];
    std::vector<float> *postProc = &paramMapProcessed[handle];
    int desiredSize = state->atomParams.numTypes;

    *postProc = preProc;
    ensureParamSize(*postProc);
    if (fillDiag) {
        SquareVector::populateDiagonal<float>(postProc, desiredSize, fillDiagFunction);
    }
    //populate will fill off-diagonal terms
    SquareVector::populate<float>(postProc, desiredSize, fillFunction);
    //process will perform unary operations on parameters, like converting rCut to rCut^2
    SquareVector::process<float>(postProc, desiredSize, processFunction);
    
    //okay, now ready to go to device!

}

void FixPair::prepareParameters(std::string handle,
                                std::function<float (float)> processFunction)
{
    std::vector<float> &preProc = *paramMap[handle];
    std::vector<float> *postProc = &paramMapProcessed[handle];
    int desiredSize = state->atomParams.numTypes;

    *postProc = preProc;
    ensureParamSize(*postProc);
    SquareVector::check_populate<float>(postProc, desiredSize);
    SquareVector::process<float>(postProc, desiredSize, processFunction);
}

void FixPair::prepareParameters(std::string handle,
                                std::function<float (int, int)>  fillFunction)
{
    //std::vector<float> &array = *paramMap[handle];
    //int desiredSize = state->atomParams.numTypes;
    //ensureParamSize(array);
    std::vector<float> &preProc = *paramMap[handle];
    std::vector<float> *postProc = &paramMapProcessed[handle];
    int desiredSize = state->atomParams.numTypes;

    *postProc = preProc;
    ensureParamSize(*postProc);
    SquareVector::populate<float>(postProc, desiredSize, fillFunction);
}

void FixPair::prepareParameters_from_other(std::string handle,
                                std::function<float (int,int)> fillFunction,
                                std::function<float (float)> processFunction,
                                bool fillDiag,
                                std::function<int ()> fillDiagFunction)
{
    std::vector<float> &preProc = *paramMap[handle];
    std::vector<float> *postProc = &paramMapProcessed[handle];
    int desiredSize = state->atomParams.numTypes;

    *postProc = preProc;
    ensureParamSize(*postProc);
    if (fillDiag) {
        SquareVector::populateDiagonal<float>(postProc, desiredSize, fillDiagFunction);
    }
    SquareVector::populate<float>(postProc, desiredSize, fillFunction);
    SquareVector::process<float>(postProc, desiredSize, processFunction);
    
    //okay, now ready to go to device!

}

void FixPair::acceptChargePairCalc(Fix *chargeFix) {
    std::vector<float> cutoffs = chargeFix->getRCuts();
    mdAssert(cutoffs.size()==1, "Charge fix gave multiple rcutoffs.  This is a bug.");
    chargeRCut = cutoffs[0];

    chargeCalcFix = chargeFix;
    //setEvalWrapper(); done in integrator after prepareForRun is done

}
void FixPair::ensureParamSize(std::vector<float> &array)
{
    int desiredSize = state->atomParams.numTypes;
    if (array.size() != desiredSize*desiredSize) {
        std::vector<float> newVals = SquareVector::copyToSize(
                array,
                sqrt((double) array.size()),
                state->atomParams.numTypes
                );
        std::vector<float> *asPtr = &array;
        *(&array) = newVals;
    }
}

void FixPair::ensureOrderGivenForAllParams() {
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        std::string handle = it->first;
        if (find(paramOrder.begin(), paramOrder.end(), handle) == paramOrder.end()) {
            mdError("Order for all parameters not specified");
        }
    }
}
void FixPair::sendAllToDevice() {
    ensureOrderGivenForAllParams();
    int totalSize = 0;
    for (auto it = paramMapProcessed.begin(); it!=paramMapProcessed.end(); it++) {
        totalSize += it->second.size(); 

    }
    paramsCoalesced = GPUArrayDeviceGlobal<float>(totalSize);
    int runningSize = 0;
    for (std::string handle : paramOrder) {
        std::vector<float> &vals = paramMapProcessed[handle];
        paramsCoalesced.set(vals.data(), runningSize, vals.size());
        runningSize += vals.size();
    }
}

bool FixPair::setParameter(std::string param,
                           std::string handleA,
                           std::string handleB,
                           double val)
{
    int i = state->atomParams.typeFromHandle(handleA);
    int j = state->atomParams.typeFromHandle(handleB);
    if (i == -1 or j == -1) {
        return false;
    }
    if (paramMap.find(param) != paramMap.end()) {
        int numTypes = state->atomParams.numTypes;
        std::vector<float> &arr = *(paramMap[param]);
        ensureParamSize(arr);
        if (i>=numTypes or j>=numTypes or i<0 or j<0) {
            std::cout << "Tried to set param " << param
                      << " for invalid atom types " << handleA
                      << " and " << handleB
                      << " while there are " << numTypes
                      << " species." << std::endl;
            return false;
        }
        squareVectorRef<float>(arr.data(), numTypes, i, j) = val;
        squareVectorRef<float>(arr.data(), numTypes, j, i) = val;
        return true;
    } 
    return false;
}


double FixPair::getParameter(std::string param,
                           std::string handleA,
                           std::string handleB)
{
    int i = state->atomParams.typeFromHandle(handleA);
    int j = state->atomParams.typeFromHandle(handleB);
    if (i == -1 or j == -1) {
        std::cout << "Tried to get parameter " << param << " for species " << handleA << " and " << handleB << ".  Invalid combination of parameter, species." << std::endl;
        exit(1);
        return -1;
    }
    if (paramMap.find(param) != paramMap.end()) {
        int numTypes = state->atomParams.numTypes;
        std::vector<float> &arr = *(paramMap[param]);
        if (i>=numTypes or j>=numTypes or i<0 or j<0) {
            std::cout << "Tried to get param " << param
                      << " for invalid atom types " << handleA
                      << " and " << handleB
                      << " while there are " << numTypes
                      << " species." << std::endl;
            exit(1);
            return -1;
        }
        return squareVectorItem<float>(arr.data(), numTypes, i, j);
    } 
    std::cout << "Tried to get parameter " << param << " for species " << handleA << " and " << handleB << ".  Invalid combination of parameter, species." << std::endl;
    exit(1);
    return -1;

}
void FixPair::initializeParameters(std::string paramHandle,
                                   std::vector<float> &params) {
    ensureParamSize(params);
    paramMap[paramHandle] = &params;
    paramMapProcessed[paramHandle] = std::vector<float>();
}


bool FixPair::readFromRestart() {
    pugi::xml_node restData = getRestartNode();
    //params must be already initialized at this point (in constructor)
    if (restData) {
        auto curr_param = restData.first_child();
        while (curr_param) {
            std::string tag = curr_param.name();
            if (tag == "parameter") {
                std::string paramHandle = curr_param.attribute("handle").value();
                auto it = find(paramOrder.begin(), paramOrder.end(), paramHandle);
                if (it == paramOrder.end()) {
                    std::cout << "Tried to read bad restart data for fix " << handle << ".  Data type " << paramHandle << std::endl;
                }
                mdAssert(it != paramOrder.end(), "Invalid restart data for fix");
                std::vector<float> *params = paramMap[paramHandle];
                ensureParamSize(*params);
                std::vector<float> src = xml_readNums<float>(curr_param);
                assert(params->size() >= src.size());
                for (int i=0; i<src.size(); i++) {
                    (*params)[i] = src[i];
                }
            }
            curr_param = curr_param.next_sibling();
        }

    }
    return true;
    
}



std::string FixPair::restartChunkPairParams(std::string format) {
    std::stringstream ss;
    //char buffer[128];
    //ignoring format for now
    for (auto it=paramMap.begin(); it!=paramMap.end(); it++) {
        //sprintf(buffer, "<parameter handle=\"%s\">", it->first.c_str());
        ss << "<parameter handle=\"" << it->first.c_str() << "\">\n";
        //ss << buffer << "\n";
        for (int i = 0; i < it->second->size(); i++) {
            ss << it->second->at(i) << "\n";
            //for (int j = 0; j < it->second[i].size(); j++) {
	        //    ss << it->second[i][j] << "\n";
           // }
        }
        /*
        for (float x : *(it->second)) {
            std::cout << "Parameter: " << x << std::endl;
              ss << x << "\n";
        }*/
        ss << "</parameter>\n";
    }
    return ss.str();
}    

void FixPair::handleBoundsChange() {
    if (hasAcceptedChargePairCalc && state->boundsGPU != boundsLast) {
        boundsLast = state->boundsGPU;
        chargeCalcFix->handleBoundsChange();
        setEvalWrapper();
    }
}

void FixPair::setMixingRules(std::string input) {
	if (input == ARITHMETICTYPE) {
		mixingRules = ARITHMETICTYPE;
	} else if (input == GEOMETRICTYPE) {
		mixingRules = GEOMETRICTYPE;
	} else {
		std::cout << "Invalid mixing rules: " << input << ". Use arithmatic or geometric." << std::endl;
		assert(0);
	}

}

void export_FixPair() {
    py::class_<FixPair,
    boost::noncopyable,
    py::bases<Fix> > (
            "FixPair", py::no_init  )
        .def("setParameter", &FixPair::setParameter,
                (py::arg("param"),
                 py::arg("handleA"),
                 py::arg("handleB"),
                 py::arg("val")
                )
            )
        .def("getParameter", &FixPair::getParameter,
                (py::arg("param"),
                 py::arg("handleA"),
                 py::arg("handleB")
                )
            )
		.def("setMixingRules", &FixPair::setMixingRules)

        ;
}

