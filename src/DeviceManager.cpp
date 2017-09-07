#include "DeviceManager.h"
#include "boost_for_export.h"

#include <iostream>
using namespace std;
using namespace boost::python;
DeviceManager::DeviceManager() {
    cudaGetDeviceCount(&nDevices);
    setDevice(nDevices-1);
}
bool DeviceManager::setDevice(int i, bool output) {
    if (i >= 0 and i < nDevices) {
        //add error handling here
        cudaSetDevice(i);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        cudaGetDeviceProperties(&prop, i);
        currentDevice = i;
        if (output) {
            cout << "Selecting device " << i<<" " << prop.name << endl;
        }
        return true;
    }
    return false;
}
void export_DeviceManager() {
    class_<DeviceManager, boost::noncopyable>("DeviceManager", no_init)
        
        .def_readonly("nDevices", &DeviceManager::nDevices)
        .def_readonly("currentDevice", &DeviceManager::currentDevice)
        .def("setDevice", &DeviceManager::setDevice, (boost::python::arg("i"), boost::python::arg("output")=true ))
        ;

}


