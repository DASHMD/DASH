#include "GPUArrayDevice.h"

bool GPUArrayDevice::resize(size_t newSize, bool force /*= false*/)
{
    if (force || newSize > cap) {
        deallocate();
        n = newSize;
        allocate();
        return true;
    } else {
        n = newSize;
        return false;
    }
}