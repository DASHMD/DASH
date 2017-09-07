#include "GPUArrayDeviceGlobal.h"
#include "memset_defs.h"

void MEMSETFUNC(void *ptr, const void *val, size_t n, size_t Tsize) {
    if (Tsize == 4) {
        memsetByValList_32<<<NBLOCK(n), PERBLOCK>>>((int *) ptr,
                                                    *(const int *)val, n);
    } else if (Tsize == 8) {
        memsetByValList_64<<<NBLOCK(n), PERBLOCK>>>((int2 *) ptr,
                                                    *(const int2 *)val, n);
    } else if (Tsize == 12) {
        memsetByValList_96<<<NBLOCK(n), PERBLOCK>>>((int3 *) ptr,
                                                    *(const int3 *)val, n);
    } else if (Tsize == 16) {
        memsetByValList_128<<<NBLOCK(n), PERBLOCK>>>((int4 *) ptr,
                                                     *(const int4 *)val, n);
    } else {
        mdError("Type parameter for memset has incompatible size");
    }
}
