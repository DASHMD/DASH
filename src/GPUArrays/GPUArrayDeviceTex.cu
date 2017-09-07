#include "GPUArrayDeviceTex.h"

#include "memset_defs.h"

void MEMSETFUNC(cudaSurfaceObject_t surf, void *val_, int n, int Tsize) {
    if (Tsize == 4) {
        int val = * (int *) val_;
        memsetByValSurf_32<<<NBLOCK(n), PERBLOCK>>>(surf, val, n);
    } else if (Tsize == 8) {
        int2 val = * (int2 *) val_;
        memsetByValSurf_64<<<NBLOCK(n), PERBLOCK>>>(surf, val, n);
    } else if (Tsize == 16) {
        int4 val = * (int4 *) val_;
        memsetByValSurf_128<<<NBLOCK(n), PERBLOCK>>>(surf, val, n);
    } else {
        mdError("Data type has incompatible size");
    }
}
