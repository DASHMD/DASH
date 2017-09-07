#pragma once
#include <string>
enum UNITS {REAL, LJ};
void export_Units();
class Units {
public:
    float boltz;
    float hbar;
    float mvv_to_eng;
    float qqr_to_eng;
    float nktv_to_press;
    float ftm_to_v;
    float *dt; //points to state's dt
    double toSIDensity;
    int unitType;
    //assumung dialectric constant is 1

    Units(float *dt_) {
        unitType = -1;
        dt = dt_;
        setLJ();
    }

    void setLJ();
    void setReal();
};
