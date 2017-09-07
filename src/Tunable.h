#pragma once

class Tunable {
private:
    int nThreadPerBlock_;
    int nThreadPerAtom_;
public:
    void nThreadPerBlock(int set) {
        nThreadPerBlock_ = set;
    }
    int nThreadPerBlock() {
        return nThreadPerBlock_;
    }
    void nThreadPerAtom(int set) {
        nThreadPerAtom_ = set;
    }
    int nThreadPerAtom() {
        return nThreadPerAtom_;
    }
    Tunable () {
        nThreadPerBlock_ = 256;
        nThreadPerAtom_ = 1;
    }
};
