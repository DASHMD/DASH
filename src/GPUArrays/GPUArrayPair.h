#pragma once
#ifndef GPUARRAYPAIR_H
#define GPUARRAYPAIR_H

#include <cuda_runtime.h>
#include <vector>

#include "GPUArray.h"
#include "GPUArrayDeviceGlobal.h"

//! GPUArray keeping data on the memory twice
/*!
 * \tparam T Data type stored in the array pair
 *
 * GPUArrayPair manages two instead of one GPUArrayDeviceGlobal to store
 * the data on the GPU. Keeping the data on the memory twice speeds up sorting
 * and helps keeping the memory contiguous, which is necessary for efficient
 * GPU operations. The data on the GPU is accessed via an index which takes the
 * values 0 or 1. There is always one active index.
 *
 * \todo This class does not make sure that a given index is 0 or 1. A few
 *       Assert() calls would help, I think.
 */
template <typename T>
class GPUArrayPair : public GPUArray {

private:
    //! Set the CPU data of the Array
    /*!
     * \param vals Vector storing the data to be copied to the CPU
     *
     * This function sets the data on the CPU. The GPU data is not affected.
     *
     * \todo Do we really need this function?
     */
    void setHost(std::vector<T> &vals) {
        h_data = vals;
    }
public:
    unsigned int activeIdx; //!< Index of active GPUArrayDeviceGlobal

    //! Switch the active GPUArrayDeviceGlobal
    /*!
     * \return Newly activated index
     */
    unsigned int switchIdx() {
        activeIdx = !activeIdx;
        return activeIdx;
    }
    std::vector<T> h_data; //!< CPU data
    GPUArrayDeviceGlobal<T> d_data[2]; //!< Pair of GPU data

    //! Default constructor */
    GPUArrayPair() : GPUArray(), activeIdx(0) {}

    //! Constructor
    /*!
     * \param vals Vector containing the data to be stored on the CPU
     *
     * This constructor allocates the memory and copies the data from the
     * given vector to the CPU memory.
     *
     * \todo Make this constructor explicit
     */
    GPUArrayPair(std::vector<T> &vals) : activeIdx(0) {
        set(vals);
        for (int i=0; i<2; i++) {
            d_data[i] = GPUArrayDeviceGlobal<T>(vals.size());
        }
    }

    //! Return pointer to GPU data
    /*!
     * \param n Index of the GPU memory to return
     * \return Pointer to data on the GPU device
     *
     * \todo This will crash if n < 0 or n > 1
     * \todo Is it possible to make activeIdx the default. Maybe for n < 0
     *       and have n = -1 the default?
     */
    T *getDevData(int n) {
        return d_data[n].data();
    }

    //! Return pointer to the active GPU data
    /*!
     * \return Pointer to the memory location on the GPU device
     */
    T *getDevData() {
        return getDevData(activeIdx);
    }

    //! Set the CPU data
    /*!
     * \param other Vector containing the data to be stored on the CPU
     * \return True if new size is smaller than old size, else False
     *
     * Sets the data of the CPU. If necessary, the GPU memory is
     * reallocated. Note that this operation may delete the GPU memory.
     */
    bool set(std::vector<T> &other) {
        if (other.size() < size()) {
            setHost(other);
            return true;
        } else {
            for (int i=0; i<2; i++) {
                d_data[i] = GPUArrayDeviceGlobal<T>(other.size());
            }
            setHost(other);
        }
        return false;

    }

    //! Get the number of elements stored
    /*!
     * \return Number of elements stored
     */
    size_t size() const { return h_data.size(); }

    //! Get pointer to one of the GPU memories
    /*!
     * \param n Index specifying which GPU memory to access
     * \return Pointer to memory location on the GPU device
     *
     * This is a convenience function/operator and behaves like getDevData()
     */
    T *operator ()(int n) {
        return getDevData(n);
    }

    //! Copy data from CPU memory to active GPU memory
    void dataToDevice() {
        CUCHECK(cudaMemcpy(d_data[activeIdx].data(), h_data.data(), size()*sizeof(T), cudaMemcpyHostToDevice ));

    }

    //! Copy data from active GPU memory to CPU memory
    void dataToHost() {
        dataToHost(activeIdx);
    }

    //! Copy data from a specific GPU memory to CPU memory
    /*!
     * \param idx Index specifying which GPU memory to access
     */
    void dataToHost(int idx) {
        CUCHECK(cudaMemcpy(h_data.data(), d_data[idx].data(), size()*sizeof(T), cudaMemcpyDeviceToHost));
    }

    //! Copy data from active GPU memory to another GPU memory location
    /*!
     * \param dest Pointer to GPU memory; destination for copy.
     */
    void copyToDeviceArray(void *dest) {
        CUCHECK(cudaMemcpy(dest, d_data[activeIdx].data(), size()*sizeof(T), cudaMemcpyDeviceToDevice));
    }

    //! Copy data between the two GPU memories
    /*!
     * \param dst Index of the destination memory
     * \param src Index of the source memory
     * \return False if dst and src are identical. Else, return True.
     */
    bool copyBetweenArrays(int dst, int src) {
        if (dst != src) {
            CUCHECK(cudaMemcpy(d_data[dst].data(), d_data[src].data(), size()*sizeof(T), cudaMemcpyDeviceToDevice));
            return true;
        }
        return false;
    }

    //! Set all elements of a given GPU memory to a specific value
    /*!
     * \param val Value to set the elements to
     * \param idx Index of the GPU memory
     */
    void memsetByVal(T val, int idx) {
        d_data[idx].memsetByVal(val);
    }

    //! Set all elements of the active GPU memory to a specifiv value
    /*!
     * \param val Value to set the elements to
     */
    void memsetByVal(T val) {
        memsetByVal(val, activeIdx);
    }
};

#endif
