#pragma once
#ifndef GPUARRAYGLOBAL_H
#define GPUARRAYGLOBAL_H

#include <vector>

#include "GPUArray.h"
#include "GPUArrayDeviceGlobal.h"

//! Array storing data on the CPU and the GPU
/*!
 * \tparam T Data type stored in the array
 *
 * GPU array stores data on the host CPU and the GPU device and is able to
 * move the data from the CPU to the GPU and back again.
 */
template <typename T>
class GPUArrayGlobal : public GPUArray {

public:
    //! Constructor
    /*!
     * Constructor creating empty host CPU data vector.
     */
    GPUArrayGlobal() {}

    //! Constructor
    /*!
     * \param size_ Size (number of elements) on the CPU and GPU data array
     *
     * Constructor creating empty arrays of the specified size on the CPU
     * and the GPU.
     */
    explicit GPUArrayGlobal(int size_)
        : h_data(std::vector<T>(size_,T())), d_data(GPUArrayDeviceGlobal<T>(size_)) {}

    //! Copy from vector constructor
    /*!
     * \param vals Vector to be passed to the CPU array
     *
     * Constructor setting the CPU data array with the specified vector.
     */
    explicit GPUArrayGlobal(std::vector<T> const &vals) {
        h_data = vals;
        d_data = GPUArrayDeviceGlobal<T>(h_data.size());
    }

    //! Move from vector constructor
    /*!
     * \param vals vector to be moved to the GPUArray host data
     */
    explicit GPUArrayGlobal(std::vector<T> &&vals)
    {
        h_data = std::move(vals);
        d_data = GPUArrayDeviceGlobal<T>(h_data.size());
    }

    //! Copy assignment from vector
    /*!
     * \param vals vector containing data to be copied to host CPU vector
     * \return This object
     */
    GPUArrayGlobal const &operator=(std::vector<T> const &vals)
    {
        d_data.resize(vals.size());
        h_data = vals;

        return *this;
    }

    //! Move assignment from vector
    /*!
     * \param vals vector containing data to be moved to host CPU vector
     * \return This object
     */
    GPUArrayGlobal const &operator=(std::vector<T> &&vals)
    {
        d_data.resize(vals.size());
        h_data = std::move(vals);

        return *this;
    }

    //! Return number of elements stored in the array
    /*!
     * \return Number of elements in the array
     */
    size_t size() const { return h_data.size(); }

    //! Send data from CPU to GPU
    void dataToDevice() {
        d_data.set(h_data.data());
    }
    bool set(std::vector<T> &other) {
    
        if (other.size() < size()) {
            h_data = other;
            return true;
        } else {
            d_data = GPUArrayDeviceGlobal<T>(other.size());
            h_data = other;
        }
        return false;
    }

    //! Send data from GPU to CPU asynchronously
    void dataToHostAsync(cudaStream_t stream) {
        d_data.get(h_data.data(), stream);
    }

    //! Send data from GPU to CPU synchronously
    void dataToHost() {
        //eeh, want to deal with the case where data originates on the device,
        //which is a real case, so removed checked on whether data is on device
        //or not
        d_data.get(h_data.data());
    }

    //! Copy data to GPU array
    void copyToDeviceArray(void *dest) {
        d_data.copyToDeviceArray(dest);
    }

    //! Return pointer to GPU data array
    T *getDevData() {
        return d_data.data();
    }

    //! Set Memory by value
    void memsetByVal(T val) {
        d_data.memsetByVal(val);
    }

public:

    std::vector<T> h_data; //!< Array storing data on the CPU
    GPUArrayDeviceGlobal<T> d_data; //!< Array storing data on the GPU
};

#endif
