#pragma once
#ifndef GPUARRAYDEVICEGLOBAL_H
#define GPUARRAYDEVICEGLOBAL_H

#include <cuda_runtime.h>

#include "globalDefs.h"
#include "GPUArrayDevice.h"
#include "Logging.h"

//! Global function to set the device memory
/*!
 * \param ptr Pointer to the memory
 * \param val Pointer to the value that will be stored into all elements
 * \param n Number of elements that will be set to val
 * \param Tsize Size of the value type
 */
void MEMSETFUNC(void *ptr, const void *val, size_t n, size_t Tsize);

//!Array on the GPU device
/*!
 * \tparam T Data type stored in the array
 *
 * Array storing data on the GPU device.
 */
template <typename T>
class GPUArrayDeviceGlobal : public GPUArrayDevice {
public:
    //! Constructor
    /*!
     * \param size Size of the array (number of elements)
     *
     * This constructor creates the array on the GPU device and allocates
     * enough memory to store n_ elements.
     */
    explicit GPUArrayDeviceGlobal(size_t size = 0)
        : GPUArrayDevice(size) { allocate(); }

    //! Copy constructor
    GPUArrayDeviceGlobal(const GPUArrayDeviceGlobal<T> &other)
        : GPUArrayDevice(other.n)
    {
        allocate();
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
    }

    //! Move constructor
    GPUArrayDeviceGlobal(GPUArrayDeviceGlobal<T> &&other)
        : GPUArrayDevice(other.n)
    {
        ptr = other.ptr;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
    }

    //! Destructor
    ~GPUArrayDeviceGlobal() {
        deallocate();
    }

    //! Assignment operator
    /*!
     * \param other GPUArrayDeviceGlobal from which data is copied
     * \return This object
     */
    GPUArrayDeviceGlobal<T> &operator=(const GPUArrayDeviceGlobal<T> &other) {
        if (n != other.n) {
            //! \todo Think about if it would be better not to force
            //!       reallocation here
            resize(other.n, true); // Force resizing
        }
        CUCHECK(cudaMemcpy(ptr, other.ptr, n*sizeof(T),
                                                cudaMemcpyDeviceToDevice));
        return *this;
    }

    //! Move assignment operator
    GPUArrayDeviceGlobal<T> &operator=(GPUArrayDeviceGlobal<T> &&other) {
        deallocate();
        n = other.n;
        ptr = other.ptr;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
        return *this;
    }

    //! Access pointer to data
    /*!
     * \return Pointer to memory location
     */
    T *data() { return (T*)ptr; }

    //! Const access pointer to data
    /*!
     * \return Const pointer to memory location
     */
    T const *data() const { return (T const*)ptr; }

    //! Copy data to given pointer
    /*!
     * \param copyTo Pointer to the memory to where the data will be copied
     * \param stream CUDA stream object for asynchronous copy
     *
     * This function copies the data stored in the GPU array to the given
     * host memory location. If a CUDA stream object is passed, the data will
     * be copied asynchronously using this stream. Otherwise, it will be copied
     * synchronously.
     */
    void get(void *copyTo, cudaStream_t stream = nullptr) const {
        get(copyTo, 0, size(), stream);
    }

    //! Copy sequence from the device array
    /*!
     * \param copyTo Pointer to CPU memory location where data is copied to
     * \param offset Index of first element to copy (array indices start at 0)
     * \param nElements Number of elements to copy
     * \param stream CUDA stream object for asynchronous copying

     * This function copies a given number of elements from the device, where
     * offset is the index of the first element and nElements is the number of
     * elements to copy. Make sure that offset + nElements <= size().
     *
     * get() calls this function to copy the whole array with offset=0 and
     * nElements = size().
     *
     * If a CUDA stream object is passed to this function, it will copy the
     * data asynchronously using this stream. Otherwise, it will copy the data
     * synchronously.
     */
    void get(void *copyTo, size_t offset, size_t nElements,
                                        cudaStream_t stream = nullptr) const
    {
        if (copyTo == nullptr) { return; }
        T *pointer = (T*)ptr;
        if (stream) {
            CUCHECK(cudaMemcpyAsync(copyTo, pointer+offset, nElements*sizeof(T),
                                            cudaMemcpyDeviceToHost, stream));
        } else {
            CUCHECK(cudaMemcpy(copyTo, pointer+offset, nElements*sizeof(T),
                                                    cudaMemcpyDeviceToHost));
        }
    }

    //! Copy data from pointer
    /*!
     * \param copyFrom Pointer to address from which the data will be
     *                 copied
     *
     * Copy data from a given adress specified by the copyFrom pointer to
     * the GPU array. The number of bytes copied from memory is the size of
     * the the GPUArrayDeviceGlobal.
     */
    void set(void const *copyFrom) {
        CUCHECK(cudaMemcpy(ptr, copyFrom, size()*sizeof(T),
                                                cudaMemcpyHostToDevice));
    }
    void set (void const *copyFrom, size_t offset, size_t nElements) {
        T *pointer = (T*)ptr;
        CUCHECK(cudaMemcpy(pointer+offset, copyFrom, nElements*sizeof(T),
                                                cudaMemcpyHostToDevice));

    }

    //! Copy data to GPU memory location
    /*!
     * \param dest Pointer to the memory to which the data should be copied
     * \param stream CUDA stream object
     *
     * Copy data from the device to the GPU memory location specified. If a
     * stream object is passed, the data is copied asynchronously using this
     * stream object. Otherwise, the data is copied synchronously.
     */
    void copyToDeviceArray(void *dest, cudaStream_t stream = nullptr) const {
        if (stream) {
            CUCHECK(cudaMemcpyAsync(dest, ptr, n*sizeof(T),
                                            cudaMemcpyDeviceToDevice, stream));
        } else {
            CUCHECK(cudaMemcpy(dest, ptr, n*sizeof(T),
                                                    cudaMemcpyDeviceToDevice));
        }
    }

    //! Set all bytes in the array to a specific value
    /*!
     * \param val Value written to all bytes in the array
     *
     * WARNING: val must be a one byte value
     *
     * Set each byte in the array to the value specified by val.
     *
     * \todo For this function val needs to be converted to unsigned char
     * and this value is used.
     */
    void memset(int val) {
        CUCHECK(cudaMemset(ptr, val, n*sizeof(T)));
    }

    //! Set array elements to a specific value
    /*!
     * \param val Value the elements are set to
     *
     * Set all array elements to the value specified by the parameter val
     */
    void memsetByVal(const T &val) {
        mdAssert(sizeof(T) == 4  || sizeof(T) == 8 ||
                 sizeof(T) == 12 || sizeof(T) == 16,
                 "Type parameter incompatible size");
        MEMSETFUNC(ptr, &val, n, sizeof(T));
    }

private:
    //! Allocate memory
    void allocate() { CUCHECK(cudaMalloc(&ptr, n * sizeof(T))); cap = size(); }

    //! Deallocate memory
    void deallocate() {
        CUCHECK(cudaFree(ptr));
        n = 0;
        cap = 0;
        ptr = nullptr;
    }
};

#endif
