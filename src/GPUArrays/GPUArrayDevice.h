#pragma once
#ifndef GPUARRAYDEVICE_H
#define GPUARRAYDEVICE_H

#include <cstddef>
#include <cuda_runtime.h>

//! Base class for GPUArrayDevices
/*!
 * A GPUArrayDevice is a memory-managed pointer for storgage on the GPU. It is
 * mainly used by the GPUArray which manages the GPU memory and takes care of
 * sending data from CPU to GPU and back.
 *
 * This class is a Base class defining the function common to all memory
 * operations on the GPU. The child classes differ in which type of memory
 * they store the data: Global memory or Texture memory. Not yet
 * used/implemented is memory stored to Constant memory or Local memory.
 */
class GPUArrayDevice {
protected:
    //! Constructor
    /*!
     * \param size Size of the array
     */
    GPUArrayDevice(size_t size = 0) : n(size), cap(0), ptr(nullptr) {}

public:
    //! Destructor
    virtual ~GPUArrayDevice(){};

    //! Get the size of the array
    /*!
     * \return Number of elements stored in the array
     */
    size_t size() const { return n; }

    //! Get the capacity of the array
    /*!
     * \return Capacity
     *
     * The capacity is the number of elements that can be stored in the
     * currently allocated memory.
     */
    size_t capacity() const { return cap; }

    //! Change size of the array
    /*!
     * \param newSize New size of the array
     * \param force Force reallocation of memory
     * \return True if memory is reallocated. Else return false.
     *
     * Resize the array. If newSize is larger than the capacity, the current
     * memory is deallocated and new memory is allocated. In this case, the
     * stored memory is lost and the function returns false.
     */
    virtual bool resize(size_t newSize, bool force = false);

    //! Get function for device arrays
    /*!
     * \param copyTo Pointer to CPU host memory location
     * \param stream CUDA stream object
     *
     * This function copies data from the device array to the specified
     * memory location on the CPU. To copy data to a different memory location
     * on the GPU, use GPUArrayDevice::copyToDevice().
     *
     * This virtual function is implemented by the child classes.
     */
    virtual void get(void *copyTo, cudaStream_t stream = nullptr) const = 0;

    //! Set function to copy data to device array
    /*!
     * \param copyFrom Memory location on the CPU to copy data from
     *
     * This function copies data from the CPU to the device array.
     */
    virtual void set(void const *copyFrom) = 0;

    //! Copy data to GPU memory location
    /*!
     * \param dest Pointer to the memory to which the data should be copied
     * \param stream CUDA stream object
     *
     * Copy data from the device to the GPU memory location specified. If a
     * stream object is passed, the data is copied asynchronously using this
     * stream object. Otherwise, the data is copied synchronously.
     */
    virtual void copyToDeviceArray(void *dest,
                                   cudaStream_t stream = nullptr) const = 0;

    //! Set all bytes in the array to a specific value
    /*!
     * \param val Value the elements are set to
     *
     * Set all bytes to the value specified in val. Note that val will be cast
     * to unsigned char. To set all values of the array, use memsetByVal
     */
    virtual void memset(int val) = 0;

private:
    //! Allocate memory for the array
    virtual void allocate() = 0;

    //! Deallocate memory
    virtual void deallocate() = 0;

protected:
    size_t n; //!< Number of elements stored in the array
    size_t cap; //!< Capacity of allocated memory
    void *ptr; //!< Pointer to memory location
};

#endif

