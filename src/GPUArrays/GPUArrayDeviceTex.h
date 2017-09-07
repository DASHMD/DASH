#pragma once
#ifndef GPUARRAYDEVICETEX_H
#define GPUARRAYDEVICETEX_H

#include <cuda_runtime.h>

#include "globalDefs.h"
#include "GPUArrayDevice.h"
#include "Logging.h"

void MEMSETFUNC(cudaSurfaceObject_t, void *, int, int);

//! Manager for data on a GPU Texture
/*!
 * \tparam T Type of data stored in the Texture
 *
 * This class manages data stored in a GPU Texture device. This type of memory
 * is often faster than the standard global or shared memory on the GPU. This
 * type of storage should be used only for runtime loop, not for general
 * storage.
 */
template <typename T>
class GPUArrayDeviceTex : public GPUArrayDevice {
public:

    //! Default constructor
    /*!
     * \param size Size (numer of elements) of the array
     */
    GPUArrayDeviceTex(size_t size = 0)
        : GPUArrayDevice(size), texObject(0), surfObject(0) { allocate(); }

    //! Constructor
    /*!
     * \param desc_ Channel descriptor
     */
    GPUArrayDeviceTex(cudaChannelFormatDesc desc_)
        : GPUArrayDevice(0), texObject(0), surfObject(0), channelDesc(desc_) {}

    //! Constructor
    /*!
     * \param size Size of the array (number of elements)
     * \param desc Channel descriptor
     */
    GPUArrayDeviceTex(size_t size, cudaChannelFormatDesc desc)
        : GPUArrayDevice(size), texObject(0), surfObject(0), channelDesc(desc)
    {
        allocate();
    }

    //! Copy constructor
    /*!
     * \param other GPUArrayDeviceTex to copy from
     */
    GPUArrayDeviceTex(const GPUArrayDeviceTex<T> &other)
        : GPUArrayDevice(other.size()), texObject(0), surfObject(0),
          channelDesc(other.channelDesc)
    {
        allocate();
        CUCHECK(cudaMemcpy2DArrayToArray(data(), 0, 0, other.data(), 0, 0,
                                         nX() * sizeof(T), nY(),
                                         cudaMemcpyDeviceToDevice));
    }

    //! Move constructor
    /*!
     * \param other GPUArrayDeviceTex containing the data to move
     */
    GPUArrayDeviceTex(GPUArrayDeviceTex<T> &&other) {
        // Take the values of the other object
        texObject = other.texObject;
        surfObject = other.surfObject;
        resDesc = other.resDesc;
        texDesc = other.texDesc;
        n = other.n;
        cap = other.cap;
        ptr = other.ptr;
        // Set the other object to zero
        other.texObject = 0;
        other.surfObject = 0;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;
    }

    //! Desctructor
    ~GPUArrayDeviceTex() {
        deallocate();
    }

    //! Assignment operator
    /*!
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayDeviceTex<T> &operator=(const GPUArrayDeviceTex<T> &other) {
        channelDesc = other.channelDesc;
        if (other.size()) {
            resize(other.size());
        }
        size_t x = nX();
        size_t y = nY();
        CUCHECK(cudaMemcpy2DArrayToArray(data(), 0, 0, other.data(), 0, 0,
                                         x*sizeof(T), y,
                                         cudaMemcpyDeviceToDevice));
        return *this;
    }

    //! Move assignment operator
    /*!
     * \param other Right hand side of assignment operator
     *
     * \return This object
     */
    GPUArrayDeviceTex<T> &operator=(GPUArrayDeviceTex<T> &&other) {
        // Clear this object
        deallocate();

        // Take the values of the other object
        texObject = other.texObject;
        surfObject = other.surfObject;
        resDesc = other.resDesc;
        texDesc = other.texDesc;
        n = other.n;
        cap = other.cap;
        ptr = other.ptr;

        // Set other object to zero
        other.texObject = 0;
        other.surfObject = 0;
        other.n = 0;
        other.cap = 0;
        other.ptr = nullptr;

        // Return this object
        return *this;
    }

    //! Access data pointer
    /*!
     * \return Pointer to device memory
     */
    cudaArray *data() { return (cudaArray *)ptr; }

    //! Const access to data pointer
    /*!
     * \return Pointer to const device memory
     */
    cudaArray const* data() const { return (cudaArray const*)ptr; }

    //! Access Texture Object
    /*!
     * \return Texture object to access current memory
     */
    cudaTextureObject_t tex() { createTexSurfObjs(); return texObject; }

    //! Access Surface Object
    /*!
     * \return Surface object to access current memory
     */
    cudaSurfaceObject_t surf() { createTexSurfObjs(); return surfObject; }

    //! Copy data from device to a given memory
    /*!
     * \param copyTo Pointer pointing to the memory taking the data
     * \param stream CUDA Stream for asynchronous copying
     *
     * This function copies data from the GPU device array to a given memory
     * location on the CPU host. If a stream object is specified, the data
     * will be copied asynchronously. Otherwise, it will be copied
     * synchronously.
     *
     * To copy data to a memory location on the GPU device, use
     * GPUArrayDeviceTex::copyToDeviceArray().
     */
     void get(void *copyTo, cudaStream_t stream = nullptr) const {
        if (copyTo == nullptr) { return; }
        size_t x = nX();
        size_t y = nY();

        if (stream) {
            CUCHECK(cudaMemcpy2DFromArrayAsync(copyTo, x * sizeof(T), data(),
                                               0, 0, x * sizeof(T), y,
                                               cudaMemcpyDeviceToHost, stream));
        } else {
            CUCHECK(cudaMemcpy2DFromArray(copyTo, x * sizeof(T), data(),
                                          0, 0, x * sizeof(T), y,
                                          cudaMemcpyDeviceToHost));
        }
    }

    //! Copy data from pointer to device
    /*!
     * \param copyFrom Pointer to memory where to copy from
     */
    void set(void const *copyFrom) {
        size_t x = nX();
        size_t y = nY();
        cudaMemcpy2DToArray(data(), 0, 0, copyFrom, x*sizeof(T),
                            x * sizeof(T), y, cudaMemcpyHostToDevice );
    }

    //! Copy data to GPU device
    /*!
     * \param dest Pointer to GPU memory
     * \param stream Optional stream object for asynchronous copying
     *
     * This function copies data from the array to another memory location on
     * the GPU device. If the function is called with a cudaStream object, the
     * data will be copied asynchronously using this stream. Otherwise, the
     * data will be copied synchronously.
     */
    void copyToDeviceArray(void *dest, cudaStream_t stream = nullptr) const {
        int numBytes = size() * sizeof(T);
        //! \todo Make sure this works for copying from 2d arrays
        if (stream) {
            CUCHECK(cudaMemcpyFromArrayAsync(dest, data(), 0, 0, numBytes,
                                            cudaMemcpyDeviceToDevice, stream));
        } else {
            CUCHECK(cudaMemcpyFromArray(dest, data(), 0, 0, numBytes,
                                                    cudaMemcpyDeviceToDevice));
        }
    }

    //! Set all bytes in the array to specific value
    /*!
     * \param val Value to set bytes
     *
     * This function sets all bytes to a specific value. To this end, val is
     * transformed to unsigned char. To set all elements of the array to a
     * specific value, use GPUArrayDeviceTex::memsetByVal().
     */
    void memset(int val)
    {
        size_t x = nX();
        size_t y = nY();
        CUCHECK(cudaMemset2D(data(), x*sizeof(T), val, x*sizeof(T), y));
    }

    //! Set all elements of GPUArrayDeviceTex to specific value
    /*!
     * \param val_ Value to set data to
     */
    void memsetByVal(T val_) {
        mdAssert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16,
                 "Type T has incompatible size");
        createTexSurfObjs();
        MEMSETFUNC(surfObject, &val_, size(), sizeof(T));
    }

private:
    //! Create Texture and Surface Objects
    /*!
     * Objects are only created if they don't exist yet.
     */
    void createTexSurfObjs() {
        if (texObject == 0) {
            resDesc.resType = cudaResourceTypeArray;
            texDesc.readMode = cudaReadModeElementType;
            cudaCreateTextureObject(&texObject, &resDesc, &texDesc, nullptr);
            // resDesc.res.array.array has been or will be set in allocate()
        }
        if (surfObject == 0) {
            cudaCreateSurfaceObject(&surfObject, &resDesc);
        }
    }

    //! Allocate memory on the Texture device
    void allocate() {
        size_t x = nX();
        size_t y = nY();
        CUCHECK(cudaMallocArray((cudaArray_t *)(&ptr), &channelDesc, x, y) );
        cap = x*y;
        //assuming address gets set in blocking manner
        resDesc.res.array.array = data();
    }

    //! Destroy Texture and Surface objects, deallocate memory
    void deallocate() {
        if (texObject != 0) {
            CUCHECK(cudaDestroyTextureObject(texObject));
            texObject = 0;
        }
        if (surfObject != 0) {
            CUCHECK(cudaDestroySurfaceObject(surfObject));
            surfObject = 0;
        }
        if (data() != nullptr) {
            CUCHECK(cudaFreeArray(data()));
        }
    }

    //! Get size in x-dimension of Texture Array
    /*!
     * \return Size in x-dimension
     */
    size_t nX() const {
        const size_t elementsPerLine = PERLINE/sizeof(T);
        return (size() > elementsPerLine) ? elementsPerLine : size();
    }

    //! Get size in y-dimension of Texture Array
    /*!
     * \return Size in y-dimension
     */
    size_t nY() const {
        const size_t elementsPerLine = PERLINE/sizeof(T);
        return (size() % elementsPerLine == 0) ? size() / elementsPerLine
                                               : size() / elementsPerLine + 1;
    }

private:
    cudaTextureObject_t texObject; //!< Texture object
    cudaSurfaceObject_t surfObject; //!< Texture surface
    cudaResourceDesc resDesc; //!< Resource descriptor
    cudaTextureDesc texDesc; //!< Texture descriptor
    cudaChannelFormatDesc channelDesc; //!< Descriptor for the texture
};

#endif
