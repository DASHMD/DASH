#pragma once
#ifndef GPUARRAYTEX_H
#define GPUARRAYTEX_H

#include <vector>

#include "GPUArray.h"
#include "GPUArrayDeviceTex.h"

//! Manage data on the CPU and a GPU Texture
/*!
 * \tparam T Type of data stored on the CPU and GPU
 *
 * This class manages data stored on the CPU and a GPU texture device. The
 * class allocates memory both on the CPU and the GPU and transfers data
 * between the two. This class is designed only for use in the runtime loop,
 * not for general storage.
 */
template <typename T>
class GPUArrayTex : public GPUArray {
    public:
        GPUArrayDeviceTex<T> d_data; //!< Array storing data on the GPU
        std::vector<T> h_data; //!< Array storing data on the CPU

        //! Default constructor
        GPUArrayTex() {
        }

        //! Constructor
        /*!
         * \param desc_ Cuda channel descriptor for asynchronous data transfer
         */
        GPUArrayTex(cudaChannelFormatDesc desc_) : d_data(desc_) {
        }

        //! Constructor
        /*!
         * \param vals Vector containing data
         * \param desc_ Cuda channel descriptor for asynchronous data transfer
         *
         * This constructor allocates memory on the CPU and GPU large enough to
         * fit the data given in the vector. Then, it copies the data to the
         * CPU memory. The GPU memory remains unset.
         */
        GPUArrayTex(std::vector<T> vals, cudaChannelFormatDesc desc_)
            : d_data(vals.size(), desc_)
        {
            set(vals);
        }

        //! Set the CPU memory
        /*!
         * \param other Vector containing data
         * \return True always
         *
         * Copy data from vector to the CPU memory.
         */
        bool set(std::vector<T> &other) {
            d_data.resize(other.size());
            h_data = other;
            h_data.reserve(d_data.capacity());
            return true;
        }

        //! Send data from CPU to GPU
        void dataToDevice() {
            d_data.set(h_data.data());
        }

        //! Send data from GPU to CPU
        void dataToHost() {
            d_data.get(h_data.data());
        }

        //! Return number of elements stored in array
        /*!
         * \return Number of elements
         */
        size_t size() const { return h_data.size(); }

        //! Resize the GPU array to be large enough to contain CPU data
        /*! \todo This function should not be necessary
         */
        void ensureSize() {
            d_data.resize(h_data.size());
        }

        //! Copy data from GPU to CPU asynchronously
        /*!
         * \param stream CUDA Stream object
         *
         * \todo It would be nicer to call dataToHost and have it copy
         *       asynchronously if a stream is passed and synchronously
         *       otherwise
         */
        void dataToHostAsync(cudaStream_t stream) {
            d_data.get(h_data.data(), stream);
        }

        //! Copy data from GPU texture to other GPU memory
        /*!
         * \param dest Pointer to GPU memory, destination for copy
         */
        void copyToDeviceArray(void *dest) { //DEST HAD BETTER BE ALLOCATED
            int numBytes = size() * sizeof(T);
            copyToDeviceArrayInternal(dest, d_data.data(), numBytes);

        }

        //! Return texture object from GPUArrayDeviceTex
        /*!
         * \return Cuda Texture Object used for GPU memory storage
         */
        cudaTextureObject_t getTex() {
            return d_data.tex();
        }

        //! Return surface object from GPUArrayDeviceTex
        /*!
         * \return Cuda Surface Object used to write to GPU texture memory
         */
        cudaSurfaceObject_t getSurf() {
            return d_data.surf();
        }

        //! Set all elements to a given value
        /*!
         * \param val Value the elements are set to
         */
        void memsetByVal(T val) {
            d_data.memsetByVal(val);
        }
};

#endif
