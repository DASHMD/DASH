#pragma once
#ifndef GPUARRAY_H
#define GPUARRAY_H

//! Base class for a GPUArray
class GPUArray {
    protected:
        //! Constructor
        GPUArray() = default;

    public:
        //! Destructor
        virtual ~GPUArray(){};

        //! Send data from host to GPU device
        virtual void dataToDevice() = 0;

        //! Send data from GPU device to host
        virtual void dataToHost() = 0;

        //! Return number of elements of array
        /*!
         * \return Number of elements
         *
         * This function returns the number of elements in the array. Note,
         * that this is not the size in bytes. For this, use size()*sizeof(T),
         * where T is the class used in the GPUArray.
         */
        virtual size_t size() const = 0;
};

#endif
