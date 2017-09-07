#pragma once
#include "globalDefs.h"
//! Global function returning a single SquareVector item
/*!
 * \param vals Pointer to SquareVector array
 * \param nCol Number of columns
 * \param i Row
 * \param j Column
 *
 * \returns Element (i,j) from vals
 *
 * This function returns a single element from a given SquareVector array.
 */
template <class T>
__host__ __device__ T squareVectorItem(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}
inline __device__ int squareVectorIndex(int nCol, int i, int j) {
    return i*nCol + j;
}

//! Global function returning a reference to a single Square Vector item
/*!
 * \param vals Pointer to SquareVector array
 * \param nCol Number of columns
 * \param i Row
 * \param j Column
 *
 * \returns Reference to element (i,j) from vals
 *
 * This function returns a reference to specific entry of a given SquareVector
 * array.
 */
template <class T>
__host__ __device__ T &squareVectorRef(T *vals, int nCol, int i, int j) {
    return vals[i*nCol + j];
}

//! Two dimensional array
/*!
 * This namespace contains functions to create std::vector elements which can
 * be treated as two-dimensional arrays.
 */
namespace SquareVector {

    //! Create SquareVector
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param size Number of rows and columns
     *
     * \returns New SquareVector
     *
     * This function creates a new SquareVector
     */
    template <class T>
    std::vector<T> create(int size) {
        return std::vector<T>(size*size, DEFAULT_FILL);
    }

    //! Set the diagonal elements of the SquareVector
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param vec Pointer to the vector to be modified
     * \param size Number of rows of the square vector
     * \param fillFunction Function taking no argument, returning the values
     *                     for the diagonal elements
     *
     * Set the diagonal elements to a value determined by the function passed.
     */
    template <class T>
    void populateDiagonal(std::vector<T> *vec, int size,
            std::function<T ()> fillFunction) {
        for (int i=0; i<size; i++) {
            T val = squareVectorRef<T>(vec->data(), size, i, i);
            if (val == DEFAULT_FILL) {
                squareVectorRef<T>(vec->data(), size, i, i) = fillFunction();
            }
        }
    }

    //! Fill SquareVector with values
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param vec Pointer to SquareVector
     * \param size Number of Rows/Columns of the SquareVector
     * \param fillFunction Funtion pointer to the function used to determine
     *                     the elements in the SquareVector
     *
     * This function can be used to set the off-diagonal elements of the
     * SquareVector. The off-diagonal elements are calculated base on the
     * diagonal elements which are passed to the fillFunction. This function
     * only sets values that have not been set before.
     *
     * \todo I think this function should overwrite values previously set
     *       instead of silently doing nothing.
     */
    template <class T>
    void populate(std::vector<T> *vec, int size, std::function<T (T, T)> fillFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                T val = squareVectorRef<T>(vec->data(), size, i, j);
                if (i==j) {
                    if (val == DEFAULT_FILL) {
                        std::cout << "You have not defined interaction parameters "
                            "for atom type with index " << i << std::endl;
                        assert(val != DEFAULT_FILL);
                    }
                } else if (val == DEFAULT_FILL) {
                    squareVectorRef<T>(vec->data(), size, i, j) =
                        fillFunction(squareVectorRef<T>(vec->data(), size, i, i),
                                     squareVectorRef<T>(vec->data(), size, j, j));
                }
            }
        }
    }
    //for Fcut LJFS 
    template <class T>
    void populate(std::vector<T> *vec, int size, std::function<T (int, int)> fillFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                squareVectorRef<T>(vec->data(), size, i, j) = fillFunction(i,j);
            }
        }
    }        

    
    //in case you want it flag any unfilled parameters
    
    template <class T>
    void check_populate(std::vector<T> *vec, int size) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                T val = squareVectorRef<T>(vec->data(), size, i, j);
                if (val == DEFAULT_FILL) {
                    std::cout << "You have not defined interaction parameters "
                        "for atom types with indices " << i <<" "<< j << std::endl;
                    assert(val != DEFAULT_FILL);
                }
            }
        }
    }    
    //! Call function on each element of the SquareVector
    /*!
     * \tparam Type of data stored in the SquareVector
     * \param vec Pointer to the vector to be modified
     * \param size Number of rows in the SquareVector
     * \param processFunction Function to be called on each element
     *
     * Call a function on each element in the SquareVector, taking the current
     * value as the argument and replacing it with the return value.
     */
    template <class T>
    void process(std::vector<T> *vec, int size, std::function<T (T)> processFunction) {
        for (int i=0; i<size; i++) {
            for (int j=0; j<size; j++) {
                squareVectorRef<T>(vec->data(), size, i, j) =
                        processFunction(squareVectorRef<T>(vec->data(), size, i, j));
            }
        }
    }

    //! Copy SquareVector onto another SquareVector with a different size
    /*!
     * \tparam T Type of data stored in the SquareVector
     * \param other Reference to old SquareVector
     * \param oldSize Number of Rows/Columns of old SquareVector
     * \param newSize Number of Rows/Columns of new SquareVector
     *
     * \returns New SquareVector
     *
     * This Function copies a SquareVector and gives it a new size.
     */
    template <class T>
    std::vector<T> copyToSize(std::vector<T> &other, int oldSize, int newSize) {
        std::vector<T> replacement(newSize*newSize, DEFAULT_FILL);
        int copyUpTo = std::fmin(oldSize, newSize);
        for (int i=0; i<copyUpTo; i++) {
            for (int j=0; j<copyUpTo; j++) {
                squareVectorRef<T>(replacement.data(), newSize, i, j) =
                            squareVectorItem<T>(other.data(), oldSize, i, j);
            }
        }
        return replacement;
    }
} // namespace SquareVector

//! Fix for pair interactions
/*!
 * This fix is the parent class for all types of pair interaction fixes.
 */
