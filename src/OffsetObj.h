#pragma once
#ifndef OFFSETOBJ_H
#define OFFSETOBJ_H
#include "Vector.h"

/*! \brief Class to store an Object together with an offset
 *
 * \tparam T Type of the object stored
 *
 * This class is designed to store an Object together with a 3-dimensional
 * Vector. This can, for example, be the position of the Object.
 */
template <class T>
class OffsetObj {
public:
    T obj; //!< Object
    Vector offset; //!< Offset

    /*! \brief Constructor
     *
     * \param obj_ Object to be stored
     * \param offset_ Offset
     */
    OffsetObj (T &obj_, Vector offset_) : obj(obj_), offset(offset_) {};

    /*! \brief Constructor */
    OffsetObj () : obj(T()), offset(Vector()) {};

    /*! \brief Equal-to operator
     *
     * \param other Other object to be compared to
     * \return True if Objects and Offsets are equal
     *
     * Two OffsetObjects are considered equal if they store the same Object
     * and the Offsets differ by less than EPSILON in each direction (see
     * Vector).
     */
    bool operator==(const OffsetObj<T> &other) {
        return obj == other.obj && (offset-other.offset).abs() < VectorEps;
    }

    /*! \brief Non-equal operator
     *
     * \param other Other OffsetObject this should be compared to
     * \return False if OffsetObjects are equal
     */
    bool operator!=(const OffsetObj<T> &other) {
        return !(*this == other);
    }
};
#endif

