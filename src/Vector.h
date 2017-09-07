#pragma once
#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <string>
#include <sstream>

#include "cuda_runtime.h"

void export_Vector();
void export_VectorInt();

/*! \class VectorGeneric
 * \brief A three-element vector
 *
 * \tparam T Type of data stored in the vector.
 *
 * This class defines a simple three-element vector and the corresponding
 * vector operations.
 *
 */
template <typename T>
class VectorGeneric {
private:
    T vals[3]; //!< Array storing the values

public:
    /*! \brief Default constructor */
    VectorGeneric<T> () {
        vals[0] = vals[1] = vals[2] = 0;
    }

    /*! \brief Constructor
     *
     * \param x First element as double
     * \param y Second element as double
     * \param z Third element as double
     *
     */
    VectorGeneric<T> (const T &x, const T &y, const T &z) {
        vals[0] = x;
        vals[1] = y;
        vals[2] = z;
    }

    /*! \brief Constructor from pointer
     *
     * \param vals_ Pointer to three element int array
     *
     */
    VectorGeneric<T> (T *vals_) {
        for (int i=0; i<3; i++) {
            vals[i] = (T) vals_[i];
        }
    }

    /*! \brief Constructor from float3
     *
     * \param other Float3 to use as values for the vector
     */
    VectorGeneric<T> (float3 other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
    }

    /*! \brief Constructor from float4
     *
     * \param other Float4 to use as values for the vector
     *
     * The forth value in the float4 will be discarded.
     */
    VectorGeneric<T> (float4 other) {
        vals[0] = other.x;
        vals[1] = other.y;
        vals[2] = other.z;
    }

    /*! \brief Conversion operator
     *
     * Used to convert VectorGeneric<T> into VectorGeneric<U> as in
     * VectorInt = Vector or, more explicit v2 = (VectorInt)v1; The conversion
     * is done by converting each of the three vector elements.
     */
    template<typename U>
    operator VectorGeneric<U> () const {
        return VectorGeneric<U>( (U)vals[0], (U)vals[1], (U)vals[2] );
    }

    /*! \brief Convert vector to float4
     *
     * \return float4 containing values of this Vector
     *
     * The first three entries correspond to the vector elements, the forth
     * entry will be set to zero.
     */
    float4 asFloat4() const {
        return make_float4(vals[0], vals[1], vals[2], 0);
    }

    /*! \brief Convert vector to int4
     *
     * \return int4 containing values from this Vector
     *
     * The first three entries correspond to the vector elements, the forth
     * entry will be set to zero.
     */
    int4 asInt4() const {
        return make_int4(vals[0], vals[1], vals[2], 0);
    }

    /*! \brief Convert vector to float3 */
    float3 asFloat3() const {
        return make_float3(vals[0], vals[1], vals[2]);
    }

    /*! \brief Convert vector to int3 */
    int3 asInt3() const {
        return make_int3(vals[0], vals[1], vals[2]);
    }

    /*! \brief Set all vector elements to zero */
    void zero() {
        vals[0] = vals[1] = vals[2] = 0;
    }

    /*! \brief Sum of all entries */
    T sum() const {
        return vals[0] + vals[1] + vals[2];
    }

    /*! \brief Product of all entries */
    T prod() const {
        return vals[0] * vals[1] * vals[2];
    }

    /*! \brief Operator accessing vector elements */
    T &operator[]( int n ) {
        return vals[n];
    }

    /*! \brief Const operator accessing vector elements */
    const T &operator[]( int n ) const {
        return vals[n];
    }

    /*! \brief Convert all elements to their absolute value
     *
     * \returns A new vector with the transformed elements.
     */
    VectorGeneric<T> abs() const {
        return VectorGeneric<T>(std::abs(vals[0]), std::abs(vals[1]), std::abs(vals[2]));
    }

    /*! \brief Unary minus operator */
    VectorGeneric<T> operator-() const {
        return VectorGeneric<T>(-vals[0], -vals[1], -vals[2]);
    }

    /*! \brief rotation in x-y plane
     *
     * \param rotation Rotation angle
     * \return New rotated Vector
     *
     * The z-component of the vector remains unchanged.
     */
    VectorGeneric<double> rotate2d( double rotation) const {
        double c = cos(rotation);
        double s = sin(rotation);
        return VectorGeneric<double> (c*vals[0] - s*vals[1], s*vals[0] + c*vals[1], vals[2]);
    }

    /*! \brief Multiplication with generic type */
    template<typename U>
    auto operator*( const U &scale ) const -> VectorGeneric< decltype(vals[0]*scale) > {
        VectorGeneric< decltype(vals[0]*scale) > newVec(*this);
        newVec *= scale;
        return newVec;
    }

    /*! \brief Element-wise multiplication with other vector */
    template<typename U>
    auto operator*( const VectorGeneric<U> &q ) const -> VectorGeneric< decltype(vals[0]*q[0]) > {
        VectorGeneric< decltype(vals[0]*q[0]) > newVec(*this);
        newVec *= q;
        return newVec;
    }

    /*! \brief Division with generic type */
    template<typename U>
    auto operator/( const U &scale ) const -> VectorGeneric< decltype(vals[0]/scale) > {
        VectorGeneric< decltype(vals[0]/scale) > newVec(*this);
        newVec /= scale;
        return newVec;
    }

    /*! \brief Element-wise division with other vector */
    template<typename U>
    auto operator/( const VectorGeneric<U> &q ) const -> VectorGeneric< decltype(vals[0]/q[0]) > {
        VectorGeneric< decltype(vals[0]/q[0]) > newVec(*this);
        newVec /= q;
        return newVec;
    }

    /*! \brief Addition of two vectors */
    template<typename U>
    auto operator+( const VectorGeneric<U> &q ) const -> VectorGeneric< decltype(vals[0]+q[0]) > {
        VectorGeneric< decltype(vals[0]+q[0]) > newVec(*this);
        newVec += q;
        return newVec;
    }

    /*! \brief Subtraction of two vectors */
    template<typename U>
    auto operator-( const VectorGeneric<U> &q ) const -> VectorGeneric< decltype(vals[0]-q[0]) > {
        VectorGeneric< decltype(vals[0]-q[0]) > newVec(*this);
        newVec -= q;
        return newVec;
    }

    /*! \brief Multiplication-assignment operator with scalar value
     *
     * \param scale Scale factor, value to be multiplied with each element
     * \return This Vector
     */
    template<typename U>
    const VectorGeneric<T> &operator*=( const U &scale ){
        vals[0]*=scale;vals[1]*=scale;vals[2]*=scale;return *this;
    }

    /*! \brief Multiplication-assignment operator with other vector
     *
     * \param q Other Vector used for element-wise multiplication
     * \return This Vector
     *
     * Performs element-wise multiplication.
     */
    template<typename U>
    const VectorGeneric<T> &operator*=( const VectorGeneric<U> &q ){
        vals[0]*=q[0];vals[1]*=q[1];vals[2]*=q[2];return *this;
    }

    /*! \brief Division-assignment operator with int
     *
     * \param scale Scale factor by which each element is divided
     * \return This Vector
     */
    template<typename U>
    const VectorGeneric<T> &operator/=( const U &scale ){
        vals[0]/=scale;vals[1]/=scale;vals[2]/=scale;return *this;
    }

    /*! \brief Division-assignment operator with other vector
     *
     * \param q Vector for element-wise division
     * \return This Vector
     *
     * Performs element-wise division.
     */
    template<typename U>
    const VectorGeneric<T> &operator/=( const VectorGeneric<U> &q ){
        vals[0]/=q[0];vals[1]/=q[1];vals[2]/=q[2];return *this;
    }

    /*! \brief Addition-assignment operator */
    template<typename U>
    const VectorGeneric<T> &operator+=( const VectorGeneric<U> &q ){
        vals[0]+=q[0];vals[1]+=q[1];vals[2]+=q[2];return *this;
    }

    /*! \brief Subtraction-assigment operator */
    template<typename U>
    const VectorGeneric<T> &operator-=( const VectorGeneric<U> &q ){
        vals[0]-=q[0];vals[1]-=q[1];vals[2]-=q[2];return *this;
    }

    /*! \brief Smaller than comparison operator
     *
     * \param q Other Vector to compare this Vector to
     * \return True if this Vector is smaller than other Vector
     *
     * The comparison is element-wise. v < q returns true if v[0] < q[0]
     * or (v[0] == q[0] && v[1] < q[1]) or (v[0] == q[0] && v[1] == q[1] &&
     * v[2] < q[2]).
     */
    template<typename U>
    bool operator<( const VectorGeneric<U> &q ) const {
        if( vals[0] != q[0] ) { return vals[0] < q[0]; }
        if( vals[1] != q[1] ) { return vals[1] < q[1]; }
        return vals[2] < q[2];
    }

    /*! \brief Larger than comparison operator
     *
     * \param q Other Vector to compare this Vector to
     * \return True if this Vector is larger than other Vector
     *
     * The comparison is element-wise. Thus, v > q returns true if v[0] > q[0]
     * or (v[0] == q[0] && v[1] > q[1]) or (v[0] == q[0] && v[1] == q[1] &&
     * v[2] > q[2]).
     */
    template<typename U>
    bool operator>( const VectorGeneric<U> &q ) const {
        if( vals[0] != q[0] ) { return vals[0] > q[0]; }
        if( vals[1] != q[1] ) { return vals[1] > q[1]; }
        return vals[2] > q[2];
    }

    /*! \brief Equality comparison operator */
    template<typename U>
    bool operator==( const VectorGeneric<U> &q ) const {
        return vals[0] == q[0] && vals[1] == q[1] && vals[2] == q[2];
    }

    /*! \brief Non-equal comparison operator */
    template<typename U>
    bool operator!=( const VectorGeneric<U> &q )const{
        return !(*this == q);
    }

    /*! \brief Larger-equal comparison operator
     *
     * \param q Other Vector to compare this Vector to
     * \return True if This Vector is larger or equal to other Vector
     */
    template<typename U>
    bool operator>=( const VectorGeneric<U> &q ) const {
        return (*this == q) || (*this > q);
    }

    /*! \brief Smaller-equal comparison operator
     *
     * \param q Other Vector to compare this Vector to
     * \return True if this Vector is smaller or equal to other Vector
     */
    template<typename U>
    bool operator<=( const VectorGeneric<U> &q ) const {
        return (*this == q) || (*this < q);
    }

    /*! \brief Dot product with another vector */
    template<typename U>
    auto dot( const VectorGeneric<U> &q ) const -> decltype(vals[0]*q[0]+vals[1]*q[1]) {
        return vals[0]*q[0]+vals[1]*q[1]+vals[2]*q[2];
    }

    /*! \brief Cross product with another vector */
    template<typename U>
    auto cross( const VectorGeneric<U> &q ) const -> VectorGeneric< decltype(vals[1]*q[2] - vals[2]*q[1]) > {
        return VectorGeneric< decltype(vals[1]*q[2] - vals[2]*q[1]) >( vals[1]*q[2]-vals[2]*q[1],vals[2]*q[0]-vals[0]*q[2],vals[0]*q[1]-vals[1]*q[0] );
    }

    /*! \brief Length of vector */
    auto len() const -> decltype(std::sqrt(vals[0]*vals[0]+vals[1]*vals[1])) {
        return std::sqrt(this->lenSqr());
    }

    /*! \brief Squared length of vector */
    auto lenSqr() const -> decltype(vals[0]*vals[0]+vals[1]*vals[1]) {
        return this->dot(*this);
    }

    /*! \brief Distance between two points
     *
     * \param q Vector determining other point
     * \return Distance between the points
     *
     * The points are specified by this and by the q vector.
     */
    template<typename U>
    auto dist( const VectorGeneric<U> &q ) const -> decltype(std::sqrt((vals[0]-q[0])*(vals[0]-q[0]))) {
        return (*this - q).len();
    }

    /*! \brief Squared distance between two points */
    template<typename U>
    auto distSqr( const VectorGeneric<U> &q) const -> decltype((vals[0]-q[0])*(vals[0]-q[0])) {
        return (*this -q).lenSqr();
    }

    /*! \brief Return normalized form of this vector */
    auto normalized() const -> VectorGeneric< decltype(vals[0]/std::sqrt(vals[0])) > {
        return *this/this->len();
    }

    /*! \brief Normalize this vector */
    void normalize(){
        *this /= this->len();
    }
	VectorGeneric<T> copy() const {
		return VectorGeneric<T>(vals[0], vals[1], vals[2]);
	}

    /*! \brief Mirror vector along y direction */
    VectorGeneric<T> perp2d() const {
        return VectorGeneric<T>(vals[1], -vals[0], vals[2]);
    }

    /*! \brief Convert vector to string for output */
    std::string asStr() const {
        std::ostringstream oss;
        oss << "(" << vals[0] << ", " << vals[1] << ", " << vals[2] << ")";
        return oss.str();
    }

    /*! \brief Get a specific element */
    T get(int i) const {
        return vals[i];
    }

    /*! \brief Set a specific element */
    void set(int i, const T &val) {
        vals[i] = val;
    }

    /*! \brief Distance between two points, with periodic boundary conditions
     *
     * \param other Second point for distance calculation
     * \param trace X-, y-, and z- length of the simulation bounding box
     * \return Distance vector
     *
     * This function calcuates the distance from this vector to another vector
     * taking periodic boundary conditions into accound. Thus calculating the
     * minimum distance between the vectors.
     */
    VectorGeneric<T> loopedVTo(const VectorGeneric<T> &other, const VectorGeneric<T> &trace) const {
        VectorGeneric<T> dist = other - *this;
        VectorGeneric<T> halfTrace = trace/ (T) 2.0;
        for (int i=0; i<3; i++) {
            if (dist[i] > halfTrace[i]) {
                dist[i] -= trace[i];
            } else if (dist[i] < -halfTrace[i]) {
                dist[i] += trace[i];
            }
        }
        return dist;
    }
};

typedef VectorGeneric<double> Vector;
typedef VectorGeneric<int> VectorInt;
#define EPSILON 0.00001f
const Vector VectorEps(EPSILON, EPSILON, EPSILON);

std::ostream &operator<<(std::ostream &os, const Vector &v);
std::ostream &operator<<(std::ostream &os, const float4 &v);

#endif
