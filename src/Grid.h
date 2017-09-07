#pragma once
#ifndef GRID_H
#define	GRID_H

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>

#include "Bounds.h"
#include "Vector.h"
#include "GridGPU.h"

/*! \brief Simulation grid
 *
 * \tparam T Object type to store in the grid
 *
 * This class helps storing certain objects in a Grid. Typically, this class
 * is used as a Base class for a more specialized type of Grid, e.g. the
 * AtomGrid.
 *
 * The grid contains ns.x * ns.y * ns.z elements. These elements can be
 * accessed via the linear index or the three Grid indizes. Furthermore, using
 * ds, Grid coordinates are mapped onto physical coordinates and vice versa.
 */
template <class T>
class Grid {

public:
    std::vector<T> saved; //!< Saved state of the Grid
    std::vector<T> raw; //!< Raw state of the Grid
    Vector trace; //!< Vector containing trace for periodic boundary conditions

    VectorInt ns; //!< Number of grid points in each direction
    Vector ds; //!< Grid resolution in each direction
    Vector os; //!< Point of origin

    Vector dsOrig; //!< Original resolution of the grid

    T fillVal; //!< Default value
    bool is2d; //!< True for 2d Simulations

    /*! \brief Default constructor */
    Grid() = default;

    /*! \brief Constructor
     *
     * \param ns_ Number of grid points in each direction
     * \param ds_ Grid resolution in each direction
     * \param os_ Point of origin
     * \param fillVal_ Default value for elements
     */
    Grid(VectorInt ns_, Vector ds_, Vector os_, T fillVal_)
        : ns(ns_), ds(ds_), os(os_), dsOrig(ds), fillVal(fillVal_)
    {
        assert(ns[0] > 0 and ns[1] > 0 and ns[2] > 0);
        int n = ns[0]*ns[1]*ns[2];
        raw.reserve(n);
        for (int i=0; i<n; i++) {
            raw.push_back(fillVal);
        }
    };

    /*! \brief Set all values to their default */
    void fillVals() {
        assert(ns[0] > 0 and ns[1] > 0 and ns[2] > 0);
        int n = ns[0]*ns[1]*ns[2];
        raw = std::vector<T>();
        raw.reserve(n);
        for (int i=0; i<n; i++) {
            raw.push_back(fillVal);
        }
    }

    /*! \brief Stores into loop in which mirror image x is
     *
     * \param x Grid point
     * \param nd Number of grid points in this direction
     * \param loop Variable to store the mirror image to
     * \return Value of x wrapped into original grid
     */
    int loopDim(const int x, const int nd, int *loop) {
        //only works if one of fewer grids offset.  Could do some flooring,
        // but meh.  Not a relevant case.
        if (x < 0) {
            *loop = -1;
            return nd + x;
        } else if (x >= nd) {
            *loop = 1;
            return x - nd;
        }
        *loop = 0;
        return x;
    };

    /*! \brief Access element in Grid
     *
     * \param x x-dimension of grid element
     * \param y y-dimension of grid element
     * \param z z-dimension of grid element
     * \return Reference to element
     *
     * Access element in the Grid using three coordinates
     */
    T &operator()(const int x, const int y, const int z) {
    //	int idx = x*ny*nz + y*nz + z;
    //	if (idx >= raw.size()) {
    //		cout << "GETTING OUT OF BOUNDS GRID SQUARE" << endl;cout.flush();
    //	}
        return raw[x*ns[1]*ns[2] + y*ns[2] + z];
    };

    /*! \brief Get linear Grid index from three coordinates
     *
     * \param coord Vector storing the x, y, and z-grid coordinate
     * \return Linear index of the element
     *
     * Convert the three Grid coordinates to a linear index.
     */
    int idxFromCoord(VectorInt &coord) {
        return coord[0] * ns[1]*ns[2] + coord[1]*ns[2] + coord[2];
    }

    /*! \brief Get Object at physical coordinate
     *
     * \param v Vector in physical space
     * \return Object at this point
     *
     * Convert the physical point to a Grid point and return the corresponding
     * Object.
     */
    T &operator()(Vector &v) {
        VectorInt coord = (v - os) / ds;
        return raw[idxFromCoord(coord)];
    };

    /*! \brief Get Object at a Grid coordinate
     *
     * \param coords Pointer to 3-element array storing Grid coordinates
     * \param didLoop Pointer to 3-element array to store if atoms are wrapped
     * \return Object at Grid coordinates
     *
     * Select an object a given Grid coordinates.
     */
    T &operator() (int coords[3], int didLoop[3]) {
        int loopRes[3];
        for (int i=0; i<3; i++) {
            loopRes[i] = loopDim(coords[i], ns[i], &(didLoop[i]));
        }
        return (*this)(loopRes[0], loopRes[1], loopRes[2]);
    };

    /*! \brief Convert Grid coordinates to physical coordinates
     *
     * \param x Grid coordinate in x dimension
     * \param y Grid coordinate in y dimension
     * \param z Grid coordinate in z dimension
     * \return Physical position
     */
    Vector pos(const int x, const int y, const int z) {
        return os + Vector(ds[0]*x, ds[1]*y, ds[2]*z);
    };

    /*! \brief Convert physical coordinates to linear index
     *
     * \param v Vector of physical coordinates
     * \return Linear index
     */
    int idxFromPos(Vector &v) {
        VectorInt coord = (v - os) / ds;
        return idxFromCoord(coord);
    }

    /*! \brief Get physical coordinates from linear index
     *
     * \param i Linear index
     * \return Physical position
     */
    Vector pos(int i) {
        int numInSlice = ns[1] * ns[2];
        int nx = i / numInSlice;
        i -= nx * numInSlice;
        int ny = i / ns[2];
        i -= ny * ns[2];
        int nz = i;
        return pos(nx, ny, nz);
    }

    /*! \brief Get physical coordinates from pointer
     *
     * \param elem Pointer to Grid element
     * \return Physical coordinates
     *
     * If the Element is within the Grid Array, return its physical position.
     * Else, return (0, 0, 0).
     */
    Vector posElem(T *elem) {
        int i = elem - &raw[0];
        if (i >= 0 and i < raw.size()) {
            return pos(i);
        }
        return Vector(0, 0, 0);
    }

    /*! \brief Set the raw vector
     *
     * \param vals Vector containing the values
     */
    void setRaw(std::vector<T> vals) {
        raw = vals;
    };

    /*! \brief Set the saved vector
     *
     * \param vals Vector containing the values to be stored to saved
     */
    void setSaved(std::vector<T> vals) {
        saved = vals;
    }

    /*! \brief Reset Grid */
    void reset() {
        raw = saved;
    }

    /*! \brief Save Elements */
    void saveRaw() {
        saved = raw;
    }

    /*! \brief Compile vector of neighbors of grid coordinates
     *
     * \param coords Pointer to three-element array storing Grid coordinates
     * \param loops Pointer storing if boundaries are periodic
     * \param trace Trace of the boundary box
     * \return vector storing neigbors and the mirror box they occupy
     *
     * Compile vector of neighbors to a given Grid coordinate. Neighbors are
     * objects at the 8 nearest and next-nearest neighbor coordinates. The
     * returned vector consists of the neighbor objects and the mirror box
     * they are currently in.
     *
     * \todo Inner Loop uses same variable as outer loop!
     */
    std::vector<OffsetObj<T*> > getNeighbors(int coords[3], bool loops[3], Vector trace)
    {
        const int x = coords[0];
        const int y = coords[1];
        const int z = coords[2];
        int zBounds[2];
        if (not is2d) {
            zBounds[0] = z-1;
            zBounds[1] = z+1;
        } else {
            zBounds[0] = z;
            zBounds[1] = z;
        }
        std::vector<OffsetObj<T*> > neighbors;
        for (int i=x-1; i<=x+1; i++) {
            for (int j=y-1; j<=y+1; j++) {
                for (int k=zBounds[0]; k<=zBounds[1]; k++) {
                    if (not (i==x and j==y and k==z)) {
                        //Vector v(i, j, k);
                        int boxCoords[3];
                        boxCoords[0] = i;
                        boxCoords[1] = j;
                        boxCoords[2] = k;
                        int didLoop[3] = {0, 0, 0};
                        Vector offset;
                        T *neigh = &(*this)(boxCoords, didLoop);
                        bool append = true;
                        for (int i=0; i<3; i++) {
                            append = append and (not didLoop[i] or (didLoop[i] and loops[i]));
                        }
                        if (append) {
                            for (int i=0; i<3; i++) {
                                offset[i] = didLoop[i];
                            }
                            neighbors.push_back(OffsetObj<T*>(neigh, offset));
                        }

                    }
                }
            }
        }
        return neighbors;

    }

    /*! \brief Compile vector of neighbors to Grid coordinate
     *
     * \param coords Pointer to 3-element array storing Grid coordinates
     * \param loops Pointer to array storing if box is periodic
     * \param trace Trace of the box
     * \return vector storing neighbors with their unwrapped coordinates
     *
     * This function does the same thing as getNeighbors() but stores the
     * unwrapped coordinate instead of the mirror box as the offset.
     *
     * \todo Code duplication is a very bad thing. Rewrite this function and
     *       getNeighbors so that they do not overlap as much.
     */
    std::vector<OffsetObj<T> > getNeighborVals(int coords[3], bool loops[3], Vector trace)
    {
        const int x = coords[0];
        const int y = coords[1];
        const int z = coords[2];
        std::vector<OffsetObj<T> > neighbors;
        for (int i=x-1; i<=x+1; i++) {
            for (int j=y-1; j<=y+1; j++) {
                for (int k=z-1; k<=z+1; k++) {
                    if (not (i==x and j==y and k==z)) {
                        //Vector v(i, j, k);
                        int boxCoords[3];
                        boxCoords[0] = i;
                        boxCoords[1] = j;
                        boxCoords[2] = k;
                        int didLoop[3] = {0, 0, 0};
                        double offsets[3] = {0, 0, 0};
                        T neigh = (*this)(boxCoords, didLoop);
                        bool append = true;
                        for (int i=0; i<3; i++) {
                            append = append and (not didLoop[i] or (didLoop[i] and loops[i]));
                        }
                        if (append) {
                            for (int i=0; i<3; i++) {
                                offsets[i] = didLoop[i] * trace[i];
                            }
                            Vector offset(offsets);
                            neighbors.push_back(OffsetObj<T>(neigh, offset));
                        }

                    }
                }
            }
        }
        return neighbors;

    }

};

#endif

