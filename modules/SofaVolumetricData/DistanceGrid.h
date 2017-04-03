/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_VOLUMETRICDATA_DISTANCEGRID_H
#define SOFA_VOLUMETRICDATA_DISTANCEGRID_H
#include "config.h"

#include <sofa/defaulttype/Vec3Types.h>


///// Forward declaration
namespace sofa
{
    namespace helper
    {
        namespace io
        {
            class Mesh ;
        }
    }
}


////// DistanceGrid declaration
namespace sofa
{
//todo(dmarchal) I see no reason why this is into component as DistanceGrid obvisouly isn't one
// can someone suggest a refactoring to have things in the right location.
namespace component
{
namespace container
{

/// Private namespace to avoid leaking data types into the files.
namespace _distancegrid_
{
using sofa::helper::io::Mesh;
using sofa::defaulttype::Vector3 ;
using sofa::defaulttype::ExtVector ;
using sofa::defaulttype::ExtVectorAllocator ;
typedef Vector3 Coord;

class SOFA_VOLUMETRIC_DATA_API DistanceGrid
{
public:
    static SReal maxDist() { return (SReal)1e10; }
    typedef Vector3 Coord;
    typedef ExtVector<SReal> VecSReal;
    typedef ExtVector<Coord> VecCoord;

    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax);
    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax,
                 ExtVectorAllocator<SReal>* alloc);

protected:
    ~DistanceGrid();

public:
    /// Load a distance grid
    static DistanceGrid* load(const std::string& filename,
                              double scale=1.0, double sampling=0.0,
                              int nx=64, int ny=64, int nz=64,
                              Coord pmin = Coord(), Coord pmax = Coord());

    static DistanceGrid* loadVTKFile(const std::string& filename,
                                     double scale=1.0, double sampling=0.0);

    /// Load or reuse a distance grid
    static DistanceGrid* loadShared(const std::string& filename,
                                    double scale=1.0, double sampling=0.0,
                                    int nx=64, int ny=64, int nz=64,
                                    Coord pmin = Coord(), Coord pmax = Coord());

    /// Add one reference to this grid. Note that loadShared already does this.
    DistanceGrid* addRef();

    /// Release one reference, deleting this grid if this is the last
    bool release();

    /// Save current grid
    bool save(const std::string& filename);

    /// Compute distance field from given mesh
    void calcDistance(Mesh* mesh, double scale=1.0);

    /// Compute distance field for a cube of the given half-size.
    /// Also create a mesh of points using np points per axis
    void calcCubeDistance(SReal dim=1, int np=5);

    /// Sample the surface with points approximately separated by the given sampling distance
    /// (expressed in voxels if the value is negative)
    void sampleSurface(double sampling=-1.0);

    /// Update bbox
    void computeBBox();

    inline int getNx() const { return nx; }
    inline int getNy() const { return ny; }
    inline int getNz() const { return nz; }
    inline const Coord& getCellWidth() const { return cellWidth; }

    inline int size() const { return nxnynz; }

    inline const Coord& getBBMin() const { return bbmin; }
    inline const Coord& getBBMax() const { return bbmax; }
    inline void setBBMin(const Coord& val) { bbmin = val; }
    inline void setBBMax(const Coord& val) { bbmax = val; }
    inline Coord getBBCorner(int i) const {
        return Coord((i&1)?bbmax[0]:bbmin[0],(i&2)?bbmax[1]:bbmin[1],(i&4)?bbmax[2]:bbmin[2]);
    }

    inline bool inBBox(const Coord& p, SReal margin=0.0f) const
    {
        for (int c=0; c<3; ++c)
            if (p[c] < bbmin[c]-margin || p[c] > bbmax[c]+margin) return false;
        return true;
    }

    inline const Coord& getPMin() const { return pmin; }
    inline const Coord& getPMax() const { return pmax; }
    inline Coord getCorner(int i) const {
        return Coord((i&1)?pmax[0]:pmin[0],(i&2)?pmax[1]:pmin[1],(i&4)?pmax[2]:pmin[2]);
    }

    inline bool isCube() const { return cubeDim != 0; }
    inline SReal getCubeDim() const { return cubeDim; }

    bool inGrid(const Coord& p) const
    {
        Coord epsilon = cellWidth*0.1;
        for (int c=0; c<3; ++c)
            if (p[c] < pmin[c]+epsilon[c] || p[c] > pmax[c]-epsilon[c]) return false;
        return true;
    }

    Coord clamp(Coord p) const
    {
        for (int c=0; c<3; ++c)
            if (p[c] < pmin[c]) p[c] = pmin[c];
            else if (p[c] > pmax[c]) p[c] = pmax[c];
        return p;
    }

    int ix(const Coord& p) const
    {
        return helper::rfloor((p[0]-pmin[0])*invCellWidth[0]);
    }

    int iy(const Coord& p) const
    {
        return helper::rfloor((p[1]-pmin[1])*invCellWidth[1]);
    }

    int iz(const Coord& p) const
    {
        return helper::rfloor((p[2]-pmin[2])*invCellWidth[2]);
    }

    int index(const Coord& p, Coord& coefs) const ;

    int index(const Coord& p) const
    {
        Coord coefs;
        return index(p, coefs);
    }

    int index(int x, int y, int z)
    {
        return x+nx*(y+ny*(z));
    }

    Coord coord(int x, int y, int z)
    {
        return pmin+Coord(x*cellWidth[0], y*cellWidth[1], z*cellWidth[2]);
    }

    SReal operator[](int index) const { return dists[index]; }
    SReal& operator[](int index) { return dists[index]; }

    static SReal interp(SReal coef, SReal a, SReal b)
    {
        return a+coef*(b-a);
    }

    SReal interp(int index, const Coord& coefs) const ;
    SReal interp(const Coord& p) const ;
    Coord grad(int index, const Coord& coefs) const ;
    Coord grad(const Coord& p) const ;
    SReal eval(const Coord& x) const ;
    SReal quickeval(const Coord& x) const ;
    SReal eval2(const Coord& x) const ;
    SReal quickeval2(const Coord& x) const ;

    template<class T>
    T tgrad(const T& p) const
    {
        Coord cp;
        for (unsigned int i=0; i < cp.size() && i < p.size(); ++i) cp[i] = (SReal) p[i];
        Coord cr = grad(cp);
        T r;
        for (unsigned int i=0; i < cr.size() && i < p.size(); ++i) r[i] = (typename T::value_type) cr[i];
        return r;
    }

    template<class T>
    typename T::value_type teval(const T& x) const
    {
        Coord cx;
        for (unsigned int i=0; i < cx.size() && i < x.size(); ++i) cx[i] = (SReal) x[i];
        return (typename T::value_type) eval(cx);
    }

    VecCoord meshPts;

protected:
    int nbRef;
    VecSReal dists;
    const int nx,ny,nz, nxny, nxnynz;
    const Coord pmin, pmax;
    const Coord cellWidth, invCellWidth;
    Coord bbmin, bbmax; ///< bounding box of the object, smaller than the grid

    SReal cubeDim; ///< Cube dimension (!=0 if this is actually a cube

    /// Fast Marching Method Update
    enum Status { FMM_FRONT0 = 0, FMM_FAR = -1, FMM_KNOWN_OUT = -2, FMM_KNOWN_IN = -3 };
    helper::vector<int> fmm_status;
    helper::vector<int> fmm_heap;
    int fmm_heap_size;

    int fmm_pop();
    void fmm_push(int index);
    void fmm_swap(int entry1, int entry2);

    /// Grid shared resources
    struct DistanceGridParams
    {
        std::string filename;
        double scale;
        double sampling;
        int nx,ny,nz;
        Coord pmin,pmax;
        bool operator==(const DistanceGridParams& v) const ;
        bool operator<(const DistanceGridParams& v) const ;
        bool operator>(const DistanceGridParams& v) const ;
    };

    static std::map<DistanceGridParams, DistanceGrid*>& getShared();
};

} // namespace _distancegrid

using _distancegrid_::DistanceGrid ;

} // namespace container

} // namespace component

} // namespace sofa

#endif
