/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_DISTANCEGRIDCOLLISIONMODEL_H
#define SOFA_COMPONENT_COLLISION_DISTANCEGRIDCOLLISIONMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/topology/SparseGridTopology.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::helper;

class SOFA_COMPONENT_COLLISION_API DistanceGrid
{
public:
    static SReal maxDist() { return (SReal)1e10; }
    typedef Vector3 Coord;
    typedef defaulttype::ExtVector<SReal> VecSReal;
    typedef defaulttype::ExtVector<Coord> VecCoord;

    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax);

    DistanceGrid(int nx, int ny, int nz, Coord pmin, Coord pmax, defaulttype::ExtVectorAllocator<SReal>* alloc);

protected:
    ~DistanceGrid();

public:
    /// Load a distance grid
    static DistanceGrid* load(const std::string& filename, double scale=1.0, int nx=64, int ny=64, int nz=64, Coord pmin = Coord(), Coord pmax = Coord());
    static DistanceGrid* loadVTKFile(const std::string& filename, double scale=1.0);

    /// Load or reuse a distance grid
    static DistanceGrid* loadShared(const std::string& filename, double scale=1.0, int nx=64, int ny=64, int nz=64, Coord pmin = Coord(), Coord pmax = Coord());

    /// Add one reference to this grid. Note that loadShared already does this.
    DistanceGrid* addRef();

    /// Release one reference, deleting this grid if this is the last
    bool release();

    /// Save current grid
    bool save(const std::string& filename);

    /// Compute distance field from given mesh
    void calcDistance(sofa::helper::io::Mesh* mesh, double scale=1.0);

    /// Compute distance field for a cube of the given half-size.
    /// Also create a mesh of points using np points per axis
    void calcCubeDistance(SReal dim=1, int np=5);

    /// Update bbox
    void computeBBox();

    int getNx() const { return nx; }
    int getNy() const { return ny; }
    int getNz() const { return nz; }
    const Coord& getCellWidth() const { return cellWidth; }

    int size() const { return nxnynz; }

    const Coord& getBBMin() const { return bbmin; }
    const Coord& getBBMax() const { return bbmax; }
    void setBBMin(const Coord& val) { bbmin = val; }
    void setBBMax(const Coord& val) { bbmax = val; }
    Coord getBBCorner(int i) const { return Coord((i&1)?bbmax[0]:bbmin[0],(i&2)?bbmax[1]:bbmin[1],(i&4)?bbmax[2]:bbmin[2]); }
    bool inBBox(const Coord& p, SReal margin=0.0f) const
    {
        for (int c=0; c<3; ++c)
            if (p[c] < bbmin[c]-margin || p[c] > bbmax[c]+margin) return false;
        return true;
    }

    const Coord& getPMin() const { return pmin; }
    const Coord& getPMax() const { return pmax; }
    Coord getCorner(int i) const { return Coord((i&1)?pmax[0]:pmin[0],(i&2)?pmax[1]:pmin[1],(i&4)?pmax[2]:pmin[2]); }

    bool isCube() const { return cubeDim != 0; }
    SReal getCubeDim() const { return cubeDim; }

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
        return rfloor((p[0]-pmin[0])*invCellWidth[0]);
    }

    int iy(const Coord& p) const
    {
        return rfloor((p[1]-pmin[1])*invCellWidth[1]);
    }

    int iz(const Coord& p) const
    {
        return rfloor((p[2]-pmin[2])*invCellWidth[2]);
    }

    int index(const Coord& p, Coord& coefs) const
    {
        coefs[0] = (p[0]-pmin[0])*invCellWidth[0];
        coefs[1] = (p[1]-pmin[1])*invCellWidth[1];
        coefs[2] = (p[2]-pmin[2])*invCellWidth[2];
        int x = rfloor(coefs[0]);
        if (x<0) x=0; else if (x>=nx-1) x=nx-2;
        coefs[0] -= x;
        int y = rfloor(coefs[1]);
        if (y<0) y=0; else if (y>=ny-1) y=ny-2;
        coefs[1] -= y;
        int z = rfloor(coefs[2]);
        if (z<0) z=0; else if (z>=nz-1) z=nz-2;
        coefs[2] -= z;
        return x+nx*(y+ny*(z));
    }

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

    SReal interp(int index, const Coord& coefs) const
    {
        return interp(coefs[2],interp(coefs[1],interp(coefs[0],dists[index          ],dists[index+1        ]),
                interp(coefs[0],dists[index  +nx     ],dists[index+1+nx     ])),
                interp(coefs[1],interp(coefs[0],dists[index     +nxny],dists[index+1   +nxny]),
                        interp(coefs[0],dists[index  +nx+nxny],dists[index+1+nx+nxny])));
    }

    SReal interp(const Coord& p) const
    {
        Coord coefs;
        int i = index(p, coefs);
        return interp(i, coefs);
    }

    Coord grad(int index, const Coord& coefs) const
    {
        // val = dist[0][0][0] * (1-x) * (1-y) * (1-z)
        //     + dist[1][0][0] * (  x) * (1-y) * (1-z)
        //     + dist[0][1][0] * (1-x) * (  y) * (1-z)
        //     + dist[1][1][0] * (  x) * (  y) * (1-z)
        //     + dist[0][0][1] * (1-x) * (1-y) * (  z)
        //     + dist[1][0][1] * (  x) * (1-y) * (  z)
        //     + dist[0][1][1] * (1-x) * (  y) * (  z)
        //     + dist[1][1][1] * (  x) * (  y) * (  z)
        // dval / dx = (dist[1][0][0]-dist[0][0][0]) * (1-y) * (1-z)
        //           + (dist[1][1][0]-dist[0][1][0]) * (  y) * (1-z)
        //           + (dist[1][0][1]-dist[0][0][1]) * (1-y) * (  z)
        //           + (dist[1][1][1]-dist[0][1][1]) * (  y) * (  z)
        const SReal dist000 = dists[index          ];
        const SReal dist100 = dists[index+1        ];
        const SReal dist010 = dists[index  +nx     ];
        const SReal dist110 = dists[index+1+nx     ];
        const SReal dist001 = dists[index     +nxny];
        const SReal dist101 = dists[index+1   +nxny];
        const SReal dist011 = dists[index  +nx+nxny];
        const SReal dist111 = dists[index+1+nx+nxny];
        return Coord(
                interp(coefs[2],interp(coefs[1],dist100-dist000,dist110-dist010),interp(coefs[1],dist101-dist001,dist111-dist011)), //*invCellWidth[0],
                interp(coefs[2],interp(coefs[0],dist010-dist000,dist110-dist100),interp(coefs[0],dist011-dist001,dist111-dist101)), //*invCellWidth[1],
                interp(coefs[1],interp(coefs[0],dist001-dist000,dist101-dist100),interp(coefs[0],dist011-dist010,dist111-dist110))); //*invCellWidth[2]);
    }

    Coord grad(const Coord& p) const
    {
        Coord coefs;
        int i = index(p, coefs);
        return grad(i, coefs);
    }

    SReal eval(const Coord& x) const
    {
        SReal d;
        if (inGrid(x))
        {
            d = interp(x);
        }
        else
        {
            Coord xclamp = clamp(x);
            d = interp(xclamp);
            d = rsqrt((x-xclamp).norm2() + d*d); // we underestimate the distance
        }
        return d;
    }

    SReal quickeval(const Coord& x) const
    {
        SReal d;
        if (inGrid(x))
        {
            d = dists[index(x)] - cellWidth[0]; // we underestimate the distance
        }
        else
        {
            Coord xclamp = clamp(x);
            d = dists[index(xclamp)] - cellWidth[0]; // we underestimate the distance
            d = rsqrt((x-xclamp).norm2() + d*d);
        }
        return d;
    }

    SReal eval2(const Coord& x) const
    {
        SReal d2;
        if (inGrid(x))
        {
            SReal d = interp(x);
            d2 = d*d;
        }
        else
        {
            Coord xclamp = clamp(x);
            SReal d = interp(xclamp);
            d2 = ((x-xclamp).norm2() + d*d); // we underestimate the distance
        }
        return d2;
    }

    SReal quickeval2(const Coord& x) const
    {
        SReal d2;
        if (inGrid(x))
        {
            SReal d = dists[index(x)] - cellWidth[0]; // we underestimate the distance
            d2 = d*d;
        }
        else
        {
            Coord xclamp = clamp(x);
            SReal d = dists[index(xclamp)] - cellWidth[0]; // we underestimate the distance
            d2 = ((x-xclamp).norm2() + d*d);
        }
        return d2;
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

    // Fast Marching Method Update
    enum Status { FMM_FRONT0 = 0, FMM_FAR = -1, FMM_KNOWN_OUT = -2, FMM_KNOWN_IN = -3 };
    helper::vector<int> fmm_status;
    helper::vector<int> fmm_heap;
    int fmm_heap_size;

    int fmm_pop();
    void fmm_push(int index);
    void fmm_swap(int entry1, int entry2);

    // Grid shared resources

    struct DistanceGridParams
    {
        std::string filename;
        double scale;
        int nx,ny,nz;
        Coord pmin,pmax;
        bool operator==(const DistanceGridParams& v) const
        {
            if (!(filename == v.filename)) return false;
            if (!(scale    == v.scale   )) return false;
            if (!(nx       == v.nx      )) return false;
            if (!(ny       == v.ny      )) return false;
            if (!(nz       == v.nz      )) return false;
            if (!(pmin[0]  == v.pmin[0] )) return false;
            if (!(pmin[1]  == v.pmin[1] )) return false;
            if (!(pmin[2]  == v.pmin[2] )) return false;
            if (!(pmax[0]  == v.pmax[0] )) return false;
            if (!(pmax[1]  == v.pmax[1] )) return false;
            if (!(pmax[2]  == v.pmax[2] )) return false;
            return true;
        }
        bool operator<(const DistanceGridParams& v) const
        {
            if (filename < v.filename) return true;
            if (filename > v.filename) return false;
            if (scale    < v.scale   ) return true;
            if (scale    > v.scale   ) return false;
            if (nx       < v.nx      ) return false;
            if (nx       > v.nx      ) return true;
            if (ny       < v.ny      ) return false;
            if (ny       > v.ny      ) return true;
            if (nz       < v.nz      ) return false;
            if (nz       > v.nz      ) return true;
            if (pmin[0]  < v.pmin[0] ) return false;
            if (pmin[0]  > v.pmin[0] ) return true;
            if (pmin[1]  < v.pmin[1] ) return false;
            if (pmin[1]  > v.pmin[1] ) return true;
            if (pmin[2]  < v.pmin[2] ) return false;
            if (pmin[2]  > v.pmin[2] ) return true;
            if (pmax[0]  < v.pmax[0] ) return false;
            if (pmax[0]  > v.pmax[0] ) return true;
            if (pmax[1]  < v.pmax[1] ) return false;
            if (pmax[1]  > v.pmax[1] ) return true;
            if (pmax[2]  < v.pmax[2] ) return false;
            if (pmax[2]  > v.pmax[2] ) return true;
            return false;
        }
        bool operator>(const DistanceGridParams& v) const
        {
            if (filename > v.filename) return true;
            if (filename < v.filename) return false;
            if (scale    > v.scale   ) return true;
            if (scale    < v.scale   ) return false;
            if (nx       > v.nx      ) return false;
            if (nx       < v.nx      ) return true;
            if (ny       > v.ny      ) return false;
            if (ny       < v.ny      ) return true;
            if (nz       > v.nz      ) return false;
            if (nz       < v.nz      ) return true;
            if (pmin[0]  > v.pmin[0] ) return false;
            if (pmin[0]  < v.pmin[0] ) return true;
            if (pmin[1]  > v.pmin[1] ) return false;
            if (pmin[1]  < v.pmin[1] ) return true;
            if (pmin[2]  > v.pmin[2] ) return false;
            if (pmin[2]  < v.pmin[2] ) return true;
            if (pmax[0]  > v.pmax[0] ) return false;
            if (pmax[0]  < v.pmax[0] ) return true;
            if (pmax[1]  > v.pmax[1] ) return false;
            if (pmax[1]  < v.pmax[1] ) return true;
            if (pmax[2]  > v.pmax[2] ) return false;
            if (pmax[2]  < v.pmax[2] ) return true;
            return false;
        }
    };
    static std::map<DistanceGridParams, DistanceGrid*>& getShared();

};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class SOFA_COMPONENT_COLLISION_API RigidDistanceGridCollisionModel;

class RigidDistanceGridCollisionElement : public core::TCollisionElementIterator<RigidDistanceGridCollisionModel>
{
public:

    RigidDistanceGridCollisionElement(RigidDistanceGridCollisionModel* model, int index);

    explicit RigidDistanceGridCollisionElement(core::CollisionElementIterator& i);

    DistanceGrid* getGrid();

    bool isTransformed();
    const Matrix3& getRotation();
    const Vector3& getTranslation();

    void setGrid(DistanceGrid* surf);

    /// @name Previous state data
    /// Used to estimate velocity in case the distance grid itself is dynamic
    /// @{
    DistanceGrid* getPrevGrid();
    const Matrix3& getPrevRotation();
    const Vector3& getPrevTranslation();
    double getPrevDt();
    /// @}

    /// Set new grid and transform, keeping the old state to estimate velocity
    void setNewState(double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation);
};

class SOFA_COMPONENT_COLLISION_API RigidDistanceGridCollisionModel : public core::CollisionModel
{
protected:

    class ElementData
    {
    public:
        Matrix3 rotation;
        Vector3 translation;
        DistanceGrid* grid;

        /// @name Previous state data
        /// Used to estimate velocity in case the distance grid itself is dynamic
        /// @{
        DistanceGrid* prevGrid; ///< Previous grid
        Matrix3 prevRotation; ///< Previous rotation
        Vector3 prevTranslation; ///< Previous translation
        double prevDt; ///< Time difference between previous and current state
        /// @}

        bool isTransformed; ///< True if translation/rotation was set
        ElementData() : grid(NULL), prevGrid(NULL), prevDt(0.0), isTransformed(false) { rotation.identity(); prevRotation.identity(); }
    };

    sofa::helper::vector<ElementData> elems;
    bool modified;

    // Input data parameters
    sofa::core::objectmodel::DataFileName fileRigidDistanceGrid;
    Data< double > scale;
    Data< helper::fixed_array<DistanceGrid::Coord,2> > box;
    Data< int > nx;
    Data< int > ny;
    Data< int > nz;
    sofa::core::objectmodel::DataFileName dumpfilename;

    core::componentmodel::behavior::MechanicalState<RigidTypes>* rigid;

    void updateGrid();
public:
    typedef Rigid3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef RigidDistanceGridCollisionElement Element;

    Data< bool > usePoints;

    RigidDistanceGridCollisionModel();

    ~RigidDistanceGridCollisionModel();

    core::componentmodel::behavior::MechanicalState<InDataTypes>* getRigidModel() { return rigid; }
    core::componentmodel::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return rigid; }

    void init();

    DistanceGrid* getGrid(int index=0)
    {
        return elems[index].grid;
    }
    bool isTransformed(int index=0)
    {
        return elems[index].isTransformed;
    }
    const Matrix3& getRotation(int index=0)
    {
        return elems[index].rotation;
    }
    const Vector3& getTranslation(int index=0)
    {
        return elems[index].translation;
    }

    void setGrid(DistanceGrid* surf, int index=0);

    DistanceGrid* getPrevGrid(int index=0)
    {
        return elems[index].prevGrid;
    }
    const Matrix3& getPrevRotation(int index=0)
    {
        return elems[index].prevRotation;
    }
    const Vector3& getPrevTranslation(int index=0)
    {
        return elems[index].prevTranslation;
    }
    double getPrevDt(int index=0)
    {
        return elems[index].prevDt;
    }

    /// Set new grid and transform, keeping the old state to estimate velocity
    void setNewState(int index, double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation);

    /// @}

    /// Set new grid and transform, keeping the old state to estimate velocity
    void setNewState(double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation);

    // -- CollisionModel interface

    void resize(int size);

    /// Create or update the bounding volume hierarchy.
    void computeBoundingTree(int maxDepth=0);

    void draw(int index);

    void draw();
};

inline RigidDistanceGridCollisionElement::RigidDistanceGridCollisionElement(RigidDistanceGridCollisionModel* model, int index)
    : core::TCollisionElementIterator<RigidDistanceGridCollisionModel>(model, index)
{}

inline RigidDistanceGridCollisionElement::RigidDistanceGridCollisionElement(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<RigidDistanceGridCollisionModel>(static_cast<RigidDistanceGridCollisionModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline DistanceGrid* RigidDistanceGridCollisionElement::getGrid() { return model->getGrid(index); }
inline void RigidDistanceGridCollisionElement::setGrid(DistanceGrid* surf) { return model->setGrid(surf, index); }

inline bool RigidDistanceGridCollisionElement::isTransformed() { return model->isTransformed(index); }
inline const Matrix3& RigidDistanceGridCollisionElement::getRotation() { return model->getRotation(index); }
inline const Vector3& RigidDistanceGridCollisionElement::getTranslation() { return model->getTranslation(index); }

inline DistanceGrid* RigidDistanceGridCollisionElement::getPrevGrid() { return model->getPrevGrid(index); }
inline const Matrix3& RigidDistanceGridCollisionElement::getPrevRotation() { return model->getPrevRotation(index); }
inline const Vector3& RigidDistanceGridCollisionElement::getPrevTranslation() { return model->getPrevTranslation(index); }
inline double RigidDistanceGridCollisionElement::getPrevDt() { return model->getPrevDt(index); }

inline void RigidDistanceGridCollisionElement::setNewState(double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation)
{
    return model->setNewState(dt, grid, rotation, translation);
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

class FFDDistanceGridCollisionModel;

class FFDDistanceGridCollisionElement : public core::TCollisionElementIterator<FFDDistanceGridCollisionModel>
{
public:

    FFDDistanceGridCollisionElement(FFDDistanceGridCollisionModel* model, int index);

    explicit FFDDistanceGridCollisionElement(core::CollisionElementIterator& i);

    DistanceGrid* getGrid();

    void setGrid(DistanceGrid* surf);
};

class SOFA_COMPONENT_COLLISION_API FFDDistanceGridCollisionModel : public core::CollisionModel
{
public:
    typedef SReal GSReal;
    typedef DistanceGrid::Coord GCoord;
    class DeformedCube
    {
    public:
        DistanceGrid* grid;
        DeformedCube() : grid(NULL) {}
        int elem; ///< Index of the corresponding element in the topology
        struct Point
        {
            GCoord bary; ///< Barycentric coordinates
            int index; ///< Index of corresponding point in in DistanceGrid
        };
        vector<Point> points; ///< barycentric coordinates of included points
        GCoord initP0,initDP,invDP; ///< Initial corners position
        GCoord corners[8]; ///< Current corners position
        enum {C000 = 0+0+0,
                C100 = 1+0+0,
                C010 = 0+2+0,
                C110 = 1+2+0,
                C001 = 0+0+4,
                C101 = 1+0+4,
                C011 = 0+2+4,
                C111 = 1+2+4
             };
        typedef Vec<4,GSReal> Plane; ///< plane equation as defined by Plane.(x y z 1) = 0
        Plane faces[6]; ///< planes corresponding to the six faces (FX0,FX1,FY0,FY1,FZ0,FZ1)
        enum {FX0 = 0+0,
                FX1 = 0+1,
                FY0 = 2+0,
                FY1 = 2+1,
                FZ0 = 4+0,
                FZ1 = 4+1
             };
        /// @name Precomputed deformation factors
        /// We have :
        ///   deform(b) = C000(1-b[0])(1-b[1])(1-b[2]) + C100(b[0])(1-b[1])(1-b[2]) + C010(1-b[0])(b[1])(1-b[2]) + C110(b[0])(b[1])(1-b[2])
        ///             + C001(1-b[0])(1-b[1])(  b[2]) + C101(b[0])(1-b[1])(  b[2]) + C011(1-b[0])(b[1])(  b[2]) + C111(b[0])(b[1])(  b[2])
        ///             = C000 + Dx b[0] + Dy b[1] + Dz b[2] + Dxy b[0]b[1] + Dxz b[0]b[2] + dyz b[1]b[2] + dxyz b[0]b[1]b[2]
        /// @{
        GCoord Dx;   ///< Dx = -C000+C100
        GCoord Dy;   ///< Dy = -C000+C010
        GCoord Dz;   ///< Dx = -C000+C001
        GCoord Dxy;  ///< Dxy = C000-C100-C010+C110 = C110-C010-Dx
        GCoord Dxz;  ///< Dxz = C000-C100-C001+C101 = C101-C001-Dx
        GCoord Dyz;  ///< Dyz = C000-C010-C001+C011 = C011-C001-Dy
        GCoord Dxyz; ///< Dxyz = - C000 + C100 + C010 - C110 + C001 - C101 - C011 + C111 = C001 - C101 - C011 + C111 - Dxy
        /// @}
        /// Update the deformation precomputed values
        void updateDeform();

        GCoord center; ///< current center;
        GSReal radius; ///< radius of enclosing sphere
        vector<GCoord> deformedPoints; ///< deformed points
        bool pointsUpdated; ///< true the deformedPoints vector has been updated with the latest positions
        void updatePoints(); ///< Update the deformedPoints position if not done yet (i.e. if pointsUpdated==false)
        bool facesUpdated; ///< true the faces plane vector has been updated with the latest positions
        void updateFaces(); ///< Update the face planes if not done yet (i.e. if facesUpdated==false)
        /// Compute the barycentric coordinates of a point from its initial position
        DistanceGrid::Coord baryCoords(const GCoord& c) const
        {
            return GCoord( (c[0]-initP0[0])*invDP[0],
                    (c[1]-initP0[1])*invDP[1],
                    (c[2]-initP0[2])*invDP[2]);
        }
        /// Compute the initial position of a point from its barycentric coordinates
        GCoord initpos(const GCoord& b) const
        {
            return GCoord( initP0[0]+initDP[0]*b[0],
                    initP0[1]+initDP[1]*b[1],
                    initP0[2]+initDP[2]*b[2]);
        }
        /// Compute the deformed position of a point from its barycentric coordinates
        GCoord deform(const GCoord& b) const
        {
            return corners[C000] + Dx*b[0] + (Dy + Dxy*b[0])*b[1] + (Dz + Dxz*b[0] + (Dyz + Dxyz*b[0])*b[1])*b[2];
        }

        static GSReal interp(GSReal coef, GSReal a, GSReal b)
        {
            return a+coef*(b-a);
        }

        /// deform a direction relative to a point in barycentric coordinates
        GCoord deformDir(const GCoord& b, const GCoord& dir) const
        {
            GCoord r;
            // dp/dx = Dx + Dxy*y + Dxz*z + Dxyz*y*z
            r  = (Dx + Dxy*b[1] + (Dxz + Dxyz*b[1])*b[2])*dir[0];
            // dp/dy = Dy + Dxy*x + Dyz*z + Dxyz*x*z
            r += (Dy + Dxy*b[0] + (Dyz + Dxyz*b[0])*b[2])*dir[1];
            // dp/dz = Dz + Dxz*x + Dyz*y + Dxyz*x*y
            r += (Dz + Dxz*b[0] + (Dyz + Dxyz*b[0])*b[1])*dir[2];
            return r;
        }

        /// Get the local jacobian matrix of the deformation
        Mat<3,3,double> Jdeform(const GCoord& b) const
        {
            Mat<3,3,double> J;
            for (int i=0; i<3; i++)
            {
                // dp/dx = Dx + Dxy*y + Dxz*z + Dxyz*y*z
                J[i][0] = (Dx[i] + Dxy[i]*b[1] + (Dxz[i] + Dxyz[i]*b[1])*b[2]);
                // dp/dy = Dy + Dxy*x + Dyz*z + Dxyz*x*z
                J[i][1] = (Dy[i] + Dxy[i]*b[0] + (Dyz[i] + Dxyz[i]*b[0])*b[2]);
                // dp/dz = Dz + Dxz*x + Dyz*y + Dxyz*x*y
                J[i][2] = (Dz[i] + Dxz[i]*b[0] + (Dyz[i] + Dxyz[i]*b[0])*b[1]);
            }
            return J;
        }

        /// Compute an initial estimate to the barycentric coordinate of a point given its deformed position
        GCoord undeform0(const GCoord& p) const
        {
            GCoord b;
            for (int i=0; i<3; i++)
            {
                GSReal b0 = faces[2*i+0]*Plane(p,1);
                GSReal b1 = faces[2*i+1]*Plane(p,1);
                b[i] = b0 / (b0 + b1);
            }
            return b;
        }
        /// Undeform a direction relative to a point in barycentric coordinates
        GCoord undeformDir(const GCoord& b, const GCoord& dir) const
        {
            // we want to find b2 so that deform(b2)-deform(b) = dir
            // we can use Newton's method using the jacobian of the deformation.
            Mat<3,3,double> m = Jdeform(b);
            Mat<3,3,double> minv;
            minv.invert(m);
            return minv*dir;
        }

        /// Compute a plane equation given 4 corners
        Plane computePlane(int c00, int c10, int c01, int c11);
    };

protected:

    sofa::helper::vector<DeformedCube> elems;

    // Input data parameters
    sofa::core::objectmodel::DataFileName  fileFFDDistanceGrid;
    Data< double > scale;
    Data< helper::fixed_array<DistanceGrid::Coord,2> > box;
    Data< int > nx;
    Data< int > ny;
    Data< int > nz;
    sofa::core::objectmodel::DataFileName dumpfilename;

    core::componentmodel::behavior::MechanicalState<Vec3Types>* ffd;
    core::componentmodel::topology::BaseMeshTopology* ffdMesh;
    //topology::RegularGridTopology* ffdGrid;
    topology::RegularGridTopology* ffdRGrid;
    topology::SparseGridTopology* ffdSGrid;

    void updateGrid();
public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef topology::RegularGridTopology Topology;
    typedef FFDDistanceGridCollisionElement Element;

    Data< bool > usePoints;

    FFDDistanceGridCollisionModel();

    ~FFDDistanceGridCollisionModel();

    core::componentmodel::behavior::MechanicalState<DataTypes>* getDeformModel() { return ffd; }
    core::componentmodel::topology::BaseMeshTopology* getDeformGrid() { return ffdMesh; }

    // alias used by ContactMapper

    core::componentmodel::behavior::MechanicalState<DataTypes>* getMechanicalState() { return ffd; }
    core::componentmodel::topology::BaseMeshTopology* getMeshTopology() { return ffdMesh; }

    void init();

    DistanceGrid* getGrid(int index=0)
    {
        return elems[index].grid;
    }

    DeformedCube& getDeformCube(int index=0)
    {
        return elems[index];
    }

    void setGrid(DistanceGrid* surf, int index=0);

    // -- CollisionModel interface

    void resize(int size);

    /// Create or update the bounding volume hierarchy.
    void computeBoundingTree(int maxDepth=0);

    void draw(int index);

    void draw();
};

inline FFDDistanceGridCollisionElement::FFDDistanceGridCollisionElement(FFDDistanceGridCollisionModel* model, int index)
    : core::TCollisionElementIterator<FFDDistanceGridCollisionModel>(model, index)
{}

inline FFDDistanceGridCollisionElement::FFDDistanceGridCollisionElement(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<FFDDistanceGridCollisionModel>(static_cast<FFDDistanceGridCollisionModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline DistanceGrid* FFDDistanceGridCollisionElement::getGrid() { return model->getGrid(index); }
inline void FFDDistanceGridCollisionElement::setGrid(DistanceGrid* surf) { return model->setGrid(surf, index); }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
