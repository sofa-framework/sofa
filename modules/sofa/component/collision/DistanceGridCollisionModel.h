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
#include <sofa/component/container/DistanceGrid.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
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

typedef container::DistanceGrid DistanceGrid;

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
    bool isFlipped();

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
public:
    SOFA_CLASS(RigidDistanceGridCollisionModel,sofa::core::CollisionModel);

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
    Data< Vector3 > translation;
    Data< double > sampling;
    Data< helper::fixed_array<DistanceGrid::Coord,2> > box;
    Data< int > nx;
    Data< int > ny;
    Data< int > nz;
    sofa::core::objectmodel::DataFileName dumpfilename;

    core::behavior::MechanicalState<RigidTypes>* rigid;

    void updateGrid();
public:
    typedef Rigid3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef RigidDistanceGridCollisionElement Element;

    Data< bool > usePoints;
    Data< bool > flipNormals;

    RigidDistanceGridCollisionModel();

    ~RigidDistanceGridCollisionModel();

    core::behavior::MechanicalState<InDataTypes>* getRigidModel() { return rigid; }
    core::behavior::MechanicalState<InDataTypes>* getMechanicalState() { return rigid; }

    void init();

    DistanceGrid* getGrid(int index=0)
    {
        return elems[index].grid;
    }
    bool isTransformed(int index=0) const
    {
        return elems[index].isTransformed;
    }
    const Matrix3& getRotation(int index=0) const
    {
        return elems[index].rotation;
    }
    const Vector3& getTranslation(int index=0) const
    {
        return elems[index].translation;
    }

    const Vector3& getInitTranslation() const
    {
        return translation.getValue();
    }

    bool isFlipped() const
    {
        return flipNormals.getValue();
    }

    void setGrid(DistanceGrid* surf, int index=0);

    DistanceGrid* getPrevGrid(int index=0)
    {
        return elems[index].prevGrid;
    }
    const Matrix3& getPrevRotation(int index=0) const
    {
        return elems[index].prevRotation;
    }
    const Vector3& getPrevTranslation(int index=0) const
    {
        return elems[index].prevTranslation;
    }
    double getPrevDt(int index=0) const
    {
        return elems[index].prevDt;
    }

    /// Set new grid and transform, keeping the old state to estimate velocity
    void setNewState(int index, double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation);

    /// @}

    /// Set new grid and transform, keeping the old state to estimate velocity
    void setNewState(double dt, DistanceGrid* grid, const Matrix3& rotation, const Vector3& translation);

    /// Update transformation matrices from current rigid state
    void updateState();

    // -- CollisionModel interface

    void resize(int size);

    /// Create or update the bounding volume hierarchy.
    void computeBoundingTree(int maxDepth=0);

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);
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
inline bool RigidDistanceGridCollisionElement::isFlipped() { return model->isFlipped(); }

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
    SOFA_CLASS(FFDDistanceGridCollisionModel,sofa::core::CollisionModel);

    typedef SReal GSReal;
    typedef DistanceGrid::Coord GCoord;
    class DeformedCube
    {
    public:
        DistanceGrid* grid;
        DeformedCube() : grid(NULL) {}
        int elem; ///< Index of the corresponding element in the topology
        std::set<int> neighbors; ///< Index of the neighbors (used for self-collisions)
        struct Point
        {
            GCoord bary; ///< Barycentric coordinates
            int index; ///< Index of corresponding point in DistanceGrid
        };
        vector<Point> points; ///< barycentric coordinates of included points
        vector<GCoord> normals; ///< normals in barycentric coordinates of included points
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
        vector<GCoord> deformedNormals; ///< deformed normals
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
    Data< double > sampling;
    Data< helper::fixed_array<DistanceGrid::Coord,2> > box;
    Data< int > nx;
    Data< int > ny;
    Data< int > nz;
    sofa::core::objectmodel::DataFileName dumpfilename;

    core::behavior::MechanicalState<Vec3Types>* ffd;
    core::topology::BaseMeshTopology* ffdMesh;
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
    Data< bool > singleContact;

    FFDDistanceGridCollisionModel();

    ~FFDDistanceGridCollisionModel();

    core::behavior::MechanicalState<DataTypes>* getDeformModel() { return ffd; }
    core::topology::BaseMeshTopology* getDeformGrid() { return ffdMesh; }

    // alias used by ContactMapper

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return ffd; }
    core::topology::BaseMeshTopology* getMeshTopology() { return ffdMesh; }

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

    bool canCollideWithElement(int index, CollisionModel* model2, int index2);

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);
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
