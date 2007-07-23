/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/core/VisualModel.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/topology/TriangleSetTopology.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class TriangleModel;

class Triangle : public core::TCollisionElementIterator<TriangleModel>
{
public:
    Triangle(TriangleModel* model, int index);

    explicit Triangle(core::CollisionElementIterator& i);

    const Vector3& p1() const;
    const Vector3& p2() const;
    const Vector3& p3() const;

    const Vector3& p1Free() const;
    const Vector3& p2Free() const;
    const Vector3& p3Free() const;

    const Vector3& v1() const;
    const Vector3& v2() const;
    const Vector3& v3() const;

    const Vector3& n() const;
    Vector3& n();

    int flags() const;
};

class TriangleModel : public core::CollisionModel, public core::VisualModel
{
protected:
    struct TriangleInfo
    {
        int i1,i2,i3,flags;
        Vector3 normal;
    };

    std::vector<TriangleInfo> elems;

    class Loader;

    int meshRevision;
    bool needsUpdate;
    void updateFromTopology();
public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Triangle Element;
    typedef topology::MeshTopology Topology;
    friend class Triangle;

    enum TriangleFlag
    {
        FLAG_P1  = 1<<0, ///< Point 1  is attached to this triangle
        FLAG_P2  = 1<<1, ///< Point 2  is attached to this triangle
        FLAG_P3  = 1<<2, ///< Point 3  is attached to this triangle
        FLAG_E12 = 1<<3, ///< Edge 1-2 is attached to this triangle
        FLAG_E23 = 1<<4, ///< Edge 2-3 is attached to this triangle
        FLAG_E31 = 1<<5, ///< Edge 3-1 is attached to this triangle
        FLAG_POINTS  = FLAG_P1|FLAG_P2|FLAG_P3,
        FLAG_EDGES   = FLAG_E12|FLAG_E23|FLAG_E31,
    };

    TriangleModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }

    // handle topological changes
    virtual void handleTopologyChange();

    core::componentmodel::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }

    topology::MeshTopology* getTopology() { return mesh; }

    void buildOctree();

    unsigned int getNbTriangles() const;

    //virtual const char* getTypeName() const { return "Triangle"; }

protected:

    core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate;
    topology::MeshTopology* mesh;

};

inline Triangle::Triangle(TriangleModel* model, int index)
    : core::TCollisionElementIterator<TriangleModel>(model, index)
{}

inline Triangle::Triangle(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<TriangleModel>(static_cast<TriangleModel*>(i.getCollisionModel()), i.getIndex())
{
}

inline const Vector3& Triangle::p1() const { return (*model->mstate->getX())[model->elems[index].i1]; }
inline const Vector3& Triangle::p2() const { return (*model->mstate->getX())[model->elems[index].i2]; }
inline const Vector3& Triangle::p3() const { return (*model->mstate->getX())[model->elems[index].i3]; }

inline const Vector3& Triangle::p1Free() const { return (*model->mstate->getXfree())[model->elems[index].i1]; }
inline const Vector3& Triangle::p2Free() const { return (*model->mstate->getXfree())[model->elems[index].i2]; }
inline const Vector3& Triangle::p3Free() const { return (*model->mstate->getXfree())[model->elems[index].i3]; }

inline const Vector3& Triangle::v1() const { return (*model->mstate->getV())[model->elems[index].i1]; }
inline const Vector3& Triangle::v2() const { return (*model->mstate->getV())[model->elems[index].i2]; }
inline const Vector3& Triangle::v3() const { return (*model->mstate->getV())[model->elems[index].i3]; }

inline const Vector3& Triangle::n() const { return model->elems[index].normal; }
inline       Vector3& Triangle::n()       { return model->elems[index].normal; }

inline int            Triangle::flags() const { return model->elems[index].flags; }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
