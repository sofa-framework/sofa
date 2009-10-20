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
#ifndef SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_H
#define SOFA_COMPONENT_COLLISION_TRIANGLEMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/collision/LocalMinDistanceFilter.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/topology/TriangleData.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/collision/PointModel.h>
#include <map>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class TriangleModel;
class TriangleLocalMinDistanceFilter;

class Triangle : public core::TCollisionElementIterator<TriangleModel>
{
public:
    Triangle(TriangleModel* model, int index);
    Triangle() {};
    explicit Triangle(core::CollisionElementIterator& i);

    const Vector3& p1() const;
    const Vector3& p2() const;
    const Vector3& p3() const;
    int p1Index() const;
    int p2Index() const;
    int p3Index() const;

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

class SOFA_COMPONENT_COLLISION_API TriangleModel : public core::CollisionModel
{
public:
    typedef Vec3Types InDataTypes;
    typedef Vec3Types DataTypes;
    typedef DataTypes::VecCoord VecCoord;
    typedef DataTypes::VecDeriv VecDeriv;
    typedef DataTypes::Coord Coord;
    typedef DataTypes::Deriv Deriv;
    typedef Triangle Element;
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


    Data<bool> bothSide; // to activate collision on both side of the triangle model

protected:
    struct TriangleInfo
    {
        //int i1,i2,i3;
        //int flags;
        Vector3 normal;

        /// Output stream
        inline friend std::ostream& operator<< ( std::ostream& os, const TriangleInfo& ti )
        {
            return os << ti.normal;
        }

        /// Input stream
        inline friend std::istream& operator>> ( std::istream& in, TriangleInfo& ti )
        {
            return in >> ti.normal;
        }
    };

    topology::TriangleData<TriangleInfo> elems;
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles* triangles;

    sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles mytriangles;

    bool needsUpdate;
    virtual void updateFromTopology();
    virtual void updateFlags(int ntri=-1);
    int getTriangleFlags(int i);
    virtual void updateNormals();

    core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate;
    Data<bool> computeNormals;
    int meshRevision;

    sofa::core::componentmodel::topology::BaseMeshTopology* _topology;

    PointModel* mpoints;

    TriangleLocalMinDistanceFilter *m_lmdFilter;

public:

    TriangleModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    void draw();

    virtual bool canCollideWithElement(int index, CollisionModel* model2, int index2);

    virtual void handleTopologyChange();

    core::componentmodel::behavior::MechanicalState<Vec3Types>* getMechanicalState() { return mstate; }
    const core::componentmodel::behavior::MechanicalState<Vec3Types>* getMechanicalState() const { return mstate; }

    const VecCoord& getX() const { return *(getMechanicalState()->getX()); }
    const sofa::core::componentmodel::topology::BaseMeshTopology::SeqTriangles& getTriangles() const { return *triangles; }

    TriangleLocalMinDistanceFilter *getFilter() const;

    //template< class TFilter >
    //TFilter *getFilter() const
    //{
    //	if (m_lmdFilter != 0)
    //		return m_lmdFilter;
    //	else
    //		return &m_emptyFilter;
    //}

    void setFilter(TriangleLocalMinDistanceFilter * /*lmdFilter*/);
};

inline Triangle::Triangle(TriangleModel* model, int index)
    : core::TCollisionElementIterator<TriangleModel>(model, index)
{}

inline Triangle::Triangle(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<TriangleModel>(static_cast<TriangleModel*>(i.getCollisionModel()), i.getIndex())
{}

inline const Vector3& Triangle::p1() const { return (*model->mstate->getX())[(*(model->triangles))[index][0]]; }
inline const Vector3& Triangle::p2() const { return (*model->mstate->getX())[(*(model->triangles))[index][1]]; }
inline const Vector3& Triangle::p3() const { return (*model->mstate->getX())[(*(model->triangles))[index][2]]; }

inline const Vector3& Triangle::p1Free() const { return (*model->mstate->getXfree())[(*(model->triangles))[index][0]]; }
inline const Vector3& Triangle::p2Free() const { return (*model->mstate->getXfree())[(*(model->triangles))[index][1]]; }
inline const Vector3& Triangle::p3Free() const { return (*model->mstate->getXfree())[(*(model->triangles))[index][2]]; }

inline int Triangle::p1Index() const { return (*(model->triangles))[index][0]; }
inline int Triangle::p2Index() const { return (*(model->triangles))[index][1]; }
inline int Triangle::p3Index() const { return (*(model->triangles))[index][2]; }

inline const Vector3& Triangle::v1() const { return (*model->mstate->getV())[(*(model->triangles))[index][0]]; }
inline const Vector3& Triangle::v2() const { return (*model->mstate->getV())[(*(model->triangles))[index][1]]; }
inline const Vector3& Triangle::v3() const { return (*model->mstate->getV())[(*(model->triangles))[index][2]]; }

inline const Vector3& Triangle::n() const { return model->elems.getValue()[index].normal; }
inline       Vector3& Triangle::n()       { return (*model->elems.beginEdit())[index].normal; }

inline int            Triangle::flags() const { return model->getTriangleFlags(index); }

} // namespace collision

} // namespace component

} // namespace sofa

#endif
