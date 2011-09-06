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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/collision/PointModel.h>
#include <map>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template<class DataTypes>
class TTriangleModel;

class TriangleLocalMinDistanceFilter;

template<class TDataTypes>
class TTriangle : public core::TCollisionElementIterator< TTriangleModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef TTriangleModel<DataTypes> ParentModel;

    TTriangle(ParentModel* model, int index);
    TTriangle() {}
    explicit TTriangle(core::CollisionElementIterator& i);

    const Coord& p1() const;
    const Coord& p2() const;
    const Coord& p3() const;

    int p1Index() const;
    int p2Index() const;
    int p3Index() const;

    const Coord& p1Free() const;
    const Coord& p2Free() const;
    const Coord& p3Free() const;

    const Deriv& v1() const;
    const Deriv& v2() const;
    const Deriv& v3() const;

    const Deriv& n() const;
    Deriv& n();

    /// Return true if the element stores a free position vector
    bool hasFreePosition() const;

    int flags() const;
};

template<class TDataTypes>
class TTriangleModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TTriangleModel, TDataTypes), core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef TTriangle<DataTypes> Element;
    friend class TTriangle<DataTypes>;

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
#if 0
    struct TriangleInfo
    {
        //int i1,i2,i3;
        //int flags;
        Deriv normal;

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
#endif

    //topology::TriangleData<TriangleInfo> elems;
    VecDeriv normals;

    const sofa::core::topology::BaseMeshTopology::SeqTriangles* triangles;

    sofa::core::topology::BaseMeshTopology::SeqTriangles mytriangles;

    bool needsUpdate;
    virtual void updateFromTopology();
    virtual void updateFlags(int ntri=-1);
    int getTriangleFlags(int i);
    virtual void updateNormals();

    core::behavior::MechanicalState<DataTypes>* mstate;
    Data<bool> computeNormals;
    int meshRevision;

    sofa::core::topology::BaseMeshTopology* _topology;

    PointModel* mpoints;

    TriangleLocalMinDistanceFilter *m_lmdFilter;

public:

    TTriangleModel();

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);

    virtual bool canCollideWithElement(int index, CollisionModel* model2, int index2);

    virtual void handleTopologyChange();

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }
    const core::behavior::MechanicalState<DataTypes>* getMechanicalState() const { return mstate; }

    const VecCoord& getX() const { return *(getMechanicalState()->getX()); }
    const sofa::core::topology::BaseMeshTopology::SeqTriangles& getTriangles() const { return *triangles; }

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


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const TTriangleModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

};

template<class DataTypes>
inline TTriangle<DataTypes>::TTriangle(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TTriangle<DataTypes>::TTriangle(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p1() const { return (*this->model->mstate->getX())[(*(this->model->triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p2() const { return (*this->model->mstate->getX())[(*(this->model->triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p3() const { return (*this->model->mstate->getX())[(*(this->model->triangles))[this->index][2]]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p1Free() const { return (*this->model->mstate->getXfree())[(*(this->model->triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p2Free() const { return (*this->model->mstate->getXfree())[(*(this->model->triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Coord& TTriangle<DataTypes>::p3Free() const { return (*this->model->mstate->getXfree())[(*(this->model->triangles))[this->index][2]]; }

template<class DataTypes>
inline int TTriangle<DataTypes>::p1Index() const { return (*(this->model->triangles))[this->index][0]; }
template<class DataTypes>
inline int TTriangle<DataTypes>::p2Index() const { return (*(this->model->triangles))[this->index][1]; }
template<class DataTypes>
inline int TTriangle<DataTypes>::p3Index() const { return (*(this->model->triangles))[this->index][2]; }

template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v1() const { return (*this->model->mstate->getV())[(*(this->model->triangles))[this->index][0]]; }
template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v2() const { return (*this->model->mstate->getV())[(*(this->model->triangles))[this->index][1]]; }
template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::v3() const { return (*this->model->mstate->getV())[(*(this->model->triangles))[this->index][2]]; }

template<class DataTypes>
inline const typename DataTypes::Deriv& TTriangle<DataTypes>::n() const { return this->model->normals[this->index]; }
template<class DataTypes>
inline       typename DataTypes::Deriv& TTriangle<DataTypes>::n()       { return this->model->normals[this->index]; }

template<class DataTypes>
inline int TTriangle<DataTypes>::flags() const { return this->model->getTriangleFlags(this->index); }

template<class DataTypes>
inline bool TTriangle<DataTypes>::hasFreePosition() const { return this->model->mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }

typedef TTriangleModel<Vec3Types> TriangleModel;
typedef TTriangle<Vec3Types> Triangle;

#if defined(WIN32) && !defined(SOFA_BUILD_COMPONENT_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_COLLISION_API TTriangleModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_COLLISION_API TTriangleModel<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
