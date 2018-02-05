/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef SOFA_COMPONENT_COLLISION_OBBMODEL_H
#define SOFA_COMPONENT_COLLISION_OBBMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>
#include <SofaBaseCollision/CubeModel.h>
//#include <SofaBaseCollision/Intersector.h>

namespace sofa
{

namespace component
{

namespace collision
{

template<class DataTypes>
class TOBBModel;

/**
  *An OBB model is a set of OBBs. It is linked to a rigid mechanical object. Each frame
  *of the rigid machanical object represents the frame of one OBB. So an OBB is represented
  *by its frame which orients it, a center and 3 extents.
  *A point P is inside the OBB obb if and only if P = obb.center() + a*obb.axis(0) + b*obb.axis(1) + c*obb.axis(2)
  *with -obb.extent(0) <= a <= obb.extent(0), -obb.extent(1) <= b <= obb.extent(1), -obb.extent(2) <= c <= obb.extent(2).
  *(obb.axis(i) is the local frame axis for i-th dimension)
  */
template<class TDataTypes>
class TOBB : public core::TCollisionElementIterator< TOBBModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Quat Quaternion;

    typedef TOBBModel<DataTypes> ParentModel;

    TOBB(ParentModel* model, int index);

    explicit TOBB(const core::CollisionElementIterator& i);

    /**
      *Returns the axis of the local frame at i-th dimension.
      */
    Coord axis(int i)const;

    /**
      *Fills v_axes of size 3 with the local frame.
      */
    void axes(Coord * v_axes)const;

    /**
      *Returns the extent at i-th dimension.
      */
    Real extent(int i)const;

    /**
      *Returns the 3 extents.
      */
    const Coord & extents()const;
    const Coord & center()const;

    /**
      *Returns the quaterion representing the rotation of the local frame.
      */
    const Quaternion & orientation()const;

    /**
      *Returns linear velocity.
      */
    const Coord & v()const;

    /**
      *Returns the coordinates of c (in general coordinate system) in the local frame.
      */
    Coord localCoordinates(const Coord &c)const;

    /**
      *Returns the coordinates of c (in the local frame) in the general coordinate system.
      */
    Coord generalCoordinates(const Coord &c)const;

    /**
      *Returns the 8 vertices in vs in general coordinate system.
      *vertex indexation below :
      *
      *                                         7--------6
      *                                        /|       /|
      *                                       3--------2 |
      *                                       | |      | |
      *                                       | 4------|-5
      *                                       |/       |/
      *                                       0--------1
      *
      */
    void vertices(std::vector<Coord> & vs)const;

    void showVertices()const;

    bool onSurface(const Coord & p)const;
};


template< class TDataTypes>
class TOBBModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TOBBModel, TDataTypes), core::CollisionModel);
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef helper::vector<Coord> VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Quat Quaternion;

    typedef TOBB<DataTypes> Element;
    friend class TOBB<DataTypes>;

    Data<VecCoord> ext;
    Data<Real> default_ext;
protected:
    TOBBModel();
    TOBBModel(core::behavior::MechanicalState<TDataTypes>* mstate );
public:
    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const core::visual::VisualParams* vparams,int index) override;

    void draw(const core::visual::VisualParams* vparams) override;

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == NULL && context->getMechanicalState() != NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const TOBBModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /**
      *Returns the axis of the local frame at i-th dimension of the OBB at index index.
      */
    Coord axis(int index,int dim)const;

    /**
      *Returns the 3 extents.
      */
    Real extent(int index,int dim)const;

    /**
      *Returns the extent at i-th dimension of the OBB at index index.
      */
    const Coord & extents(int index)const;

    const Coord & center(int index)const;

    /**
      *Returns the quaterion representing the rotation of the local frame of the OBB at index index.
      */
    const Quaternion & orientation(int index)const;

    //num is the vertex number
    //vertex indexation below :
    //
    //                                         7--------6
    //                                        /|       /|
    //                                       3--------2 |
    //                                       | |      | |
    //                                       | 4------|-5
    //                                       |/       |/
    //                                       0--------1
    //
    Coord vertex(int index,int num)const;

    /**
      *Returns the 8 vertices in vs in general coordinate system of the OBB at index index.
      *vertex indexation below :
      *
      *                                         7--------6
      *                                        /|       /|
      *                                       3--------2 |
      *                                       | |      | |
      *                                       | 4------|-5
      *                                       |/       |/
      *                                       0--------1
      *
      */
    void vertices(int index,std::vector<Coord> & vs)const;

    /**
      *Fills v_axes of size 3 with the local frame of the OBB at index index.
      */
    void axes(int index,Coord * v_axes)const;

    /**
      *Returns linear velocity.
      */
    const Coord & lvelocity(int index)const;

    /**
      *Returns the coordinates of c (in general coordinate system) in the local frame of the OBB at index index.
      */
    Coord localCoordinates(const Coord & c,int index)const;

    /**
      *Returns the coordinates of c (in the local frame) in the general coordinate system of the OBB at index index.
      */
    Coord generalCoordinates(const Coord & c,int index)const;

    Data<VecCoord> & writeExtents();


    virtual void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

protected:
    core::behavior::MechanicalState<DataTypes>* _mstate;
};

template<class DataTypes>
inline TOBB<DataTypes>::TOBB(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TOBB<DataTypes>::TOBB(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}


typedef TOBBModel<sofa::defaulttype::Rigid3Types> OBBModel;
typedef TOBB<sofa::defaulttype::Rigid3Types> OBB;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_OBBMODEL_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TOBB<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_COLLISION_API TOBBModel<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_OBBMODEL_H
