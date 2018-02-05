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
#ifndef SOFA_COMPONENT_COLLISION_CAPSULEMODEL_H
#define SOFA_COMPONENT_COLLISION_CAPSULEMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>


namespace sofa
{

namespace component
{

namespace collision
{


template<class DataTypes>
class TCapsuleModel;

/**
  *A capsule can be viewed as a segment with a radius, here the segment is
  *defined by its apexes.
  */
template<class TDataTypes>
class TCapsule : public core::TCollisionElementIterator< TCapsuleModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef TCapsuleModel<DataTypes> ParentModel;

    TCapsule(ParentModel* model, int index);

    explicit TCapsule(const core::CollisionElementIterator& i);

    /**
      *Gives one apex of the capsule segment.
      */
    Coord point1()const;

    /**
      *Gives other apex of the capsule segment.
      */
    Coord point2()const;

    Coord axis()const;

    Real radius() const;

    Coord v()const;

    bool shareSameVertex(const TCapsule<TDataTypes> & other)const;
};

/**
  *A capsule model is a set of capsules. It is linked to a topology more precisely edge topology since a capsule
  *is a segment with a radius.
  */
template< class TDataTypes>
class TCapsuleModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TCapsuleModel, TDataTypes), core::CollisionModel);
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename  DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef TCapsule<DataTypes> Element;
    friend class TCapsule<DataTypes>;
protected:
    Data<VecReal > _capsule_radii;
    Data<Real> _default_radius;
    sofa::helper::vector<std::pair<int,int> > _capsule_points;

    TCapsuleModel();
    TCapsuleModel(core::behavior::MechanicalState<TDataTypes>* mstate );
public:
    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const core::visual::VisualParams* vparams,int index) override;

    void draw(const core::visual::VisualParams* vparams) override;


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    Real radius(int index) const;

    inline const Coord & point(int i)const;

    const Coord & point1(int index)const;

    const Coord & point2(int index)const;

    //Returns the point1-point2 normalized vector
    Coord axis(int index)const;

    sofa::defaulttype::Quaternion orientation(int index)const;

    int point1Index(int index)const;

    int point2Index(int index)const;

    Coord center(int index)const;

    Real height(int index)const;

    inline unsigned int nbCap()const;

    Real defaultRadius()const;

    Deriv velocity(int index)const;

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

    static std::string templateName(const TCapsuleModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /**
      *Returns true if capsules at indexes i1 and i2 share the same vertex.
      */
    bool shareSameVertex(int i1,int i2)const;

    Data<VecReal > & writeRadii();
protected:
    core::behavior::MechanicalState<DataTypes>* _mstate;
};

template<class DataTypes>
inline TCapsule<DataTypes>::TCapsule(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TCapsule<DataTypes>::TCapsule(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}


typedef TCapsuleModel<sofa::defaulttype::Vec3Types> CapsuleModel;
typedef TCapsule<sofa::defaulttype::Vec3Types> Capsule;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_CAPSULEMODEL_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
