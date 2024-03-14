/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <CollisionOBBCapsule/config.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace collisionobbcapsule::geometry
{
using namespace sofa;

template<class DataTypes>
class CapsuleCollisionModel;

template<class DataTypes>
class TCapsule;

/**
  *A capsule can be viewed as a segment with a radius, here the segment is
  *defined by its apexes.
  */
template< class MyReal>
class TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> > : public core::TCollisionElementIterator< CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> > >
{
public:
    typedef sofa::defaulttype::StdRigidTypes<3,MyReal> DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef CapsuleCollisionModel<DataTypes> ParentModel;

    using Index = sofa::Index;

    TCapsule(ParentModel* model, Index index);

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

    const Coord & v()const;

    void displayIndex()const{
        msg_info("TCapsule") << "index "<< this->index ;
    }

    bool shareSameVertex(const TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> > & other)const;
};
using RigidCapsule = TCapsule<sofa::defaulttype::Rigid3Types>;


/**
  *CapsuleCollisionModel templated by RigidTypes (frames), direction is given by Y direction of the frame.
  */
template< class MyReal>
class CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> > : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CapsuleCollisionModel, SOFA_TEMPLATE2(sofa::defaulttype::StdRigidTypes, 3, MyReal)), core::CollisionModel);


    typedef sofa::defaulttype::StdRigidTypes<3,MyReal> DataTypes;
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
    Data<VecReal > d_capsule_radii; ///< Radius of each capsule
    Data<VecReal > d_capsule_heights; ///< The capsule heights

    Data<Real> d_default_radius; ///< The default radius
    Data<Real> d_default_height; ///< The default height

    sofa::type::vector<std::pair<int,int> > _capsule_points;

    CapsuleCollisionModel();
    CapsuleCollisionModel(core::behavior::MechanicalState<DataTypes>* mstate );
public:
    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const core::visual::VisualParams* vparams, sofa::Index index) override;

    void draw(const core::visual::VisualParams* vparams) override;


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    Real radius(sofa::Index index) const;

    const Coord & center(sofa::Index i)const;

    Coord point1(sofa::Index index)const;

    Coord point2(sofa::Index index)const;

    //Returns the point1-point2 normalized vector
    Coord axis(sofa::Index index)const;

    const sofa::type::Quat<SReal> orientation(sofa::Index index)const;

    Real height(sofa::Index index)const;

    inline sofa::Size nbCap()const;

    Real defaultRadius()const;

    const Coord & velocity(sofa::Index index)const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr && context->getMechanicalState() != nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    Data<VecReal > & writeRadii();
protected:
    core::behavior::MechanicalState<DataTypes>* _mstate;
};


template<class MyReal>
inline TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCapsule(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class MyReal>
inline TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCapsule(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

#if !defined(SOFA_COMPONENT_COLLISION_RIGIDCAPSULECOLLISIONMODEL_CPP)
extern template class COLLISIONOBBCAPSULE_API TCapsule<defaulttype::Rigid3Types>;
extern template class COLLISIONOBBCAPSULE_API CapsuleCollisionModel<defaulttype::Rigid3Types>;
#endif

} // namespace collisionobbcapsule::geometry
