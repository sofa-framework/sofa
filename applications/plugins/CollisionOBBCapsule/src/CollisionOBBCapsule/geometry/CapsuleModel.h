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
#include <sofa/core/topology/BaseMeshTopology.h>

namespace collisionobbcapsule::geometry
{

template<class DataTypes>
class CapsuleCollisionModel;

using namespace sofa;

/**
  *A capsule can be viewed as a segment with a radius, here the segment is
  *defined by its apexes.
  */
template<class TDataTypes>
class TCapsule : public core::TCollisionElementIterator< CapsuleCollisionModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef CapsuleCollisionModel<DataTypes> ParentModel;

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

    Deriv v()const;

    bool shareSameVertex(const TCapsule<TDataTypes> & other)const;
};
using Capsule = TCapsule<sofa::defaulttype::Vec3Types>;

/**
  *A capsule model is a set of capsules. It is linked to a topology more precisely edge topology since a capsule
  *is a segment with a radius.
  */
template< class TDataTypes>
class CapsuleCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CapsuleCollisionModel, TDataTypes), core::CollisionModel);
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

    using Index = sofa::Index;
protected:
    Data<VecReal > _capsule_radii; ///< Radius of each capsule
    Data<Real> _default_radius; ///< The default radius
    sofa::type::vector<std::pair<Index, Index> > _capsule_points;

    CapsuleCollisionModel();
    CapsuleCollisionModel(core::behavior::MechanicalState<TDataTypes>* mstate );
public:
    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const core::visual::VisualParams* vparams, Index index) override;

    void draw(const core::visual::VisualParams* vparams) override;


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    Real radius(Index index) const;

    inline const Coord & point(Index i)const;

    const Coord & point1(Index index)const;

    const Coord & point2(Index index)const;

    //Returns the point1-point2 normalized vector
    Coord axis(Index index)const;

    sofa::type::Quat<SReal> orientation(Index index)const;

    Index point1Index(Index index)const;

    Index point2Index(Index index)const;

    Coord center(Index index)const;

    Real height(Index index)const;

    inline sofa::Size nbCap()const;

    Real defaultRadius()const;

    Deriv velocity(Index index)const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == nullptr && context->getMechanicalState() != nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    sofa::core::topology::BaseMeshTopology* getCollisionTopology() override
    {
        return l_topology.get();
    }

    /**
      *Returns true if capsules at indexes i1 and i2 share the same vertex.
      */
    bool shareSameVertex(Index i1, Index i2)const;

    Data<VecReal > & writeRadii();

    /// Link to be set to the topology container in the component graph.
    SingleLink<CapsuleCollisionModel<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    core::behavior::MechanicalState<DataTypes>* _mstate;
};

template<class DataTypes>
inline TCapsule<DataTypes>::TCapsule(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TCapsule<DataTypes>::TCapsule(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

#if !defined(SOFA_COMPONENT_COLLISION_CAPSULECOLLISIONMODEL_CPP)
extern template class COLLISIONOBBCAPSULE_API TCapsule<sofa::defaulttype::Vec3Types>;
extern template class COLLISIONOBBCAPSULE_API CapsuleCollisionModel<sofa::defaulttype::Vec3Types>;
#endif

} // namespace collisionobbcapsule::geometry
