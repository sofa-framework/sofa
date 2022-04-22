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

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>

namespace collisionobbcapsule::geometry
{

using namespace sofa;

template<class DataTypes>
class OBBCollisionModel;

/**
  *An OBB model is a set of OBBs. It is linked to a rigid mechanical object. Each frame
  *of the rigid machanical object represents the frame of one OBB. So an OBB is represented
  *by its frame which orients it, a center and 3 extents.
  *A point P is inside the OBB obb if and only if P = obb.center() + a*obb.axis(0) + b*obb.axis(1) + c*obb.axis(2)
  *with -obb.extent(0) <= a <= obb.extent(0), -obb.extent(1) <= b <= obb.extent(1), -obb.extent(2) <= c <= obb.extent(2).
  *(obb.axis(i) is the local frame axis for i-th dimension)
  */
template<class TDataTypes>
class TOBB : public core::TCollisionElementIterator< OBBCollisionModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Quat Quaternion;

    typedef OBBCollisionModel<DataTypes> ParentModel;

    using Index = sofa::Index;

    TOBB(ParentModel* model, Index index);

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

using OBB = TOBB<sofa::defaulttype::Rigid3Types>;


template< class TDataTypes>
class OBBCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(OBBCollisionModel, TDataTypes), sofa::core::CollisionModel);
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::Coord::Pos Coord;
    typedef sofa::type::vector<Coord> VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Quat Quaternion;

    typedef TOBB<DataTypes> Element;
    friend class TOBB<DataTypes>;

    sofa::core::objectmodel::Data<VecCoord> ext; ///< Extents in x,y and z directions
    sofa::core::objectmodel::Data<Real> default_ext; ///< Default extent
protected:
    OBBCollisionModel();
    OBBCollisionModel(sofa::core::behavior::MechanicalState<TDataTypes>* mstate );
public:
    void init() override;

    // -- CollisionModel interface

    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const sofa::core::visual::VisualParams* vparams, sofa::Index index) override;

    void draw(const sofa::core::visual::VisualParams* vparams) override;

    sofa::core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<sofa::core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == nullptr && context->getMechanicalState() != nullptr)
        {
            arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    /**
      *Returns the axis of the local frame at i-th dimension of the OBB at index index.
      */
    Coord axis(sofa::Index index, int dim)const;

    /**
      *Returns the 3 extents.
      */
    Real extent(sofa::Index index,int dim)const;

    /**
      *Returns the extent at i-th dimension of the OBB at index index.
      */
    const Coord & extents(sofa::Index index)const;

    const Coord & center(sofa::Index index)const;

    /**
      *Returns the quaterion representing the rotation of the local frame of the OBB at index index.
      */
    const Quaternion & orientation(sofa::Index index)const;

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
    Coord vertex(sofa::Index index,int num)const;

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
    void vertices(sofa::Index index,std::vector<Coord> & vs)const;

    /**
      *Fills v_axes of size 3 with the local frame of the OBB at index index.
      */
    void axes(sofa::Index index,Coord * v_axes)const;

    /**
      *Returns linear velocity.
      */
    const Coord & lvelocity(sofa::Index index)const;

    /**
      *Returns the coordinates of c (in general coordinate system) in the local frame of the OBB at index index.
      */
    Coord localCoordinates(const Coord & c, sofa::Index index)const;

    /**
      *Returns the coordinates of c (in the local frame) in the general coordinate system of the OBB at index index.
      */
    Coord generalCoordinates(const Coord & c, sofa::Index index)const;

    sofa::core::objectmodel::Data<VecCoord> & writeExtents();


    void computeBBox(const sofa::core::ExecParams* params, bool onlyVisible=false) override;

protected:
    sofa::core::behavior::MechanicalState<DataTypes>* _mstate;
};

template<class DataTypes>
inline TOBB<DataTypes>::TOBB(ParentModel* model, sofa::Index index)
    : sofa::core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TOBB<DataTypes>::TOBB(const sofa::core::CollisionElementIterator& i)
    : sofa::core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

#if !defined(SOFA_COMPONENT_COLLISION_OBBMODEL_CPP)
extern template class COLLISIONOBBCAPSULE_API TOBB<sofa::defaulttype::Rigid3Types>;
extern template class COLLISIONOBBCAPSULE_API OBBCollisionModel<sofa::defaulttype::Rigid3Types>;
#endif

} // namespace collisionobbcapsule::geometry
