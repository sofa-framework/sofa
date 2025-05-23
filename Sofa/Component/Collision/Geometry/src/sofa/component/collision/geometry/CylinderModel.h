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
#include <sofa/component/collision/geometry/config.h>

#include <sofa/core/CollisionModel.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::component::collision::geometry
{

template<class DataTypes>
class CylinderCollisionModel;

template<class DataTypes>
class TCylinder;

/**
  *A Cylinder can be viewed as a segment with a radius, here the segment is
  *defined by its apexes.
  */
template< class TDataTypes>
class TCylinder: public core::TCollisionElementIterator< CylinderCollisionModel< TDataTypes > >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename TDataTypes::CPos Coord;
    typedef typename TDataTypes::Deriv Deriv;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef CylinderCollisionModel<DataTypes> ParentModel;

    TCylinder(ParentModel* model, Index index);

    explicit TCylinder(const core::CollisionElementIterator& i);

    Coord axis()const;

    Real radius() const;

    Coord point1() const;
    Coord point2() const;

    const Coord & v()const;
};
using Cylinder = TCylinder<sofa::defaulttype::Rigid3Types>;


/**
  *CylinderModel templated by RigidTypes (frames), direction is given by Y direction of the frame.
  */
template< class TDataTypes>
class CylinderCollisionModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(CylinderCollisionModel, TDataTypes), core::CollisionModel);

    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename  DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename type::vector<typename DataTypes::Vec3> VecAxisCoord;

    typedef TCylinder<DataTypes> Element;
    friend class TCylinder<DataTypes>;

    Data<VecReal> d_cylinder_radii; ///< Radius of each cylinder
    Data<VecReal> d_cylinder_heights; ///< The cylinder heights
    Data<VecAxisCoord> d_cylinder_local_axes;

    Data<Real> d_default_radius; ///< The default radius
    Data<Real> d_default_height; ///< The default height
    Data<Coord> d_default_local_axis; ///< The default local axis cylinder is modeled around

protected:
    CylinderCollisionModel();
    CylinderCollisionModel(core::behavior::MechanicalState<DataTypes>* mstate );

public:
    void init() override;

    // -- CollisionModel interface
    void resize(sofa::Size size) override;

    void computeBoundingTree(int maxDepth=0) override;

    void draw(const core::visual::VisualParams* vparams,sofa::Index index) override;

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return m_mstate; }

    Real radius(sofa::Index index) const;

    const Coord & center(sofa::Index i)const;

    //Returns the direction of the cylinder at index index
    Coord axis(sofa::Index index)const;
    //Returns the direction of the cylinder at index in local coordinates
    Coord local_axis(sofa::Index index) const;

    const sofa::type::Quat<SReal> orientation(sofa::Index index)const;

    Real height(sofa::Index index)const;

    Coord point1(sofa::Index i) const;

    Coord point2(sofa::Index i) const;

    Real defaultRadius()const;

    const Coord & velocity(sofa::Index index)const;

    Data<VecReal>& writeRadii();
    Data<VecReal>& writeHeights();
    Data<VecAxisCoord>& writeLocalAxes();

protected:
    core::behavior::MechanicalState<DataTypes>* m_mstate;
};


template<class DataTypes>
inline TCylinder<DataTypes>::TCylinder(ParentModel* model, Index index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TCylinder<DataTypes>::TCylinder(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{}

#if !defined(SOFA_COMPONENT_COLLISION_CYLINDERCOLLISIONMODEL_CPP)
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API TCylinder<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_COLLISION_GEOMETRY_API CylinderCollisionModel<defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::collision::geometry
