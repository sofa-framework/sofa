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
#include <sofa/component/solidmechanics/spring/config.h>

#include <sofa/type/RGBAColor.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/type/vector.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/component/solidmechanics/spring/FixedWeakConstraint.h>
#include <sofa/core/objectmodel/DataCallback.h>


namespace sofa::core::behavior
{

template< class T > class MechanicalState;

} // namespace sofa::core::behavior

namespace sofa::component::solidmechanics::spring
{

/**
* @brief This class describes a simple elastic springs ForceField between DOFs positions and rest positions.
*
* Springs are applied to given degrees of freedom between their current positions and their rest shape positions.
* An external MechanicalState reference can also be passed to the ForceField as rest shape position.
*/
template<class DataTypes>
class RestShapeSpringsForceField : public FixedWeakConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RestShapeSpringsForceField, DataTypes), SOFA_TEMPLATE(FixedWeakConstraint, DataTypes));

    typedef FixedWeakConstraint<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector< sofa::Index > VecIndex;
    typedef sofa::core::topology::TopologySubsetIndices DataSubsetIndex;
    typedef type::vector< Real >	 VecReal;

    static constexpr sofa::Size spatial_dimensions = Coord::spatial_dimensions;
    static constexpr sofa::Size coord_total_size = Coord::total_size;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data< type::fixed_array<bool, coord_total_size> > d_activeDirections; ///< directions (translation, and rotation in case of Rigids) in which the spring is active
    Data< VecIndex > d_externalIndices; ///< points from the external Mechanical State that define the rest shape springs
    core::objectmodel::lifecycle::RemovedData d_external_points{this,"v24.12","v25.06","external_points","This data has been replaced by \'externalIndices\'. Please update your scene."};
    core::objectmodel::DataCallback c_fixAllCallback;

    SingleLink<RestShapeSpringsForceField<DataTypes>, sofa::core::behavior::MechanicalState< DataTypes >, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_restMState;
    SingleLink<RestShapeSpringsForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    /// BaseObject initialization method.
    void bwdInit() override ;



protected :
    RestShapeSpringsForceField();

    virtual const DataVecCoord* getExtPosition() const override;
    virtual const VecIndex& getExtIndices() const override;
    virtual const type::fixed_array<bool, coord_total_size>& getActiveDirections() const override;

    virtual bool checkOutOfBoundsIndices();

private :

    bool m_useRestMState; /// An external MechanicalState is used as rest reference.
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_RESTSHAPESPRINGSFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RestShapeSpringsForceField<sofa::defaulttype::Rigid3Types>;
#endif

} // namespace sofa::component::solidmechanics::spring
