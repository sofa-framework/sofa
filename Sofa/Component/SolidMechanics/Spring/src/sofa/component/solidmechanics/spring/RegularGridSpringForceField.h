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

#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/component/topology/container/grid/RegularGridTopology.h>


namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
class RegularGridSpringForceField : public SpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RegularGridSpringForceField, DataTypes), SOFA_TEMPLATE(SpringForceField, DataTypes));

    typedef SpringForceField<DataTypes> Inherit;
    typedef typename Inherit::Spring Spring;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;


protected:
    Data< Real > d_linesStiffness; ///< Lines Stiffness
    Data< Real > d_linesDamping; ///< Lines Damping
    Data< Real > d_quadsStiffness; ///< Quads Stiffness
    Data< Real > d_quadsDamping; ///< Quads Damping
    Data< Real > d_cubesStiffness; ///< Cubes Stiffness
    Data< Real > d_cubesDamping; ///< Cubes Damping

    RegularGridSpringForceField();
    RegularGridSpringForceField(core::behavior::MechanicalState<DataTypes>* object1, core::behavior::MechanicalState<DataTypes>* object2);

public:
    Real getStiffness() const { return d_linesStiffness.getValue(); }
    Real getLinesStiffness() const { return d_linesStiffness.getValue(); }
    Real getQuadsStiffness() const { return d_quadsStiffness.getValue(); }
    Real getCubesStiffness() const { return d_cubesStiffness.getValue(); }
    void setStiffness(Real val)
    {
        d_linesStiffness.setValue(val);
        d_quadsStiffness.setValue(val);
        d_cubesStiffness.setValue(val);
    }
    void setLinesStiffness(Real val)
    {
        d_linesStiffness.setValue(val);
    }
    void setQuadsStiffness(Real val)
    {
        d_quadsStiffness.setValue(val);
    }
    void setCubesStiffness(Real val)
    {
        d_cubesStiffness.setValue(val);
    }

    Real getDamping() const { return d_linesDamping.getValue(); }
    Real getLinesDamping() const { return d_linesDamping.getValue(); }
    Real getQuadsDamping() const { return d_quadsDamping.getValue(); }
    Real getCubesDamping() const { return d_cubesDamping.getValue(); }
    void setDamping(Real val)
    {
        d_linesDamping.setValue(val);
        d_quadsDamping.setValue(val);
        d_cubesDamping.setValue(val);
    }
    void setLinesDamping(Real val)
    {
        d_linesDamping.setValue(val);
    }
    void setQuadsDamping(Real val)
    {
        d_quadsDamping.setValue(val);
    }
    void setCubesDamping(Real val)
    {
        d_cubesDamping.setValue(val);
    }

    void init() override;

    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor);

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    topology::container::grid::RegularGridTopology* topology;
};
#if !defined(SOFA_COMPONENT_FORCEFIELD_REGULARGRIDSPRINGFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RegularGridSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RegularGridSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RegularGridSpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API RegularGridSpringForceField<defaulttype::Vec6Types>;

#endif

} // namespace sofa::component::solidmechanics::spring
