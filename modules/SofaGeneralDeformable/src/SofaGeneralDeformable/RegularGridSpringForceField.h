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

#include <SofaGeneralDeformable/config.h>

#include <SofaDeformable/StiffSpringForceField.h>
#include <SofaBaseTopology/RegularGridTopology.h>


namespace sofa::component::interactionforcefield
{

template<class DataTypes>
class RegularGridSpringForceField : public StiffSpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RegularGridSpringForceField, DataTypes), SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    typedef StiffSpringForceField<DataTypes> Inherit;
    typedef typename Inherit::Spring Spring;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;


protected:
    Data< Real > linesStiffness; ///< Lines Stiffness
    Data< Real > linesDamping; ///< Lines Damping
    Data< Real > quadsStiffness; ///< Quads Stiffness
    Data< Real > quadsDamping; ///< Quads Damping
    Data< Real > cubesStiffness; ///< Cubes Stiffness
    Data< Real > cubesDamping; ///< Cubes Damping

    RegularGridSpringForceField();
    RegularGridSpringForceField(core::behavior::MechanicalState<DataTypes>* mstate);

public:
    Real getStiffness() const { return linesStiffness.getValue(); }
    Real getLinesStiffness() const { return linesStiffness.getValue(); }
    Real getQuadsStiffness() const { return quadsStiffness.getValue(); }
    Real getCubesStiffness() const { return cubesStiffness.getValue(); }
    void setStiffness(Real val)
    {
        linesStiffness.setValue(val);
        quadsStiffness.setValue(val);
        cubesStiffness.setValue(val);
    }
    void setLinesStiffness(Real val)
    {
        linesStiffness.setValue(val);
    }
    void setQuadsStiffness(Real val)
    {
        quadsStiffness.setValue(val);
    }
    void setCubesStiffness(Real val)
    {
        cubesStiffness.setValue(val);
    }

    Real getDamping() const { return linesDamping.getValue(); }
    Real getLinesDamping() const { return linesDamping.getValue(); }
    Real getQuadsDamping() const { return quadsDamping.getValue(); }
    Real getCubesDamping() const { return cubesDamping.getValue(); }
    void setDamping(Real val)
    {
        linesDamping.setValue(val);
        quadsDamping.setValue(val);
        cubesDamping.setValue(val);
    }
    void setLinesDamping(Real val)
    {
        linesDamping.setValue(val);
    }
    void setQuadsDamping(Real val)
    {
        quadsDamping.setValue(val);
    }
    void setCubesDamping(Real val)
    {
        cubesDamping.setValue(val);
    }

    void init() override;

    void addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    topology::RegularGridTopology* topology;
};
#if  !defined(SOFA_COMPONENT_FORCEFIELD_REGULARGRIDSPRINGFORCEFIELD_CPP)
extern template class SOFA_SOFAGENERALDEFORMABLE_API RegularGridSpringForceField<defaulttype::Vec3Types>;
extern template class SOFA_SOFAGENERALDEFORMABLE_API RegularGridSpringForceField<defaulttype::Vec2Types>;
extern template class SOFA_SOFAGENERALDEFORMABLE_API RegularGridSpringForceField<defaulttype::Vec1Types>;
extern template class SOFA_SOFAGENERALDEFORMABLE_API RegularGridSpringForceField<defaulttype::Vec6Types>;

#endif

} // namespace sofa::component::interactionforcefield
