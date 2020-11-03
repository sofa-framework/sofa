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
#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/ForceField.h>

namespace sofa::constraint
{

template <class DataTypes>
class MappingGeometricStiffnessForceField final : public sofa::core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MappingGeometricStiffnessForceField, DataTypes), 
               SOFA_TEMPLATE(sofa::core::behavior::ForceField,DataTypes) );
    
    typedef Inherit1 Inherit;
    typedef sofa::SingleLink< MyType, sofa::core::BaseMapping, 
        sofa::BaseLink::FLAG_STRONGLINK | sofa::BaseLink::FLAG_STOREPATH > MappingLink;
    typedef typename Inherit::DataVecDeriv DataVecDeriv;
    typedef typename Inherit::DataVecCoord DataVecCoord;

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const DataVecCoord&) const override
    {
        return 0;
    }

protected:
    MappingGeometricStiffnessForceField();

    ~MappingGeometricStiffnessForceField();

private:
    MappingLink l_mapping;
};

} // namespace sofa::constraint
