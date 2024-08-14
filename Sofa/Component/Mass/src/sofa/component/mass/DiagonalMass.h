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
#include <sofa/config.h>
#include <sofa/component/mass/MeshMatrixMass.h>

SOFA_HEADER_DEPRECATED("v24.12", "v25.12", "sofa/component/mass/MeshMatrixMass.h")

#include <sofa/component/mass/config.h>
#include <sofa/type/vector.h>
#include <sofa/type/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/Mass.h>
#include <sofa/core/topology/TopologyData.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/mass/VecMassType.h>
#include <sofa/component/mass/RigidMassType.h>

#include <type_traits>
#include <string>

namespace sofa::component::mass
{

/**
* @class    DiagonalMass
* @brief    This component computes the integral of this mass density over the volume of the object geometry but it supposes that the Mass matrix is diagonal.
* @remark   Similar to MeshMatrixMass but it does not simplify the Mass Matrix as diagonal.
* @remark   https://www.sofa-framework.org/community/doc/components/masses/diagonalmass/
* @tparam   DataTypes type of the state associated with this mass
* @tparam   GeometricalTypes type of the geometry, i.e type of the state associated with the topology (if the topology and the mass relates to the same state, this will be the same as DataTypes)
*/
template <class DataTypes, class GeometricalTypes = DataTypes>
class DiagonalMass : public MeshMatrixMass<DataTypes, GeometricalTypes>
{

public:
    typedef sofa::component::mass::MeshMatrixMass<DataTypes, GeometricalTypes> Inherited;

    SOFA_CLASS(SOFA_TEMPLATE2(DiagonalMass,DataTypes, GeometricalTypes), SOFA_TEMPLATE2(MeshMatrixMass,DataTypes, GeometricalTypes));

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override
    {
        Inherited::parse(arg);

        const char* prefix = "Rigid";

        if (strncmp(arg->getAttribute("template"), prefix, std::strlen(prefix)) == 0)
        {
            msg_warning() << "DiagonalMass templated on Rigid types has been removed since #3912. "
                             "For rigid bodies, UniformMass should be prefered";
        }
    }

protected:
    DiagonalMass()
    {
        this->d_lumping.setValue(true);
    }

};

} // namespace sofa::component::mass
