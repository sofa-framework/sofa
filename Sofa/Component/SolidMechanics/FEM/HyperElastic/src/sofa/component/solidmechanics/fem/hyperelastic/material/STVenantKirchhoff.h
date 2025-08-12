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

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/material/HyperelasticMaterial.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <string>


namespace sofa::component::solidmechanics::fem::hyperelastic::material
{
/**
 * Saint Venant-Kirchhoff material
 */
template <class DataTypes>
class STVenantKirchhoff : public HyperelasticMaterial<DataTypes>
{
    typedef typename DataTypes::Coord::value_type Real;
    typedef type::Mat<3, 3, Real> Matrix3;
    typedef type::Mat<6, 6, Real> Matrix6;
    typedef type::MatSym<3, Real> MatrixSym;

public:
    static constexpr std::string_view Name = "StVenantKirchhoff";

    Real getStrainEnergy(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        // right Cauchy-Green deformation tensor
        const MatrixSym& C = sinfo->deformationTensor;

        //trace(C) -> first invariant
        const Real I1 = sinfo->trC;

        //trace(C*C)
        const Real trCxC = C[0] * C[0] + C[2] * C[2] + C[5] * C[5]
                + 2 * (C[1] * C[1] + C[3] * C[3] + C[4] * C[4]);

        //trace of the strain tensor in terms of the first invariant
        // E = 1/2 * (C-I)
        // => tr(E) = 1/2 * (tr(C) - 3)
        const Real trE = 0.5 * (I1 - 3);

        //trace(E*E) = 1/4 * tr( (C-I)^2)
        const Real trE_2 = 0.25 * (trCxC - 2 * I1 + 3);

        return 0.5 * lambda * trE * trE + mu * trE_2;
    }

    void deriveSPKTensor(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param, MatrixSym& SPKTensorGeneral) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        // right Cauchy-Green deformation tensor
        const MatrixSym& C = sinfo->deformationTensor;

        //trace(C) -> first invariant
        const Real I1 = sinfo->trC;

        //trace of the strain tensor in terms of the first invariant
        // E = 1/2 * (C-I)
        // => tr(E) = 1/2 * (tr(C) - 3)
        const Real trE = 0.5 * (I1 - 3);

        //the usual formulation is (lambda * trE) * ID + (2 * mu) * E
        //but we simplify it by replacing E by 1/2 * (C-I)
        SPKTensorGeneral = (lambda * trE - mu) * ID + mu * C;
    }

    void applyElasticityTensor(StrainInformation<DataTypes>*,
                               const MaterialParameters<DataTypes>& param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        const Real trH = sofa::type::trace(inputTensor);

        outputTensor = ID * (trH * lambda / 2.0) + inputTensor * mu;
    }

    void ElasticityTensor(StrainInformation<DataTypes>*,
                          const MaterialParameters<DataTypes>& param, Matrix6& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        Matrix6 IDHID;
        IDHID.identity();

        Matrix6 trIDHID;
        trIDHID[0] = ID[0] * ID;
        trIDHID[1] = ID[1] * ID;
        trIDHID[2] = ID[2] * ID;
        trIDHID[3] = ID[3] * ID;
        trIDHID[4] = ID[4] * ID;
        trIDHID[5] = ID[5] * ID;

        outputTensor = lambda * trIDHID + (2 * mu) * IDHID;
    }

private:
    //identity tensor
    inline static const MatrixSym ID = []()
    {
        MatrixSym id;
        id.identity();
        return id;
    }();
};
} // namespace sofa::component::solidmechanics::fem::hyperelastic::material
