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
 * Compressible Neo-Hookean material
 */
template <class DataTypes>
class NeoHookean : public HyperelasticMaterial<DataTypes>
{
public:
    static constexpr std::string_view Name = "NeoHookean";

    typedef typename DataTypes::Coord::value_type Real;
    typedef type::Mat<3, 3, Real> Matrix3;
    typedef type::Mat<6, 6, Real> Matrix6;
    typedef type::MatSym<3, Real> MatrixSym;

    /**
     * Strain energy density function for a compressible Neo-Hookean material,
     * taken from:
     * "Javier Bonet and Richard D Wood. 2008. Nonlinear continuum mechanics for
     * finite element analysis. Cambridge University Press"
     */
    Real getStrainEnergy(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //trace(C) -> first invariant
        const Real IC = sinfo->trC;

        //det(F) = J
        const Real J = sinfo->J;

        return 0.5 * mu * (IC - 3)
            - mu * std::log(J)
            + 0.5 * lambda * std::pow(std::log(J), 2);
    }

    /**
     * Compute the second Piola-Kirchhoff stress tensor in terms of the right
     * Cauchy-Green deformation tensor
     */
    void deriveSPKTensor(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param,
                         MatrixSym& SPKTensorGeneral) override
    {
        // inverse of the right Cauchy-Green deformation tensor
        MatrixSym inverse_C;
        sofa::type::invertMatrix(inverse_C, sinfo->deformationTensor);

        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //det(F) = J
        const Real J = sinfo->J;

        //second Piola-Kirchhoff stress tensor
        SPKTensorGeneral = mu * ID
            + (lambda * std::log(J) - mu) * inverse_C;
    }


    void applyElasticityTensor(StrainInformation<DataTypes>* sinfo,
                               const MaterialParameters<DataTypes>& param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        // inverse of the right Cauchy-Green deformation tensor
        MatrixSym inverse_C;
        sofa::type::invertMatrix(inverse_C, sinfo->deformationTensor);

        // trace(C^-1 * H)
        Real trHC = inputTensor[0] * inverse_C[0] + inputTensor[2] * inverse_C[2] + inputTensor[5] * inverse_C[5]
                + 2 * inputTensor[1] * inverse_C[1] + 2 * inputTensor[3] * inverse_C[3] + 2 *
                inputTensor[4] * inverse_C[4];

        // C^-1 * H * C^-1
        MatrixSym Firstmatrix;
        MatrixSym::Mat2Sym(inverse_C * (inputTensor * inverse_C), Firstmatrix);

        outputTensor = Firstmatrix * (mu - lambda * log(sinfo->J))
            + inverse_C * (lambda * trHC / 2);
    }

    void ElasticityTensor(StrainInformation<DataTypes>* sinfo,
                          const MaterialParameters<DataTypes>& param, Matrix6& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        MatrixSym inverse_C;
        invertMatrix(inverse_C, sinfo->deformationTensor);

        MatrixSym CC;
        CC = inverse_C;
        CC[1] += inverse_C[1];
        CC[3] += inverse_C[3];
        CC[4] += inverse_C[4];

        Matrix6 C_H_C;
        C_H_C(0,0) = inverse_C[0] * inverse_C[0];
        C_H_C(1,1) = inverse_C[1] * inverse_C[1] + inverse_C[0] * inverse_C[2];
        C_H_C(2,2) = inverse_C[2] * inverse_C[2];
        C_H_C(3,3) = inverse_C[3] * inverse_C[3] + inverse_C[0] * inverse_C[5];
        C_H_C(4,4) = inverse_C[4] * inverse_C[4] + inverse_C[2] * inverse_C[5];
        C_H_C(5,5) = inverse_C[5] * inverse_C[5];
        C_H_C(1,0) = inverse_C[0] * inverse_C[1];
        C_H_C(0,1) = 2 * C_H_C(1,0);
        C_H_C(2,0) = C_H_C(0,2) = inverse_C[1] * inverse_C[1];
        C_H_C(5,0) = C_H_C(0,5) = inverse_C[3] * inverse_C[3];
        C_H_C(3,0) = inverse_C[0] * inverse_C[3];
        C_H_C(0,3) = 2 * C_H_C(3,0);
        C_H_C(4,0) = inverse_C[1] * inverse_C[3];
        C_H_C(0,4) = 2 * C_H_C(4,0);
        C_H_C(1,2) = inverse_C[2] * inverse_C[1];
        C_H_C(2,1) = 2 * C_H_C(1,2);
        C_H_C(1,5) = inverse_C[3] * inverse_C[4];
        C_H_C(5,1) = 2 * C_H_C(1,5);
        C_H_C(3,1) = C_H_C(1,3) = inverse_C[0] * inverse_C[4] + inverse_C[1] * inverse_C[3];
        C_H_C(1,4) = C_H_C(4,1) = inverse_C[1] * inverse_C[4] + inverse_C[2] * inverse_C[3];
        C_H_C(3,2) = inverse_C[4] * inverse_C[1];
        C_H_C(2,3) = 2 * C_H_C(3,2);
        C_H_C(4,2) = inverse_C[4] * inverse_C[2];
        C_H_C(2,4) = 2 * C_H_C(4,2);
        C_H_C(2,5) = C_H_C(5,2) = inverse_C[4] * inverse_C[4];
        C_H_C(3,5) = inverse_C[3] * inverse_C[5];
        C_H_C(5,3) = 2 * C_H_C(3,5);
        C_H_C(4,3) = C_H_C(3,4) = inverse_C[3] * inverse_C[4] + inverse_C[5] * inverse_C[1];
        C_H_C(4,5) = inverse_C[4] * inverse_C[5];
        C_H_C(5,4) = 2 * C_H_C(4,5);

        Matrix6 trC_HC_;
        trC_HC_[0] = inverse_C[0] * CC;
        trC_HC_[1] = inverse_C[1] * CC;
        trC_HC_[2] = inverse_C[2] * CC;
        trC_HC_[3] = inverse_C[3] * CC;
        trC_HC_[4] = inverse_C[4] * CC;
        trC_HC_[5] = inverse_C[5] * CC;

        outputTensor =
            (C_H_C * (mu - lambda * std::log(sinfo->J)) * 2 + trC_HC_ * lambda) ;
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
