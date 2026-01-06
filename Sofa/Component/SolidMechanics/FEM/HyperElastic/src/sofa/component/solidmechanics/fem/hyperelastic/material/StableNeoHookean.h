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

#include <sofa/component/solidmechanics/fem/hyperelastic/material/HyperelasticMaterial.h>

namespace sofa::component::solidmechanics::fem::hyperelastic::material
{

/**
 * Stable Neo-Hookean material
 * From:
 * "Smith, Breannan, Fernando De Goes, and Theodore Kim. "Stable neo-hookean
 * flesh simulation." ACM Transactions on Graphics (TOG) 37.2 (2018): 1-15.)"
 */
template <class DataTypes>
class StableNeoHookean : public HyperelasticMaterial<DataTypes>
{
public:
    static constexpr std::string_view Name = "StableNeoHookean";

    typedef typename DataTypes::Coord::value_type Real;
    typedef type::Mat<6, 6, Real> Matrix6;
    typedef type::MatSym<3, Real> MatrixSym;

    /**
     * Strain energy density function for a stable Neo-Hookean material.
     * The regularized origin barrier is removed according to "Kim, Theodore,
     * and David Eberle. "Dynamic deformables: implementation and production
     * practicalities (now with code!)." ACM SIGGRAPH 2022 Courses. 2022. 1-259."
     */
    Real getStrainEnergy(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //rest stabilization term
        const Real alpha = 1 + mu / (lambda + mu);

        //First Right Cauchy-Green invariant
        const Real I_C = sinfo->trC;

        //Relative volume change -> J = det(F)
        const Real J = sinfo->J;

        return static_cast<Real>(0.5) *
            (mu * (I_C - 3) + (lambda + mu) * std::pow(J - alpha, 2));
    }

    /**
     * Compute the second Piola-Kirchhoff stress tensor in terms of the right
     * Cauchy-Green deformation tensor
     */
    void deriveSPKTensor(StrainInformation<DataTypes>* sinfo,
                         const MaterialParameters<DataTypes>& param,
                         MatrixSym& SPKTensorGeneral) override
    {
        // right Cauchy-Green deformation tensor
        const auto& C = sinfo->deformationTensor;

        // Inverse of C
        MatrixSym C_1;
        invertMatrix(C_1, C);

        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //rest stabilization term
        const Real alpha = 1 + mu / (lambda + mu);

        //Relative volume change -> J = det(F)
        const Real J = sinfo->J;

        //Second Piola-Kirchoff stress tensor is written in terms of C:
        // PK2 = 2 * dW/dC
        SPKTensorGeneral = mu * ID + ((lambda + mu) * J * (J - alpha)) * C_1;
    }

    void applyElasticityTensor(StrainInformation<DataTypes>* sinfo,
                               const MaterialParameters<DataTypes>& param,
                               const MatrixSym& inputTensor, MatrixSym& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //rest stabilization term
        const Real alpha = 1 + mu / (lambda + mu);

        //Relative volume change -> J = det(F)
        const Real J = sinfo->J;

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

        outputTensor = 0.5_sreal * (lambda + mu) * (Firstmatrix * (-2 * J * (J - alpha))
            + inverse_C * (J * (2 * J - alpha) * trHC));
    }

    void ElasticityTensor(StrainInformation<DataTypes>* sinfo,
                          const MaterialParameters<DataTypes>& param, Matrix6& outputTensor) override
    {
        //Lamé constants
        const Real mu = param.parameterArray[0];
        const Real lambda = param.parameterArray[1];

        //rest stabilization term
        const Real alpha = 1 + mu / (lambda + mu);

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

        //Relative volume change -> J = det(F)
        const Real J = sinfo->J;

        outputTensor = (lambda + mu) *
            (C_H_C * (-2 * J * (J - alpha)) + trC_HC_ * (J * (2 * J-alpha))) ;
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

}
