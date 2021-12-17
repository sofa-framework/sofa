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
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaSimpleFem/config.h>

namespace sofa::component::forcefield
{

template<class DataTypes>
class TetrahedronFEMForceFieldImpl
{
public:
    using Coord    = typename DataTypes::Coord;
    using Deriv    = typename DataTypes::Deriv;
    using Real     = typename Coord::value_type;
    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;

    using Element    = sofa::core::topology::BaseMeshTopology::Tetra;
    using VecElement = sofa::core::topology::BaseMeshTopology::SeqTetrahedra;

    using Force              = sofa::type::VecNoInit<12, Real>;
    using Displacement       = sofa::type::VecNoInit<12, Real>;
    using VoigtTensor        = sofa::type::VecNoInit< 6, Real>;
    using MaterialStiffness  = sofa::type::Mat      < 6,  6, Real>;
    using StrainDisplacement = sofa::type::Mat      <12,  6, Real>;
    using StiffnessMatrix    = sofa::type::Mat      <12, 12, Real>;
    using Transformation     = sofa::type::MatNoInit< 3,  3, Real>;

    using VecStrainDisplacement = sofa::type::vector<StrainDisplacement>;
    using VecMaterialStiffness  = sofa::type::vector<MaterialStiffness>;

    typedef std::pair<Index,Real> Col_Value;
    typedef type::vector< Col_Value > CompressedValue;
    typedef type::vector< CompressedValue > CompressedMatrix;

    struct PlasticityParameters
    {
        Real maxThreshold;
        Real yieldThreshold;
        Real creep;

        bool hasPlasticity() const
        {
            return maxThreshold > static_cast<Real>(0);
        }
    } m_plasticity;

    bool m_assemble { false };

    struct FiniteElementArrays
    {
        /// Input
        ///@{
        const VecElement* elements      { nullptr };
        const VecCoord*   initialPoints { nullptr };
        const VecCoord*   positions     { nullptr };
        const VecDeriv*   dx            { nullptr };
        ///@}

        /// Output
        ///@{
        VecDeriv*                        force                { nullptr };
        VecDeriv*                        dForce               { nullptr };
        VecStrainDisplacement*           strainDisplacements  { nullptr };
        sofa::type::vector<VoigtTensor>* plasticStrains       { nullptr };
        VecMaterialStiffness*            materialsStiffnesses { nullptr };
        CompressedMatrix*                stiffnesses          { nullptr };
        ///@}
    };

    virtual ~TetrahedronFEMForceFieldImpl() = default;

    virtual void init(const FiniteElementArrays& finiteElementArrays) = 0;
    virtual void addForce(const FiniteElementArrays& finiteElementArrays) = 0;
    virtual void addDForce(const FiniteElementArrays& finiteElementArrays, Real kFactor) = 0;

    virtual Displacement getDisplacement(const VecCoord& positions, const Element& element, const unsigned elementIndex) const
    {
        return {};
    }

    virtual Transformation getRotation(const unsigned int elementIndex) const
    {
        Transformation identity;
        identity.identity();
        return identity;
    }

    void initStiffnesses(const FiniteElementArrays& finiteElementArrays)
    {
        if (finiteElementArrays.stiffnesses)
        {
            if (finiteElementArrays.initialPoints)
            {
                finiteElementArrays.stiffnesses->resize( finiteElementArrays.initialPoints->size() * 3);
            }
            for (auto& s : *finiteElementArrays.stiffnesses)
            {
                s.clear();
            }
        }
    }

    static void copyForceFromElementToGlobal(VecCoord& globalForceVector, const Force& elementForce, const Element& element)
    {
        globalForceVector[element[0]] += Deriv( elementForce[0], elementForce[ 1], elementForce[ 2] );
        globalForceVector[element[1]] += Deriv( elementForce[3], elementForce[ 4], elementForce[ 5] );
        globalForceVector[element[2]] += Deriv( elementForce[6], elementForce[ 7], elementForce[ 8] );
        globalForceVector[element[3]] += Deriv( elementForce[9], elementForce[10], elementForce[11] );
    }

    static void copyForceFromElementToGlobal(VecCoord& globalForceVector, const Force& elementForce, const Element& element, const Transformation& rotation)
    {
        // globalForceVector[element[0]] += rotation * Deriv( elementForce[0], elementForce[ 1], elementForce[ 2] );
        // globalForceVector[element[1]] += rotation * Deriv( elementForce[3], elementForce[ 4], elementForce[ 5] );
        // globalForceVector[element[2]] += rotation * Deriv( elementForce[6], elementForce[ 7], elementForce[ 8] );
        // globalForceVector[element[3]] += rotation * Deriv( elementForce[9], elementForce[10], elementForce[11] );

        auto& fa = globalForceVector[element[0]];
        auto& fb = globalForceVector[element[1]];
        auto& fc = globalForceVector[element[2]];
        auto& fd = globalForceVector[element[3]];

        fa[0] += rotation[0][0] *  elementForce[0] +  rotation[0][1] * elementForce[1]  + rotation[0][2] * elementForce[2];
        fa[1] += rotation[1][0] *  elementForce[0] +  rotation[1][1] * elementForce[1]  + rotation[1][2] * elementForce[2];
        fa[2] += rotation[2][0] *  elementForce[0] +  rotation[2][1] * elementForce[1]  + rotation[2][2] * elementForce[2];

        fb[0] += rotation[0][0] *  elementForce[3] +  rotation[0][1] * elementForce[4]  + rotation[0][2] * elementForce[5];
        fb[1] += rotation[1][0] *  elementForce[3] +  rotation[1][1] * elementForce[4]  + rotation[1][2] * elementForce[5];
        fb[2] += rotation[2][0] *  elementForce[3] +  rotation[2][1] * elementForce[4]  + rotation[2][2] * elementForce[5];

        fc[0] += rotation[0][0] *  elementForce[6] +  rotation[0][1] * elementForce[7]  + rotation[0][2] * elementForce[8];
        fc[1] += rotation[1][0] *  elementForce[6] +  rotation[1][1] * elementForce[7]  + rotation[1][2] * elementForce[8];
        fc[2] += rotation[2][0] *  elementForce[6] +  rotation[2][1] * elementForce[7]  + rotation[2][2] * elementForce[8];

        fd[0] += rotation[0][0] *  elementForce[9] +  rotation[0][1] * elementForce[10] + rotation[0][2] * elementForce[11];
        fd[1] += rotation[1][0] *  elementForce[9] +  rotation[1][1] * elementForce[10] + rotation[1][2] * elementForce[11];
        fd[2] += rotation[2][0] *  elementForce[9] +  rotation[2][1] * elementForce[10] + rotation[2][2] * elementForce[11];

    }

    static StrainDisplacement computeStrainDisplacement(const Coord& a, const Coord& b, const Coord& c, const Coord& d)
    {
        StrainDisplacement J;

        // shape functions matrix
        type::Mat<2, 3, Real> M;

        const auto peudo_determinant_for_coef = [](const type::Mat<2, 3, Real>& M)
        {
            return  M[0][1]*M[1][2] - M[1][1]*M[0][2] -  M[0][0]*M[1][2] + M[1][0]*M[0][2] + M[0][0]*M[1][1] - M[1][0]*M[0][1];
        };

        M[0][0] = b[1];
        M[0][1] = c[1];
        M[0][2] = d[1];
        M[1][0] = b[2];
        M[1][1] = c[2];
        M[1][2] = d[2];
        J[0][0] = J[1][3] = J[2][5]   = - peudo_determinant_for_coef( M );
        M[0][0] = b[0];
        M[0][1] = c[0];
        M[0][2] = d[0];
        J[0][3] = J[1][1] = J[2][4]   = peudo_determinant_for_coef( M );
        M[1][0] = b[1];
        M[1][1] = c[1];
        M[1][2] = d[1];
        J[0][5] = J[1][4] = J[2][2]   = - peudo_determinant_for_coef( M );

        M[0][0] = c[1];
        M[0][1] = d[1];
        M[0][2] = a[1];
        M[1][0] = c[2];
        M[1][1] = d[2];
        M[1][2] = a[2];
        J[3][0] = J[4][3] = J[5][5]   = peudo_determinant_for_coef( M );
        M[0][0] = c[0];
        M[0][1] = d[0];
        M[0][2] = a[0];
        J[3][3] = J[4][1] = J[5][4]   = - peudo_determinant_for_coef( M );
        M[1][0] = c[1];
        M[1][1] = d[1];
        M[1][2] = a[1];
        J[3][5] = J[4][4] = J[5][2]   = peudo_determinant_for_coef( M );

        M[0][0] = d[1];
        M[0][1] = a[1];
        M[0][2] = b[1];
        M[1][0] = d[2];
        M[1][1] = a[2];
        M[1][2] = b[2];
        J[6][0] = J[7][3] = J[8][5]   = - peudo_determinant_for_coef( M );
        M[0][0] = d[0];
        M[0][1] = a[0];
        M[0][2] = b[0];
        J[6][3] = J[7][1] = J[8][4]   = peudo_determinant_for_coef( M );
        M[1][0] = d[1];
        M[1][1] = a[1];
        M[1][2] = b[1];
        J[6][5] = J[7][4] = J[8][2]   = - peudo_determinant_for_coef( M );

        M[0][0] = a[1];
        M[0][1] = b[1];
        M[0][2] = c[1];
        M[1][0] = a[2];
        M[1][1] = b[2];
        M[1][2] = c[2];
        J[9][0] = J[10][3] = J[11][5]   = peudo_determinant_for_coef( M );
        M[0][0] = a[0];
        M[0][1] = b[0];
        M[0][2] = c[0];
        J[9][3] = J[10][1] = J[11][4]   = - peudo_determinant_for_coef( M );
        M[1][0] = a[1];
        M[1][1] = b[1];
        M[1][2] = c[1];
        J[9][5] = J[10][4] = J[11][2]   = peudo_determinant_for_coef( M );

        return J;
    }

    static void computeElasticJtD(VoigtTensor& JtD, const StrainDisplacement &J, const Displacement &Depl)
    {
        JtD[0] =   J[ 0][0]*Depl[ 0]+/*J[ 1][0]*Depl[ 1]+  J[ 2][0]*Depl[ 2]+*/
                   J[ 3][0]*Depl[ 3]+/*J[ 4][0]*Depl[ 4]+  J[ 5][0]*Depl[ 5]+*/
                   J[ 6][0]*Depl[ 6]+/*J[ 7][0]*Depl[ 7]+  J[ 8][0]*Depl[ 8]+*/
                   J[ 9][0]*Depl[ 9] /*J[10][0]*Depl[10]+  J[11][0]*Depl[11]*/;
        JtD[1] = /*J[ 0][1]*Depl[ 0]+*/J[ 1][1]*Depl[ 1]+/*J[ 2][1]*Depl[ 2]+*/
                 /*J[ 3][1]*Depl[ 3]+*/J[ 4][1]*Depl[ 4]+/*J[ 5][1]*Depl[ 5]+*/
                 /*J[ 6][1]*Depl[ 6]+*/J[ 7][1]*Depl[ 7]+/*J[ 8][1]*Depl[ 8]+*/
                 /*J[ 9][1]*Depl[ 9]+*/J[10][1]*Depl[10] /*J[11][1]*Depl[11]*/;
        JtD[2] = /*J[ 0][2]*Depl[ 0]+  J[ 1][2]*Depl[ 1]+*/J[ 2][2]*Depl[ 2]+
                 /*J[ 3][2]*Depl[ 3]+  J[ 4][2]*Depl[ 4]+*/J[ 5][2]*Depl[ 5]+
                 /*J[ 6][2]*Depl[ 6]+  J[ 7][2]*Depl[ 7]+*/J[ 8][2]*Depl[ 8]+
                 /*J[ 9][2]*Depl[ 9]+  J[10][2]*Depl[10]+*/J[11][2]*Depl[11]  ;
        JtD[3] =   J[ 0][3]*Depl[ 0]+  J[ 1][3]*Depl[ 1]+/*J[ 2][3]*Depl[ 2]+*/
                   J[ 3][3]*Depl[ 3]+  J[ 4][3]*Depl[ 4]+/*J[ 5][3]*Depl[ 5]+*/
                   J[ 6][3]*Depl[ 6]+  J[ 7][3]*Depl[ 7]+/*J[ 8][3]*Depl[ 8]+*/
                   J[ 9][3]*Depl[ 9]+  J[10][3]*Depl[10] /*J[11][3]*Depl[11]*/;
        JtD[4] = /*J[ 0][4]*Depl[ 0]+*/J[ 1][4]*Depl[ 1]+  J[ 2][4]*Depl[ 2]+
                 /*J[ 3][4]*Depl[ 3]+*/J[ 4][4]*Depl[ 4]+  J[ 5][4]*Depl[ 5]+
                 /*J[ 6][4]*Depl[ 6]+*/J[ 7][4]*Depl[ 7]+  J[ 8][4]*Depl[ 8]+
                 /*J[ 9][4]*Depl[ 9]+*/J[10][4]*Depl[10]+  J[11][4]*Depl[11]  ;
        JtD[5] =   J[ 0][5]*Depl[ 0]+/*J[ 1][5]*Depl[ 1]*/ J[ 2][5]*Depl[ 2]+
                   J[ 3][5]*Depl[ 3]+/*J[ 4][5]*Depl[ 4]*/ J[ 5][5]*Depl[ 5]+
                   J[ 6][5]*Depl[ 6]+/*J[ 7][5]*Depl[ 7]*/ J[ 8][5]*Depl[ 8]+
                   J[ 9][5]*Depl[ 9]+/*J[10][5]*Depl[10]*/ J[11][5]*Depl[11];
    }

    static void computePlasticJtD(VoigtTensor& JtD, VoigtTensor& plasticStrain, const PlasticityParameters& plasticity)
    {
        VoigtTensor elasticStrain = JtD; // JtD is the total strain
        elasticStrain -= plasticStrain; // totalStrain = elasticStrain + plasticStrain

        if( elasticStrain.norm2() > plasticity.yieldThreshold * plasticity.yieldThreshold )
        {
            plasticStrain += plasticity.creep * elasticStrain;
        }

        const Real plasticStrainNorm2 = plasticStrain.norm2();
        if( plasticStrainNorm2 > plasticity.maxThreshold * plasticity.maxThreshold )
        {
            plasticStrain *= plasticity.maxThreshold / helper::rsqrt( plasticStrainNorm2 );
        }

        // remaining elasticStrain = totatStrain - plasticStrain
        JtD -= plasticStrain;
    }

    static void computeKJtD(VoigtTensor& KJtD, const MaterialStiffness& K, const VoigtTensor& JtD)
    {
        KJtD[0] =   K[0][0]*JtD[0]+  K[0][1]*JtD[1]+  K[0][2]*JtD[2]
                  /*K[0][3]*JtD[3]+  K[0][4]*JtD[4]+  K[0][5]*JtD[5]*/;
        KJtD[1] =   K[1][0]*JtD[0]+  K[1][1]*JtD[1]+  K[1][2]*JtD[2]
                  /*K[1][3]*JtD[3]+  K[1][4]*JtD[4]+  K[1][5]*JtD[5]*/;
        KJtD[2] =   K[2][0]*JtD[0]+  K[2][1]*JtD[1]+  K[2][2]*JtD[2]
                  /*K[2][3]*JtD[3]+  K[2][4]*JtD[4]+  K[2][5]*JtD[5]*/;
        KJtD[3] = /*K[3][0]*JtD[0]+  K[3][1]*JtD[1]+  K[3][2]*JtD[2]+*/
                    K[3][3]*JtD[3] /*K[3][4]*JtD[4]+  K[3][5]*JtD[5]*/;
        KJtD[4] = /*K[4][0]*JtD[0]+  K[4][1]*JtD[1]+  K[4][2]*JtD[2]+*/
                  /*K[4][3]*JtD[3]+*/K[4][4]*JtD[4] /*K[4][5]*JtD[5]*/;
        KJtD[5] = /*K[5][0]*JtD[0]+  K[5][1]*JtD[1]+  K[5][2]*JtD[2]+*/
                  /*K[5][3]*JtD[3]+  K[5][4]*JtD[4]+*/K[5][5]*JtD[5]  ;
    }

    static void computeJKJtD(Force& F, const StrainDisplacement &J, const VoigtTensor& KJtD)
    {
        F[ 0] =   J[ 0][0]*KJtD[0]+/*J[ 0][1]*KJtD[1]+  J[ 0][2]*KJtD[2]+*/
                  J[ 0][3]*KJtD[3]+/*J[ 0][4]*KJtD[4]+*/J[ 0][5]*KJtD[5]  ;
        F[ 1] = /*J[ 1][0]*KJtD[0]+*/J[ 1][1]*KJtD[1]+/*J[ 1][2]*KJtD[2]+*/
                  J[ 1][3]*KJtD[3]+  J[ 1][4]*KJtD[4] /*J[ 1][5]*KJtD[5]*/;
        F[ 2] = /*J[ 2][0]*KJtD[0]+  J[ 2][1]*KJtD[1]+*/J[ 2][2]*KJtD[2]+
                /*J[ 2][3]*KJtD[3]+*/J[ 2][4]*KJtD[4]+  J[ 2][5]*KJtD[5]  ;
        F[ 3] =   J[ 3][0]*KJtD[0]+/*J[ 3][1]*KJtD[1]+  J[ 3][2]*KJtD[2]+*/
                  J[ 3][3]*KJtD[3]+/*J[ 3][4]*KJtD[4]+*/J[ 3][5]*KJtD[5]  ;
        F[ 4] = /*J[ 4][0]*KJtD[0]+*/J[ 4][1]*KJtD[1]+/*J[ 4][2]*KJtD[2]+*/
                  J[ 4][3]*KJtD[3]+  J[ 4][4]*KJtD[4] /*J[ 4][5]*KJtD[5]*/;
        F[ 5] = /*J[ 5][0]*KJtD[0]+  J[ 5][1]*KJtD[1]+*/J[ 5][2]*KJtD[2]+
                /*J[ 5][3]*KJtD[3]+*/J[ 5][4]*KJtD[4]+  J[ 5][5]*KJtD[5]  ;
        F[ 6] =   J[ 6][0]*KJtD[0]+/*J[ 6][1]*KJtD[1]+  J[ 6][2]*KJtD[2]+*/
                  J[ 6][3]*KJtD[3]+/*J[ 6][4]*KJtD[4]+*/J[ 6][5]*KJtD[5]  ;
        F[ 7] = /*J[ 7][0]*KJtD[0]+*/J[ 7][1]*KJtD[1]+/*J[ 7][2]*KJtD[2]+*/
                  J[ 7][3]*KJtD[3]+  J[ 7][4]*KJtD[4] /*J[ 7][5]*KJtD[5]*/;
        F[ 8] = /*J[ 8][0]*KJtD[0]+  J[ 8][1]*KJtD[1]+*/J[ 8][2]*KJtD[2]+
                /*J[ 8][3]*KJtD[3]+*/J[ 8][4]*KJtD[4]+  J[ 8][5]*KJtD[5]  ;
        F[ 9] =   J[ 9][0]*KJtD[0]+/*J[ 9][1]*KJtD[1]+  J[ 9][2]*KJtD[2]+*/
                  J[ 9][3]*KJtD[3]+/*J[ 9][4]*KJtD[4]+*/J[ 9][5]*KJtD[5]  ;
        F[10] = /*J[10][0]*KJtD[0]+*/J[10][1]*KJtD[1]+/*J[10][2]*KJtD[2]+*/
                  J[10][3]*KJtD[3]+  J[10][4]*KJtD[4] /*J[10][5]*KJtD[5]*/;
        F[11] = /*J[11][0]*KJtD[0]+  J[11][1]*KJtD[1]+*/J[11][2]*KJtD[2]+
                /*J[11][3]*KJtD[3]+*/J[11][4]*KJtD[4]+  J[11][5]*KJtD[5]  ;
    }

    /// Compute the force as the following product : (J * (K * (J^T * Depl)))
    /// Note:Although (JKJ^T) is constant, it is faster to compute (J * (K * (J^T * Depl)))
    /// than (A * Depl) where A = (JKJ^T) and has been pre-computed
    static void computeForce(
        Force& F,
        const Displacement &Depl,
        VoigtTensor &plasticStrain,
        const MaterialStiffness &K,
        const StrainDisplacement &J,
        const PlasticityParameters& plasticity)
    {
        VoigtTensor JtD;
        computeElasticJtD(JtD, J, Depl);

        // eventually remove a part of the strain to simulate plasticity
        if( plasticity.hasPlasticity() )
        {
            computePlasticJtD(JtD, plasticStrain, plasticity);
        }

        VoigtTensor KJtD;
        computeKJtD(KJtD, K, JtD);
        computeJKJtD(F, J, KJtD);
    }

    static void computeForce(
        Force& F,
        const Displacement &Depl,
        const MaterialStiffness &K,
        const StrainDisplacement &J)
    {
        VoigtTensor JtD;
        computeElasticJtD(JtD, J, Depl);

        VoigtTensor KJtD;
        computeKJtD(KJtD, K, JtD);
        computeJKJtD(F, J, KJtD);
    }

    virtual StiffnessMatrix computeStiffnessMatrix(const MaterialStiffness &K, const StrainDisplacement &J, const unsigned elementIndex) const
    {
        SOFA_UNUSED(elementIndex);
        const auto Jt = J.transposed();
        return J * K * Jt;
    }

    StiffnessMatrix assembleStiffnessMatrix(const FiniteElementArrays& finiteElementArrays, const Element& element,
        const unsigned elementIndex)
    {
        const auto JKJt = computeStiffnessMatrix(
            (*finiteElementArrays.materialsStiffnesses)[elementIndex],
            (*finiteElementArrays.strainDisplacements)[elementIndex],
            elementIndex);

        for(int i = 0; i < 12; ++i)
        {
            const Index row = element[i/3] * 3 + i % 3;

            for(int j = 0; j < 12; ++j)
            {
                if(JKJt[i][j] != 0)
                {
                    const Index col = element[j/3] * 3 + j % 3;
                    // search if the vertex is already taken into account by another element
                    auto& stiffnessRow = (*finiteElementArrays.stiffnesses)[row];
                    auto result = std::find_if(stiffnessRow.begin(), stiffnessRow.end(),
                                               [col](const auto& e){ return e.first == col;});

                    if( result == stiffnessRow.end() )
                        stiffnessRow.emplace_back( col, JKJt[i][j] );
                    else
                        result->second += JKJt[i][j];
                }
            }
        }

        return JKJt;
    }
};

}
