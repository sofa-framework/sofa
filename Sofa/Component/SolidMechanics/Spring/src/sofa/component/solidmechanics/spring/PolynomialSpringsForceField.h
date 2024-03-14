/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2020 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/component/solidmechanics/spring/config.h>
#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::solidmechanics::spring
{

/**
* @brief This class describes a polynomial elastic springs ForceField
*
*/
template<class DataTypes>
class PolynomialSpringsForceField : public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PolynomialSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector<unsigned int> VecIndex;
    typedef type::vector<Real> VecReal;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;

    // connected objects indices
    Data<VecIndex> d_firstObjectPoints; ///< points related to the first object
    Data<VecIndex> d_secondObjectPoints; ///< points related to the second object

    // polynomial data
    /// Describe set of polynomial coefficients combines in one array.
    /// The coefficients are put from smaller degree to bigger one, and the free polynomial parameter is also zero
    /// (for zero strain we have zero stress).
    /// For examples the coeffiencts for polynomials with degrees [3, 2, 4] will be put as [ a1, a2, a3, b1, b2, c1, c2, c3, c4]
    Data<VecReal> d_polynomialStiffness;
    /// Describe set of polynomial degrees fro every spring
    Data< type::vector<unsigned int> > d_polynomialDegree;

    Data<int> d_computeZeroLength;                    ///< Flag to verify if initial length has to be computed during the first iteration
    Data<VecReal> d_zeroLength;                       ///< Springs initial lengths
    Data<bool> d_recomputeIndices; ///< Recompute indices (should be false for BBOX)

    Data <bool> d_compressible;                       ///< flag to put compressible springs

    Data<int> d_drawMode;                             ///< Draw Mode: 0=Line - 1=Cylinder - 2=Arrow
    Data<float> d_showArrowSize;                      ///< size of the axis
    Data<sofa::type::RGBAColor> d_springColor; ///< spring color
    Data<float> d_showIndicesScale; ///< Scale for indices display. (default=0.02)


    // data to compute spring derivatives
    typedef type::Mat<Coord::total_size, Coord::total_size, Real> JacobianMatrix;


protected:
    PolynomialSpringsForceField();
    PolynomialSpringsForceField(MechanicalState* object1, MechanicalState* object2);

    void recomputeIndices();

    VecIndex m_firstObjectIndices;
    VecIndex m_secondObjectIndices;

    type::vector<JacobianMatrix> m_differential;

    VecReal m_springLength;
    VecReal m_strainValue;
    VecCoord m_weightedCoordinateDifference;

    type::vector<type::vector<unsigned int>> m_polynomialsMap;

    VecReal m_initialSpringLength;
    VecReal m_strainSign;
    std::vector<int> m_computeSpringsZeroLength;


    const unsigned int m_dimension;
    static double constexpr MATH_PI = 3.14159265;

    void ComputeJacobian(unsigned int stiffnessIndex, unsigned int springIndex);
    double PolynomialValue(unsigned int springIndex, double strainValue);
    double PolynomialDerivativeValue(unsigned int springIndex, double strainValue);

public:
    void bwdInit() override;

    /// Add the forces.
    void addForce(const core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2,
                  const DataVecCoord& data_p1, const DataVecCoord& data_p2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2) override;

    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2,
                   const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;

    /// Brings ForceField contribution to the global system stiffness matrix.
    virtual void addKToMatrix(const core::MechanicalParams* mparams, const core::behavior::MultiMatrixAccessor* matrix) override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/,
                                     const DataVecCoord& /*x1*/, const DataVecCoord& /*x2*/) const override
    {
        msg_error() << "Get potentialEnergy not implemented";
        return 0.0;
    }

    void addMBKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override
    {
        sofa::core::BaseMapping *bmapping;
        this->getContext()->get(bmapping);
        if (bmapping ) /// do not call addKToMatrix since the object is mapped
            return;
        if (sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams,this->rayleighStiffness.getValue()) != 0.0 )
            this->addKToMatrix(mparams, matrix);
        if (sofa::core::mechanicalparams::bFactor(mparams) != 0.0)
            this->addBToMatrix(mparams, matrix);
    }

    const VecIndex& getFirstObjectIndices() const { return m_firstObjectIndices; }
    const VecIndex& getSecondObjectIndices() const { return m_secondObjectIndices; }

    core::behavior::MechanicalState<DataTypes>* getObject1() { return this->mstate1; }
    core::behavior::MechanicalState<DataTypes>* getObject2() { return this->mstate2; }
};


#if !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_POLYNOMIAL_SPRINGS_FORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API PolynomialSpringsForceField<defaulttype::Vec3Types>;

#endif // !defined(SOFA_COMPONENT_INTERACTIONFORCEFIELD_POLYNOMIAL_SPRINGS_FORCEFIELD_CPP)

} // namespace namespace sofa::component::solidmechanics::spring
