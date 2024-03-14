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
#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::core::behavior
{

template< class T > class MechanicalState;

} // namespace sofa::core::behavior

namespace sofa::component::solidmechanics::spring
{

/**
* @brief This class describes a polynomial elastic springs ForceField between DOFs positions and rest positions.
*
* Springs are applied to given degrees of freedom between their current positions and their rest shape positions.
* An external MechanicalState reference can also be passed to the ForceField as rest shape position.
*/
template<class DataTypes>
class PolynomialRestShapeSpringsForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PolynomialRestShapeSpringsForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef type::vector<sofa::Index> VecIndex;
    typedef type::vector<Real> VecReal;

    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;


    Data< type::vector<sofa::Index> > d_points; ///< points controlled by the rest shape springs
    Data< type::vector<sofa::Index> > d_external_points; ///< points from the external Mechancial State that define the rest shape springs

    /// polynomial data
    /// Describe set of polynomial coefficients combines in one array.
    /// The coefficients are put from smaller degree to bigger one, and the free polynomial parameter is also zero
    /// (for zero strain we have zero stress)
    /// For examples the coeffiencts for polynomials with degrees [3, 2, 4] will be put as [ a1, a2, a3, b1, b2, c1, c2, c3, c4]
    Data< VecReal > d_polynomialStiffness;
    /// Describe set of polynomial degrees fro every spring
    Data< type::vector<sofa::Size> > d_polynomialDegree;


    Data<bool> d_recomputeIndices; ///< Recompute indices (should be false for BBOX)
    Data<bool> d_drawSpring;                      ///< draw Spring
    Data<sofa::type::RGBAColor> d_springColor; ///< spring color
    Data<float> d_showIndicesScale; ///< Scale for indices display. (default=0.02)

    Data<VecReal> d_zeroLength;       ///< Springs initial lengths
    Data<Real> d_smoothShift; ///< denominator correction adding shift value
    Data<Real> d_smoothScale; ///< denominator correction adding scale

    SingleLink<PolynomialRestShapeSpringsForceField<DataTypes>, sofa::core::behavior::MechanicalState<DataTypes>,
        BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> d_restMState;

    // data to compute spring derivatives
    typedef type::Vec<Coord::total_size, Real> JacobianVector;


protected:
    PolynomialRestShapeSpringsForceField();

    void recomputeIndices();

    VecIndex m_indices;
    VecIndex m_ext_indices;

    type::vector<JacobianVector> m_differential;

    VecReal m_directionSpringLength;
    VecReal m_strainValue;
    VecCoord m_weightedCoordinateDifference;
    VecReal m_coordinateSquaredNorm;

    type::vector<type::vector<sofa::Size>> m_polynomialsMap;

    bool m_useRestMState; /// Indicator whether an external MechanicalState is used as rest reference.


    void ComputeJacobian(sofa::Index stiffnessIndex, sofa::Index springIndex);
    Real PolynomialValue(sofa::Index springIndex, Real strainValue);
    Real PolynomialDerivativeValue(sofa::Index springIndex, Real strainValue);

public:
    void bwdInit() override;

    /// Add the forces.
    virtual void addForce(const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx) override;

    /// Brings ForceField contribution to the global system stiffness matrix.
    virtual void addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix ) override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    virtual void draw(const core::visual::VisualParams* vparams) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord& /* x */) const override
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


    const DataVecCoord* getExtPosition() const;
    const VecIndex& getIndices() const { return m_indices; }
    const VecIndex& getExtIndices() const { return (m_useRestMState ? m_ext_indices : m_indices); }
};

#if !defined(SOFA_COMPONENT_FORCEFIELD_POLYNOMIAL_RESTSHAPESPRINGSFORCEFIELD_CPP)

extern template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API PolynomialRestShapeSpringsForceField< sofa::defaulttype::Vec3Types>;

#endif // !defined(SOFA_COMPONENT_FORCEFIELD_POLYNOMIAL_RESTSHAPESPRINGFORCEFIELD_CPP)

} // namespace sofa::component::solidmechanics::spring
