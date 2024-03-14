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

#include <sofa/core/behavior/ForceField.inl>
#include <sofa/component/solidmechanics/spring/PolynomialRestShapeSpringsForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>


namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
PolynomialRestShapeSpringsForceField<DataTypes>::PolynomialRestShapeSpringsForceField()
    : d_points(initData(&d_points, "points", "points controlled by the rest shape springs"))
    , d_external_points(initData(&d_external_points, "external_points", "points from the external Mechancial State that define the rest shape springs"))
    , d_polynomialStiffness(initData(&d_polynomialStiffness, "polynomialStiffness", "coefficients for all spring polynomials"))
    , d_polynomialDegree(initData(&d_polynomialDegree, "polynomialDegree", "vector of values that show polynomials degrees"))
    , d_recomputeIndices(initData(&d_recomputeIndices, false, "recompute_indices", "Recompute indices (should be false for BBOX)"))
    , d_drawSpring(initData(&d_drawSpring,false,"drawSpring","draw Spring"))
    , d_springColor(initData(&d_springColor, sofa::type::RGBAColor(0.0f, 1.0f, 0.0f, 1.0f), "springColor","spring color"))
    , d_showIndicesScale(initData(&d_showIndicesScale, (float)0.02, "showIndicesScale", "Scale for indices display. (default=0.02)"))
    , d_zeroLength(initData(&d_zeroLength,"initialLength","initial virtual length of the spring"))
    , d_smoothShift(initData(&d_smoothShift,static_cast<Real>(0.0),"smoothShift","denominator correction adding shift value"))
    , d_smoothScale(initData(&d_smoothScale,static_cast<Real>(1.0),"smoothScale","denominator correction adding scale"))
    , d_restMState(initLink("external_rest_shape", "rest_shape can be defined by the position of an external Mechanical State"))
{        
}


template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::bwdInit()
{
    core::behavior::ForceField<DataTypes>::init();

    if (d_polynomialStiffness.getValue().empty())
    {
        msg_info() << "ExtendedRestShapeSpringForceField : No stiffness is defined, assuming equal stiffness on each node, k = 100.0";

        VecReal stiffs;
        stiffs.push_back(100.0);
        d_polynomialStiffness.setValue(stiffs);
    }

    if (d_zeroLength.getValue().empty())
    {
        VecReal dist;
        dist.push_back(1.0);
        d_zeroLength.setValue(dist);
    }

    if (d_restMState.get() == nullptr)
    {
        m_useRestMState = false;
        msg_info() << "no external rest shape used";

        if (!d_restMState.empty())
        {
            msg_warning() << "external_rest_shape in node " << this->getContext()->getName() << " not found";
        }
    }
    else
    {
        msg_info() << "external rest shape used";
        m_useRestMState = true;
    }

    if (m_useRestMState)
    {
        msg_info() << "[" << this->getName() << "]: using the external state " << d_restMState->getName();
    }
    else
    {
        msg_info() << "[" << this->getName() << "]: using the rest state " << this->mstate->getName();
    }

    recomputeIndices();

    const core::behavior::BaseMechanicalState* state = this->getContext()->getMechanicalState();
    if(!state)
    {
        msg_warning() << "MechanicalState of the current context returns null pointer";
    }

    // read and fill polynomial parameters
    if (d_polynomialDegree.getValue().empty()) {
        helper::WriteAccessor<Data<type::vector<unsigned int>>> vPolynomialWriteDegree = d_polynomialDegree;
        vPolynomialWriteDegree.push_back(1);
    }

    const helper::ReadAccessor<Data<type::vector<unsigned int>>> vPolynomialDegree = d_polynomialDegree;

    m_polynomialsMap.clear();
    type::vector<unsigned int> polynomial;
    sofa::Index inputIndex = 0;
    for (sofa::Index degreeIndex = 0; degreeIndex < vPolynomialDegree.size(); degreeIndex++) {
        polynomial.clear();
        polynomial.resize(vPolynomialDegree[degreeIndex]);
        for (sofa::Index polynomialIndex = 0; polynomialIndex < vPolynomialDegree[degreeIndex]; polynomialIndex++) {
            polynomial[polynomialIndex] = inputIndex;
            inputIndex++;
        }
        m_polynomialsMap.push_back(polynomial);
    }

    msg_info() << "Polynomial data: ";
    for (sofa::Index degreeIndex = 0; degreeIndex < vPolynomialDegree.size(); degreeIndex++) {
        for (sofa::Index polynomialIndex = 0; polynomialIndex < vPolynomialDegree[degreeIndex]; polynomialIndex++) {
            msg_info() << m_polynomialsMap[degreeIndex][polynomialIndex] << " ";
        }
    }

    this->f_listening.setValue(true);

    /// recreate derivatives matrices
    m_differential.resize(m_indices.size());

    m_directionSpringLength.resize(m_indices.size());
    m_strainValue.resize(m_indices.size());
    m_weightedCoordinateDifference.resize(m_indices.size());
    m_coordinateSquaredNorm.resize(m_indices.size());
}


template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::recomputeIndices()
{
    m_indices.clear();
    m_ext_indices.clear();

    for (sofa::Index i = 0; i < d_points.getValue().size(); i++)
        m_indices.push_back(d_points.getValue()[i]);

    for (sofa::Index i = 0; i < d_external_points.getValue().size(); i++)
        m_ext_indices.push_back(d_external_points.getValue()[i]);

    if (m_indices.empty())
    {
        msg_info() << "in PolynomialRestShapeSpringForceField no point are defined, default case: points = all points ";
        for (sofa::Index i = 0; i < this->mstate->getSize(); i++) {
            m_indices.push_back(i);
        }
    }

    if (m_ext_indices.empty())
    {
        msg_info() << "in PolynomialRestShapeSpringForceField no external_points are defined, default case: points = all points ";

        if (m_useRestMState)
        {
            if (const DataVecCoord* extPosition = getExtPosition())
            {
                const auto& extPositionValue = extPosition->getValue();
                for (sofa::Index i = 0; i < extPositionValue.size(); i++)
                {
                    m_ext_indices.push_back(i);
                }
            }
            else
            {
                this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            }
        }
        else
        {
            for (sofa::Index i = 0; i < this->mstate->getSize(); i++)
            {
                m_ext_indices.push_back(i);
            }
        }
    }

    if (m_indices.size() > m_ext_indices.size())
    {
        msg_error() << "Error : the dimension of the source and the targeted points are different ";
        m_indices.clear();
        m_ext_indices.clear();
    }
}


template<class DataTypes>
const typename PolynomialRestShapeSpringsForceField<DataTypes>::DataVecCoord* PolynomialRestShapeSpringsForceField<DataTypes>::getExtPosition() const
{
    if(m_useRestMState)
    {
        if (d_restMState)
        {
            return d_restMState->read(core::VecCoordId::position());
        }
    }
    else
    {
        if (this->mstate)
        {
            return this->mstate->read(core::VecCoordId::restPosition());
        }
    }
    return nullptr;
}


template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& f,
                                                               const DataVecCoord& x, const DataVecDeriv& v)
{
    SOFA_UNUSED(mparams);
    SOFA_UNUSED(v);

    helper::WriteAccessor<DataVecDeriv> f1 = f;
    helper::ReadAccessor<DataVecCoord> p1 = x;

    const DataVecCoord* extPosition = getExtPosition();
    if (!extPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    helper::ReadAccessor< DataVecCoord > p0 = *extPosition;

    msg_info() << "P1 = " << p1.ref();
    msg_info() << "P0 = " << p0.ref();
    msg_info() << "F = " << f1.ref();

    const VecReal& zeroLength = d_zeroLength.getValue();
    f1.resize(p1.size());

    if (d_recomputeIndices.getValue())
    {
        recomputeIndices();
    }

    msg_info() << "\n\nNew step:\n";
    if ( d_polynomialDegree.getValue().size() != m_indices.size() )
    {
        msg_warning() << "WARNING : stiffness is not defined on each point, first stiffness is used";
        for (unsigned int i = 0; i < m_indices.size(); i++)
        {
            const sofa::Index index = m_indices[i];
            sofa::Index ext_index = m_indices[i];
            if(m_useRestMState)
                ext_index= m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            m_coordinateSquaredNorm[i] = dot(dx, dx);
            msg_info() << "dx value: " << dx;

            // to compute stress value use original spring length
            Real springLength = dx.norm();
            msg_info() << "Spring length value: " << springLength;
            m_strainValue[i] = springLength / (i < zeroLength.size() ? zeroLength[i] : zeroLength[0]);
            Real forceValue = PolynomialValue(0, m_strainValue[i]);
            msg_info() << "Strain value: " << m_strainValue[i];
            msg_info() << "Force value: " << forceValue;

            // to compute direction use the modified length denominator: dx^2 + exp^(shift - scale * dx^2)
            Real squaredDenominator = dot(dx, dx);
            squaredDenominator += exp(d_smoothShift.getValue() - d_smoothScale.getValue() * dot(dx, dx));
            msg_info() << "Denominator value: " << std::sqrt(squaredDenominator);

            // compute the length with modified values
            m_directionSpringLength[i] = std::sqrt(squaredDenominator);
            m_weightedCoordinateDifference[i] = dx;
            m_weightedCoordinateDifference[i] = m_weightedCoordinateDifference[i] / m_directionSpringLength[i];

            f1[index] -=  forceValue * m_weightedCoordinateDifference[i];
            msg_info() << "Applied force value: " << -forceValue * m_weightedCoordinateDifference[i];

            ComputeJacobian(0, i);
        }
    }
    else
    {
        for (sofa::Index i = 0; i < m_indices.size(); i++)
        {
            const sofa::Index index = m_indices[i];
            sofa::Index ext_index = m_indices[i];
            if(m_useRestMState)
                ext_index= m_ext_indices[i];

            Deriv dx = p1[index] - p0[ext_index];
            m_coordinateSquaredNorm[i] = dot(dx, dx);
            msg_info() << "dx value: " << dx;

            // to compute stress value use original spring length
            Real springLength = dx.norm();
            msg_info() << "Spring length value: " << springLength;
            m_strainValue[i] = springLength / (i < zeroLength.size() ? zeroLength[i] : zeroLength[0]);
            Real forceValue = PolynomialValue(i, m_strainValue[i]);
            msg_info() << "Strain value: " << m_strainValue[i];
            msg_info() << "Force value: " << forceValue;

            // to compute direction use the modified length denominator: dx^2 + exp^(shift - scale * dx^2)
            Real squaredDenominator = dot(dx, dx);
            squaredDenominator += exp(d_smoothShift.getValue() - d_smoothScale.getValue() * dot(dx, dx));
            msg_info() << "Denominator value: " << std::sqrt(squaredDenominator);

            // compute the length with modified values
            m_directionSpringLength[i] = std::sqrt(squaredDenominator);
            m_weightedCoordinateDifference[i] = dx;
            m_weightedCoordinateDifference[i] = m_weightedCoordinateDifference[i] / m_directionSpringLength[i];

            f1[index] -= forceValue * m_weightedCoordinateDifference[i];
            msg_info() << "Applied force value: " << -forceValue * m_weightedCoordinateDifference[i];

            ComputeJacobian(i, i);
        }
    }
}



template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::ComputeJacobian(sofa::Index stiffnessIndex, sofa::Index springIndex)
{
    msg_info() << "\nCompute derivative: ";
    msg_info() << "spring length: " << m_directionSpringLength[springIndex];
    // Compute stiffness dF/dX for nonlinear case

    msg_info() << "weighted difference: " << m_weightedCoordinateDifference[springIndex][0] << " " <<
                   m_weightedCoordinateDifference[springIndex][1] << " " << m_weightedCoordinateDifference[springIndex][2];

    const VecReal& zeroLength = d_zeroLength.getValue();

    // NOTE: Since we compute derivatives only for one of mechanical models we take only diagonal of jacobian matrix

    // compute polynomial result
    Real polynomialForceRes = PolynomialValue(stiffnessIndex, m_strainValue[springIndex]) / m_directionSpringLength[springIndex];
    msg_info() << "PolynomialForceRes: " << polynomialForceRes;

    Real polynomialDerivativeRes = PolynomialDerivativeValue(stiffnessIndex, m_strainValue[springIndex]) /
            (springIndex < zeroLength.size() ? zeroLength[springIndex] : zeroLength[0]);
    msg_info() << "PolynomialDerivativeRes: " << polynomialDerivativeRes;

    const Real exponentialDerivative = 1.0 - d_smoothScale.getValue() *
            exp(d_smoothShift.getValue() - d_smoothScale.getValue() * m_coordinateSquaredNorm[springIndex]);

    // compute data for Jacobian matrix
    JacobianVector& jacobVector = m_differential[springIndex];
    for(sofa::Index index = 0; index < Coord::total_size; index++)
    {
        jacobVector[index] = (polynomialDerivativeRes - polynomialForceRes) * exponentialDerivative *
                    m_weightedCoordinateDifference[springIndex][index] * m_weightedCoordinateDifference[springIndex][index] + polynomialForceRes;
    }

    for(sofa::Index index = 0; index < Coord::total_size; index++)
    {
        msg_info() << "for indices " << index << " the values is: " << jacobVector[index];
    }
}



template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams,
                                                                DataVecDeriv& df, const DataVecDeriv& dx)
{
    msg_info() << "[" <<  this->getName() << "]: addDforce";

    helper::WriteAccessor< DataVecDeriv > df1 = df;
    helper::ReadAccessor< DataVecDeriv > dx1 = dx;
    Real kFactor = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    for (sofa::Index index = 0; index < m_indices.size(); index++)
    {
        const JacobianVector& jacobVector = m_differential[index];
        for(sofa::Index coordIndex = 0; coordIndex < Coord::total_size; coordIndex++)
        {
            df1[m_indices[index]][coordIndex] -= jacobVector[coordIndex] * dx1[m_indices[index]][coordIndex] * kFactor;
        }
    }
}



template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields() || !d_drawSpring.getValue())
        return;

    const DataVecCoord* extPosition = getExtPosition();
    if (!extPosition)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    helper::ReadAccessor< DataVecCoord > p0 = *extPosition;
    helper::ReadAccessor< DataVecCoord > p  = this->mstate->read(core::VecCoordId::position());

    const VecIndex& indices = m_indices;
    const VecIndex& ext_indices = (m_useRestMState ? m_ext_indices : m_indices);

    std::vector< type::Vec3 > points;

    for (sofa::Index i=0; i<indices.size(); i++)
    {
        const sofa::Index index = indices[i];
        const sofa::Index ext_index = ext_indices[i];
        points.push_back(p[index]);
        points.push_back(p0[ext_index]);
    }

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->setLightingEnabled(false);

    vparams->drawTool()->drawLines(points, 5, d_springColor.getValue());



    // draw connected point indices
    Real scale = (vparams->sceneBBox().maxBBox() - vparams->sceneBBox().minBBox()).norm() * d_showIndicesScale.getValue();

    type::vector<type::Vec3> positions;
    for (sofa::Index i = 0; i < indices.size(); i++) {
        const sofa::Index index = indices[i];
        positions.push_back(type::Vec3(p0[index][0], p0[index][1], p0[index][2] ));
    }

    vparams->drawTool()->draw3DText_Indices(positions, float(scale), type::RGBAColor::white());

}



template<class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams,
                                                                   const sofa::core::behavior::MultiMatrixAccessor* matrix )
{    
    msg_info() << "[" <<  this->getName() << "]: addKToMatrix";

    SCOPED_TIMER("restShapePolynomialSpringAddKToMatrix");

    const sofa::core::behavior::MultiMatrixAccessor::MatrixRef mref = matrix->getMatrix(this->mstate);
    sofa::linearalgebra::BaseMatrix* mat = mref.matrix;
    const unsigned int offset = mref.offset;
    Real kFact = (Real)sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue());

    sofa::Index curIndex = 0;
    const int Dimension = Coord::total_size;

    for (sofa::Index index = 0; index < m_indices.size(); index++)
    {
        curIndex = m_indices[index];
        const JacobianVector& jacobVector = m_differential[index];
        for(unsigned int i = 0; i < Dimension; i++)
        {
            mat->add(offset + Dimension * curIndex + i, offset + Dimension * curIndex + i, -kFact * jacobVector[i]);
        }
    }
}

template <class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix)
{
    static constexpr sofa::SignedIndex Dimension = Coord::total_size;

    auto dfdx = matrix->getForceDerivativeIn(this->mstate)
                       .withRespectToPositionsIn(this->mstate);

    for (std::size_t index = 0; index < m_indices.size(); ++index)
    {
        const sofa::Index curIndex = m_indices[index];
        const JacobianVector& jacobVector = m_differential[index];
        for(sofa::SignedIndex i = 0; i < Dimension; ++i)
        {
            dfdx(Dimension * curIndex + i, Dimension * curIndex + i) += -jacobVector[i];
        }
    }
}

template <class DataTypes>
void PolynomialRestShapeSpringsForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
typename PolynomialRestShapeSpringsForceField<DataTypes>::Real
PolynomialRestShapeSpringsForceField<DataTypes>::PolynomialValue(unsigned int springIndex, Real strainValue)
{
    helper::ReadAccessor<Data<VecReal>> vPolynomialStiffness = d_polynomialStiffness;
    const helper::ReadAccessor<Data<type::vector<unsigned int> >> vPolynomialDegree = d_polynomialDegree;

    msg_info() << "Polynomial data: ";
    Real highOrderStrain = 1.0;
    Real result = 0.0;
    for (sofa::Index degreeIndex = 0; degreeIndex < vPolynomialDegree[springIndex]; degreeIndex++)
    {
        highOrderStrain *= strainValue;
        result += vPolynomialStiffness[m_polynomialsMap[springIndex][degreeIndex]] * highOrderStrain;
        msg_info() << "Degree:" << (degreeIndex + 1) << ", result: " << result;
    }

    return result;
}


template<class DataTypes>
typename PolynomialRestShapeSpringsForceField<DataTypes>::Real
PolynomialRestShapeSpringsForceField<DataTypes>::PolynomialDerivativeValue(unsigned int springIndex, Real strainValue)
{
    helper::ReadAccessor<Data<VecReal>> vPolynomialStiffness = d_polynomialStiffness;
    const helper::ReadAccessor<Data<type::vector<unsigned int> >> vPolynomialDegree = d_polynomialDegree;

    msg_info() << "Polynomial derivative data: ";
    Real highOrderStrain = 1.0;
    Real result = 0.0;
    for (sofa::Index degreeIndex = 0; degreeIndex < vPolynomialDegree[springIndex]; degreeIndex++)
    {
        result += (degreeIndex + 1) * vPolynomialStiffness[m_polynomialsMap[springIndex][degreeIndex]] * highOrderStrain;
        highOrderStrain *= strainValue;
        msg_info() << "Degree:" << (degreeIndex + 1) << ", result: " << result;
    }

    return result;
}


} // namespace sofa::component::solidmechanics::spring
