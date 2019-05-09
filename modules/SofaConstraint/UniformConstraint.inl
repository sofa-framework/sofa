#ifndef SOFA_CONSTRAINT_UNIFORMCONSTRAINT_INL
#define SOFA_CONSTRAINT_UNIFORMCONSTRAINT_INL

#include "UniformConstraint.h"
#include <sofa/core/behavior/Constraint.inl>
#include <sofa/core/objectmodel/Data.h>
#include <SofaConstraint/BilateralConstraintResolution.h>

namespace sofa
{
namespace constraint
{

template< class DataTypes >
UniformConstraint<DataTypes>::UniformConstraint()
    :d_iterative(initData(&d_iterative, true, "iterative", "Iterate over the bilateral constraints, otherwise a block factorisation is computed."))
    ,d_constraintRestPos(initData(&d_constraintRestPos, false, "constrainToRestPos", "if false, constrains the pos to be zero / if true constraint the current position to stay at rest position"))
    ,m_constraintIndex(0)
{

}

template< class DataTypes >
void UniformConstraint<DataTypes>::buildConstraintMatrix(const sofa::core::ConstraintParams* cParams, DataMatrixDeriv & c, unsigned int &cIndex, const DataVecCoord &x)
{
    const std::size_t N = Deriv::size(); // MatrixDeriv is a container of Deriv types.

    auto& jacobian = sofa::helper::write(c, cParams).wref();
    auto  xVec     = sofa::helper::read(x, cParams);

    m_constraintIndex = cIndex; // we should not have to remember this, it should be available through the API directly.

    for (std::size_t i = 0; i < xVec.size(); ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            auto row = jacobian.writeLine(N*i + j + m_constraintIndex);
            Deriv d;
            d[j] = Real(1);
            row.setCol(i, d);
            ++cIndex;
        }
    }
}

template<class DstV, class Free>
void computeViolation(DstV& resV, unsigned int constraintIndex, const
                      Free& free, size_t N, std::function<double(int i, int j)> f)
{
    for (std::size_t i = 0; i < free.size(); ++i)
    {
        for (std::size_t j = 0; j < N; ++j)
        {
            resV->set(constraintIndex + i*N + j, f(i, j) );
        }
    }
}

template< class DataTypes >
void UniformConstraint<DataTypes>::getConstraintViolation(const sofa::core::ConstraintParams* cParams, sofa::defaulttype::BaseVector *resV, const DataVecCoord &x, const DataVecDeriv &v)
{
    auto xfree = sofa::helper::read(x, cParams);
    auto vfree = sofa::helper::read(v, cParams);
    const SReal dt = this->getContext()->getDt();
    const SReal invDt = 1.0 / dt;

    auto pos     = this->getMState()->readPositions();
    auto restPos = this->getMState()->readRestPositions();

    if (cParams->constOrder() == sofa::core::ConstraintParams::VEL)
    {
        if (d_constraintRestPos.getValue()){
            computeViolation(resV, m_constraintIndex, vfree, Deriv::size(),[&invDt,&pos,&vfree,&restPos](int i, int j)
            { return vfree[i][j] + invDt *(pos[i][j]-restPos[i][j]); });
        }
        else {
            computeViolation(resV, m_constraintIndex, vfree, Deriv::size(),[&invDt,&pos,&vfree](int i, int j)
            { return vfree[i][j] + invDt *pos[i][j]; });
        }
    }
    else
    {
        if( d_constraintRestPos.getValue() )
            computeViolation(resV, m_constraintIndex, xfree, Coord::size(),
                             [&xfree,&restPos](int i, int j){ return xfree[i][j] - restPos[i][j]; });
        else
            computeViolation(resV, m_constraintIndex, xfree, Coord::size(),[&xfree](int i, int j){ return xfree[i][j]; });
    }
}

template< class DataTypes >
void UniformConstraint<DataTypes>::getConstraintResolution(const sofa::core::ConstraintParams* cParams, std::vector<sofa::core::behavior::ConstraintResolution*>& crVector, unsigned int& offset)
{

    if (d_iterative.getValue(cParams))
    {
        for (std::size_t i = 0; i < this->getMState()->getSize(); ++i)
        {
            for (std::size_t j = 0; j < Deriv::size(); ++j)
            {
                sofa::component::constraintset::BilateralConstraintResolution* cr = new sofa::component::constraintset::BilateralConstraintResolution();
                crVector[offset++] = cr;
            }
        }
    }
    else
    {
        const std::size_t nbLines = this->getMState()->getSize() * Deriv::size();
        sofa::component::constraintset::BilateralConstraintResolutionNDof* cr = new sofa::component::constraintset::BilateralConstraintResolutionNDof(nbLines);
        crVector[offset] = cr;
        offset += nbLines;
    }
}

}
}

#endif // SOFA_CONSTRAINT_UNIFORMCONSTRAINT_INL
