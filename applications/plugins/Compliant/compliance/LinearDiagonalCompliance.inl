#include "LinearDiagonalCompliance.h"

namespace sofa
{
namespace component
{
namespace forcefield
{

template<class DataTypes>
LinearDiagonalCompliance<DataTypes>::LinearDiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , d_complianceMin(initData(&d_complianceMin, "complianceMin", "Minimum compliance"))
    , d_errorMin(initData(&d_errorMin, "errorMin", "complianceMin is reached for this error value"))
    , m_lastTime(-1.)
{}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::init()
{
    Real complianceMin = d_complianceMin.getValue();
    {
        WriteOnlyVecDeriv diag(this->diagonal);
        diag.resize(this->getContext()->getMechanicalState()->getSize());
        for (std::size_t i=0; i<diag.size(); ++i)
            diag[i].fill(complianceMin);
    }
    Inherit::init();
}


//template<class DataTypes>
//SReal LinearDiagonalCompliance<DataTypes>::getPotentialEnergy( const core::MechanicalParams* mparams, const typename Inherit::DataVecCoord& x ) const
//{

//}

template<class DataTypes>
const sofa::defaulttype::BaseMatrix* LinearDiagonalCompliance<DataTypes>::getComplianceMatrix(const core::MechanicalParams* mparam)
{
    updateDiagonalCompliance();
    return Inherit::getComplianceMatrix(mparam);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset )
{
    updateDiagonalCompliance();
    Inherit::addKToMatrix(matrix, kFact, offset);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset )
{
    updateDiagonalCompliance();
    Inherit::addBToMatrix(matrix, bFact, offset);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::addForce(const core::MechanicalParams * mparam, typename Inherit::DataVecDeriv& f, const typename Inherit::DataVecCoord& x, const typename Inherit::DataVecDeriv& v)
{
    updateDiagonalCompliance();
    Inherit::addForce(mparam, f, x, v);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::addDForce(const core::MechanicalParams *mparams, typename Inherit::DataVecDeriv& df,  const typename Inherit::DataVecDeriv& dx)
{
    updateDiagonalCompliance();
    Inherit::addDForce(mparams, df, dx);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::addClambda(const core::MechanicalParams *mparams, typename Inherit::DataVecDeriv &res, const typename Inherit::DataVecDeriv &lambda, SReal cfactor)
{
    updateDiagonalCompliance();
    Inherit::addClambda(mparams, res, lambda, cfactor);
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::updateDiagonalCompliance()
{
    if (m_lastTime != this->getContext()->getTime()) {
        computeDiagonalCompliance();
        Inherit::reinit();
        m_lastTime = this->getContext()->getTime();
    }
}

template<class DataTypes>
void LinearDiagonalCompliance<DataTypes>::computeDiagonalCompliance()
{
    {
        typename sofa::core::State<DataTypes>::ReadVecCoord errors = this->getMState()->readPositions();
        WriteOnlyVecDeriv diag(this->diagonal);
        diag.resize(errors.size());

        Real errorMin = d_errorMin.getValue();
        Real complianceMin = d_complianceMin.getValue();

        for (std::size_t i=0; i<errors.size(); ++i) {
            Real errorNorm = errors[i].norm();
            if (errorNorm < errorMin)
                diag[i].fill(complianceMin);
            else
                diag[i].fill(complianceMin * errorNorm/errorMin);
        }
    }
}

}
}
}
