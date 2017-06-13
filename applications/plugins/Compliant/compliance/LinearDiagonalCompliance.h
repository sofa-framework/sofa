#ifndef SOFA_COMPONENT_COMPLIANCE_LinearDiagonalCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_LinearDiagonalCompliance_H

#include <Compliant/config.h>
#include "DiagonalCompliance.h"

namespace sofa
{
namespace component
{
namespace forcefield
{

/** A compliance defined by a simple piece wise linear function.
 * c = d_complianceMin if error < errorMin
 *     d_complianceMin * error / errorMin if error > errorMin
 *
 * \todo more complex function if needed
 * \todo non uniform compliance on dof Coord
 * \author Thomas Lemaire \date 2016
 */
template<class TDataTypes>
class LinearDiagonalCompliance : public DiagonalCompliance<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearDiagonalCompliance, TDataTypes), SOFA_TEMPLATE(DiagonalCompliance, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef DiagonalCompliance<TDataTypes> Inherit;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef helper::WriteOnlyAccessor< Data< typename Inherit::VecDeriv > > WriteOnlyVecDeriv;

    Data< Real > d_complianceMin; ///< Minimum compliance
    Data< Real > d_errorMin; ///< complianceMin is reached for this error value

    virtual void init();

//    virtual SReal getPotentialEnergy( const core::MechanicalParams* mparams, const typename Inherit::DataVecCoord& x ) const;
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*);

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset );
    virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset );
    virtual void addForce(const core::MechanicalParams *, typename Inherit::DataVecDeriv &, const typename Inherit::DataVecCoord &, const typename Inherit::DataVecDeriv &);
    virtual void addDForce(const core::MechanicalParams *, typename Inherit::DataVecDeriv &, const typename Inherit::DataVecDeriv &);
    virtual void addClambda(const core::MechanicalParams *, typename Inherit::DataVecDeriv &, const typename Inherit::DataVecDeriv &, SReal);

protected:
    LinearDiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    void updateDiagonalCompliance();
    void computeDiagonalCompliance();

    SReal m_lastTime;

};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_LinearDiagonalCompliance_H
