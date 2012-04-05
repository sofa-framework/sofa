#ifndef SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#include "Compliance.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/linearsolver/EigenSparseSquareMatrix.h>

namespace sofa
{
namespace component
{
namespace compliance
{

/** Compliance uniformly applied to all the DOF.
  Each dof represents a constraint violation, and undergoes force \f$ \lambda = -\frac{1}{c} ( x - d v ) \f$, where c is the compliance and d the damping ratio.
  */
template<class TDataTypes>
class SOFA_Compliant_API UniformCompliance : public core::behavior::ForceField<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformCompliance, TDataTypes), SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

    typedef TDataTypes DataTypes;
    typedef core::behavior::ForceField<TDataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    enum { N=DataTypes::deriv_total_size };
    typedef defaulttype::Mat<N,N,Real> Block;

    Data< Block > compliance;   ///< Same compliance applied to all the DOFs
    Data< SReal > dampingRatio; ///< Same damping ratio applied to all the DOFs

    virtual SReal getDampingRatio() { return dampingRatio.getValue(); }


    virtual void init();

    /// Compute the compliance matrix
    virtual void reinit();

    /// Set the constraint value
    virtual void writeConstraintValue(const core::MechanicalParams* mparams, core::MultiVecDerivId fId );

    /// return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*);

    /// set a uniform, diagonal compliance
    void setCompliance( Real c );

    /// addForce does nothing because this component is processed like a compliance.
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &) {}

protected:
    UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    linearsolver::EigenSparseSquareMatrix<typename DataTypes::Real> matC; ///< compliance matrix
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H


