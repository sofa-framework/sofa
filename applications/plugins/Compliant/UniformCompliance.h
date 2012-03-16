#ifndef SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#include "Compliance.h"
#include <sofa/defaulttype/Mat.h>
#include <plugins/ModelHierarchies/EigenSparseSquareMatrix.h>

namespace sofa
{
namespace component
{
namespace compliance
{

/** Compliance uniformly applied to all the DOF.
  Each dof represents a constraint violation, and undergoes force \f$ \lambda = -\frac{1}{c} ( x - d v ) \f$, where c is the compliance and d the damping ratio.
  */
template<class DataTypes>
class SOFA_Compliant_API UniformCompliance : public core::behavior::Compliance<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformCompliance, DataTypes), SOFA_TEMPLATE(core::behavior::Compliance, DataTypes));

    typedef core::behavior::Compliance<DataTypes> Inherit;
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

    virtual void init();

    /// Compute the compliance matrix
    virtual void reinit();

    /// Set the constraint value
    virtual void setConstraint(const core::ComplianceParams* mparams, core::MultiVecDerivId fId );

    /// return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getMatrix(const core::MechanicalParams*);

protected:
    UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    linearsolver::EigenSparseSquareMatrix<typename DataTypes::Real> matC; ///< compliance matrix
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H


