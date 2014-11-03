#ifndef SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#include "initCompliant.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

/** Compliance uniformly applied to all the DOF.
  Each dof represents a constraint violation, and undergoes force \f$ \lambda = -\frac{1}{c} ( x - d v ) \f$, where c is the compliance and d the damping ratio.
  */
template<class TDataTypes>
class UniformCompliance : public core::behavior::ForceField<TDataTypes>
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
//    typedef defaulttype::Mat<N,N,Real> Block;

    Data< Real > compliance;    ///< Same compliance applied to all the DOFs

    Data< Real > damping;

    virtual void init();

    /// Compute the compliance matrix
    virtual void reinit();

    /// Return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*);

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, double kFact, unsigned int &offset );

	virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, double bFact, unsigned int &offset );

    /// addForce does nothing when this component is processed like a compliance.
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &);

    /// addDForce does nothing when this component is processed like a compliance.
    virtual void addDForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &);

    virtual double getPotentialEnergy(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord&  /* x */) const
    {
        serr << "getPotentialEnergy() not implemented" << sendl;
        return 0.0;
    }

protected:
    UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

	typedef linearsolver::EigenBaseSparseMatrix<typename DataTypes::Real> matrix_type;
	matrix_type matC; ///< compliance matrix

    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> block_matrix_type;
    block_matrix_type matK; ///< stiffness matrix (Negative S.D.)
    block_matrix_type matB; /// damping matrix (Negative S.D.)
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H


