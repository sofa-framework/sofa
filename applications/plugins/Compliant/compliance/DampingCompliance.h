#ifndef DAMPINGCOMPLIANCE_H
#define DAMPINGCOMPLIANCE_H

#include "initCompliant.h"
#include <sofa/core/behavior/ForceField.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>


namespace sofa
{
namespace component
{
namespace forcefield
{

/** 
    A compliance for viscous damping, i.e. generates a force along - \alpha.v
 */

template<class TDataTypes>
class SOFA_Compliant_API DampingCompliance : public core::behavior::ForceField<TDataTypes> {
public:
	
	SOFA_CLASS(SOFA_TEMPLATE(DampingCompliance, TDataTypes), 
	           SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));
	
	
	typedef TDataTypes DataTypes;
	typedef core::behavior::ForceField<TDataTypes> Inherit;
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecDeriv VecDeriv;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef typename Coord::value_type Real;
	typedef core::objectmodel::Data<VecCoord> DataVecCoord;
	typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    DampingCompliance() : damping(initData(&damping, real(0.0), "damping", "damping value")) { this->isCompliance.setValue(true); }

    /// Return a pointer to the compliance matrix
	virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams* params) {
		if( !damping.getValue() ) return 0;
		
		real value = params->dt() / damping.getValue();
		
		typename matrix_type::CompressedMatrix& C = matrix.compressedMatrix;

		C.resize( this->mstate->getMatrixSize(),
		          this->mstate->getMatrixSize() );

		C.setZero();

		for(unsigned i = 0, n = this->mstate->getMatrixSize(); i < n; ++i) {
			C.startVec( i );
			C.insertBack(i, i) = value;
		}
		
		C.finalize();

		return &matrix;
	}


	/// this does nothing as we are a compliance
	virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &)  { }

    virtual void addDForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv&   /*df*/ , const DataVecDeriv&  /*dx*/)
    {
        serr << "addDForce() not implemented" << sendl;
    }

    virtual double getPotentialEnergy(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, const DataVecCoord&  /* x */) const
    {
        serr << "getPotentialEnergy() not implemented" << sendl;
        return 0.0;
    }



protected:
	typedef typename DataTypes::Real real;

	Data<real> damping;

	typedef linearsolver::EigenBaseSparseMatrix<real> matrix_type;
	matrix_type matrix; ///< compliance matrix
};

}
}
}


#endif
