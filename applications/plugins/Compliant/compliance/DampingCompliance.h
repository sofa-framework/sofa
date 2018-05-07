#ifndef DAMPINGCOMPLIANCE_H
#define DAMPINGCOMPLIANCE_H

#include <Compliant/config.h>
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
    This component must be a compliance, otherwise there is other way to generate damping (regular B matrix)
    @warning: must be coupled with a DampingValue

    @author Matthieu Nesme
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


    DampingCompliance()
        : damping(initData(&damping, real(0.0), "damping", "damping value"))
    {
        this->isCompliance.setValue(true);
        this->isCompliance.setDisplayed( false );
        this->isCompliance.setReadOnly( true );
    }


    virtual void reinit()
    {
        this->rayleighStiffness.setValue(0); // Rayleigh damping makes no sense here

        // if no damping, set it as a stiffness that does nothing
        if( !damping.getValue() )
        {
            this->isCompliance.setValue(false);
            matC.resize(0,0);
            return;
        }


        this->isCompliance.setValue(true);
        m_lastDt = this->getContext()->getDt();

        unsigned int matrixsize = this->mstate->getMatrixSize();

        real compliance = this->getContext()->getDt() / damping.getValue();

        matC.resize(matrixsize,matrixsize);

        for(unsigned i = 0; i < matrixsize; ++i) {
            matC.beginRow( i );
            matC.insertBack(i, i, compliance);
        }
        matC.compressedMatrix.finalize();

    }


    /// Return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*) {
        if( m_lastDt != this->getContext()->getDt() ) reinit();
        return &matC;
	}

    /// unassembled API
    virtual void addClambda(const core::MechanicalParams *, DataVecDeriv & res, const DataVecDeriv &lambda, SReal cfactor)
    {
        if( m_lastDt != this->getContext()->getDt() ) reinit();
        matC.addMult( res, lambda, cfactor );
    }

    // stiffness implementation makes no sense
    virtual SReal getPotentialEnergy( const core::MechanicalParams*, const DataVecCoord& ) const { return 0; }
    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix*, SReal, unsigned int& ) {}
    virtual void addForce(const core::MechanicalParams*, DataVecDeriv&, const DataVecCoord&, const DataVecDeriv&) {}
    virtual void addDForce(const core::MechanicalParams*, DataVecDeriv&, const DataVecDeriv&) {}


	typedef typename DataTypes::Real real;

	Data<real> damping; ///< damping value

protected:

    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> matrix_type;
    matrix_type matC; ///< compliance matrix

    SReal m_lastDt; /// if the dt changed, the compliance matrix must be updated

};


}
}
}


#endif
