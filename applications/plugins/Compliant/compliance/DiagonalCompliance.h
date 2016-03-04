#ifndef SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H
#include <Compliant/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

/**

    A simple diagonal compliance matrix. Entries are entered in @diagonal.

    @author: Maxime Tournier

  */
template<class TDataTypes>
class DiagonalCompliance : public core::behavior::ForceField<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DiagonalCompliance, TDataTypes), SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

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

    Data< VecDeriv > diagonal; ///< diagonal values

    Data< helper::vector<SReal> > damping; ///< diagonal damping

    virtual void init();

    /// Compute the compliance matrix
    virtual void reinit();

    virtual SReal getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x ) const;

    /// Return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*);

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset );

    virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset );

    /// addForce does nothing when this component is processed like a compliance.
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &);

    /// addDForce does nothing when this component is processed like a compliance.
    virtual void addDForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &);


protected:
    DiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    typedef linearsolver::EigenBaseSparseMatrix<typename DataTypes::Real> matrix_type;
    matrix_type matC; ///< compliance matrix

    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> block_matrix_type;
    block_matrix_type matK; ///< stiffness matrix (Negative S.D.)
    block_matrix_type matB; /// damping matrix (Negative S.D.)

    static const Real s_complianceEpsilon; /// threshold for which stiffness can be computed by inverting compliance
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H


