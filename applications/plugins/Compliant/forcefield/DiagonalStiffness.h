#ifndef SOFA_COMPONENT_Stiffness_DiagonalStiffness_H
#define SOFA_COMPONENT_Stiffness_DiagonalStiffness_H
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

    A simple diagonal Stiffness matrix. Entries are entered in @diagonal.

    @author: Matthieu Nesme
    @date 2017

  */
template<class TDataTypes>
class DiagonalStiffness : public core::behavior::ForceField<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DiagonalStiffness, TDataTypes), SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

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

    /// Compute the Stiffness matrix
    virtual void reinit();

    virtual SReal getPotentialEnergy( const core::MechanicalParams* mparams, const DataVecCoord& x ) const;

    /// Return a pointer to the Stiffness matrix
    virtual const sofa::defaulttype::BaseMatrix* getStiffnessMatrix(const core::MechanicalParams*);

    using Inherit1::addKToMatrix;

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset );

    virtual void addBToMatrix( sofa::defaulttype::BaseMatrix * matrix, SReal bFact, unsigned int &offset );

    /// addForce does nothing when this component is processed like a Stiffness.
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &);

    /// addDForce does nothing when this component is processed like a Stiffness.
    virtual void addDForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &);

    /// unassembled API
    virtual void addClambda(const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &, SReal);

protected:
    DiagonalStiffness( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> block_matrix_type;
    block_matrix_type matC; ///< Stiffness matrix
    block_matrix_type matK; ///< stiffness matrix (Negative S.D.)
    block_matrix_type matB; /// damping matrix (Negative S.D.)
};

}
}
}

#endif // SOFA_COMPONENT_Stiffness_DiagonalStiffness_H


