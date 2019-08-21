#ifndef SOFA_COMPONENT_Stiffness_UniformStiffness_H
#define SOFA_COMPONENT_Stiffness_UniformStiffness_H
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

/** Stiffness uniformly applied to all the DOF.

  w = sum_i k x_i^2
  f(x) = -k x
  K(x) = -k

  @author Matthieu Nesme
  @date 2017

  */
template<class TDataTypes>
class UniformStiffness : public core::behavior::ForceField<TDataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformStiffness, TDataTypes), SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

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

    Data< Real > stiffness;    ///< Same Stiffness applied to all the DOFs

    Data< Real > damping; ///< uniform viscous damping.

    Data< bool > resizable; ///< can the associated dofs can be resized? (in which case the matrices must be updated)


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
    UniformStiffness( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    typedef linearsolver::EigenSparseMatrix<TDataTypes,TDataTypes> block_matrix_type;
    block_matrix_type matC; ///< Stiffness matrix
    block_matrix_type matK; ///< stiffness matrix (Negative S.D.)
    block_matrix_type matB; /// damping matrix (Negative S.D.)

};

}
}
}

#endif // SOFA_COMPONENT_Stiffness_UniformStiffness_H


