#ifndef SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H
#include "initCompliant.h"
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/component/linearsolver/EigenSparseMatrix.h>

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

    Data< VecDeriv > diagonal; /// diagonal values


    virtual void init();

    /// Compute the compliance matrix
    virtual void reinit();

    /// Return a pointer to the compliance matrix
    virtual const sofa::defaulttype::BaseMatrix* getComplianceMatrix(const core::MechanicalParams*);

    /// addForce does nothing when this component is processed like a compliance.
    virtual void addForce(const core::MechanicalParams *, DataVecDeriv &, const DataVecCoord &, const DataVecDeriv &);

protected:
    DiagonalCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);

    linearsolver::EigenBaseSparseMatrix<typename DataTypes::Real> matC; ///< compliance matrix
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_DiagonalCompliance_H


