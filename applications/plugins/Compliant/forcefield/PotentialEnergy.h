#ifndef SOFA_COMPONENT_COMPLIANCE_PotentialEnergy_H
#define SOFA_COMPONENT_COMPLIANCE_PotentialEnergy_H

#include <Compliant/config.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include "../utils/edit.h"

namespace sofa
{
namespace component
{
namespace forcefield
{

/** 
    
    dummy forcefield to flag a potential energy, all the work should
    be done upstream in mappings. The potential energy is given for 1D
    dofs by:

    V(x) = x

  */

template<class TDataTypes>
class PotentialEnergy : public core::behavior::ForceField<TDataTypes>
{
public:
    
    SOFA_CLASS(SOFA_TEMPLATE(PotentialEnergy, TDataTypes),
               SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

    typedef core::behavior::ForceField<TDataTypes> Base;

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data<SReal> sign; ///< scalar factor

    virtual SReal getPotentialEnergy( const core::MechanicalParams* /*mparams*/,
                                      const DataVecCoord& x ) const {
        return sign.getValue() * x.getValue()[0][0];
    }
    
    virtual void addForce(const core::MechanicalParams *,
                          DataVecDeriv &f,
                          const DataVecCoord &/*x*/,
                          const DataVecDeriv &) {
        (*edit(f, false))[0][0] -= sign.getValue();
    }

    virtual void addDForce(const core::MechanicalParams *,
                           DataVecDeriv &,
                           const DataVecDeriv &) {
        // nothing lol
    }

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * /*matrix*/,
                               SReal /*kFact*/, unsigned int &offset ) {
        // nothing lol
        ++offset;
    }

    
    PotentialEnergy( core::behavior::MechanicalState<DataTypes> *mm = 0)
        : Base(mm),
          sign(initData(&sign, (SReal)1.0, "sign", "scalar factor")) {}

protected:
    
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_PotentialEnergy_H


