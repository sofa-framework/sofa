#ifndef SOFA_COMPONENT_COMPLIANCE_UniformLinearPotentialEnergy_H
#define SOFA_COMPONENT_COMPLIANCE_UniformLinearPotentialEnergy_H

#include <Compliant/config.h>
#include <sofa/core/behavior/ForceField.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

/** 

    The linear potential energy is given for 1D dofs by:
    V(x) = k.x   (would it bring something to use "k.x + b"?)
    f = dV/dx = k
    K = df/fx = 0

  */

template<class TDataTypes>
class UniformLinearPotentialEnergy : public core::behavior::ForceField<TDataTypes>
{
public:
    
    SOFA_CLASS(SOFA_TEMPLATE(UniformLinearPotentialEnergy, TDataTypes),
               SOFA_TEMPLATE(core::behavior::ForceField, TDataTypes));

    typedef core::behavior::ForceField<TDataTypes> Base;

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;

    Data<SReal> d_factor; ///< scalar factor

    virtual SReal getPotentialEnergy( const core::MechanicalParams* /*mparams*/,
                                      const DataVecCoord& x ) const {
        const VecCoord& pos = x.getValue();
        const SReal& k = d_factor.getValue();

        SReal e = k * pos[0][0];
        for( size_t i=1, iend = pos.size() ;  i<iend ; ++i )
            e += k * pos[i][0];

        return e;
    }
    
    virtual void addForce(const core::MechanicalParams *,
                          DataVecDeriv &f,
                          const DataVecCoord & x,
                          const DataVecDeriv &) {

        const VecCoord& pos = x.getValue();
        const SReal& k = d_factor.getValue();
        VecDeriv& ff = *f.beginEdit();

        for( size_t i=0, iend = pos.size() ;  i<iend ; ++i )
            ff[i][0] -= k;

        f.endEdit();
    }

    virtual void addDForce( const core::MechanicalParams *, DataVecDeriv &, const DataVecDeriv &) {}

    virtual void addKToMatrix( sofa::defaulttype::BaseMatrix * /*matrix*/, SReal /*kFact*/, unsigned int &offset )
    {
        offset += this->mstate->getMatrixSize();
    }


protected:

    UniformLinearPotentialEnergy( core::behavior::MechanicalState<DataTypes> *mm = 0)
        : Base(mm)
        , d_factor(initData(&d_factor, (SReal)1.0, "factor", "scalar factor"))
    {}
    
};

}
}
}

#endif


