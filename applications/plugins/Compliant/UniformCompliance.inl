#include "UniformCompliance.h"

namespace sofa
{
namespace component
{
namespace compliance
{

template<class DataTypes>
UniformCompliance<DataTypes>::UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm )
    : Inherit(mm)
    , compliance( initData(&compliance, Block(), "compliance", "Compliance value uniformly applied to all the DOF."))
{
}

// Compute the displacement in response to the given force
template<class DataTypes>
void UniformCompliance<DataTypes>::computeDisplacement(const core::MechanicalParams* /*mparams*/, DataVecDeriv& displacement, const DataVecDeriv& force )
{
    helper::WriteAccessor< DataVecDeriv > d(displacement);
    helper::ReadAccessor< DataVecDeriv > f(force);
    for(unsigned i=0; i<d.size(); i++)
        d[i] = compliance.getValue() * f[i];
}


}
}
}
