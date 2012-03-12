#ifndef SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#define SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H
#include "Compliance.h"
#include <sofa/defaulttype/Mat.h>

namespace sofa
{
namespace component
{
namespace compliance
{

template<class DataTypes>
class SOFA_Compliant_API UniformCompliance : public core::behavior::Compliance<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UniformCompliance, DataTypes), SOFA_TEMPLATE(core::behavior::Compliance, DataTypes));

    typedef core::behavior::Compliance<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    enum { N=DataTypes::deriv_total_size };
    typedef defaulttype::Mat<N,N,Real> Block;

    /// Compute the displacement in response to the given force
    virtual void computeDisplacement(const core::MechanicalParams* mparams, DataVecDeriv& displacement, const DataVecDeriv& force );

    Data< Block > compliance;  ///< Same compliance applied to all the DOF

protected:
    UniformCompliance( core::behavior::MechanicalState<DataTypes> *mm = NULL);
};

}
}
}

#endif // SOFA_COMPONENT_COMPLIANCE_UniformCompliance_H


