#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H


#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/MechanicalMultiMapping.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/component/component.h>

namespace sofa
{
namespace component
{
namespace mapping
{

template < class BasicMultiMapping >
class CenterOfMassMultiMapping : public BasicMultiMapping
{
public :
    SOFA_CLASS(SOFA_TEMPLATE(CenterOfMassMultiMapping,BasicMultiMapping), BasicMultiMapping);

    typedef BasicMultiMapping     Inherit;
    typedef typename Inherit::In  In;
    typedef typename Inherit::Out Out;
    typedef typename In::Coord    InCoord;
    typedef typename In::Deriv    InDeriv;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename OutCoord::value_type Real;

    CenterOfMassMultiMapping():Inherit() {};

    virtual ~CenterOfMassMultiMapping()
    {};
    virtual void apply(const helper::vector<typename Out::VecCoord*>& OutPos, const helper::vector<const typename In::VecCoord*>& InPos );
    virtual void applyJ(const helper::vector< typename Out::VecDeriv*>& OutDeriv, const helper::vector<const typename In::VecDeriv*>& InDeriv);
    virtual void applyJT( const helper::vector<typename In::VecDeriv*>& OutDeriv , const helper::vector<const typename Out::VecDeriv*>& InDeriv );

    virtual void init();
    void draw();



protected:

    helper::vector<const core::behavior::BaseMass*> inputBaseMass;
    helper::vector<typename In::Coord> inputWeightedCOM;
    helper::vector<typename In::Deriv> inputWeightedForce;
    helper::vector<double> inputTotalMass;
    double invTotalMass;




};


using namespace core::behavior;
using namespace sofa::defaulttype;
using namespace sofa::core;
#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping<
MultiMapping<
MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes>
>
> ;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping<
MechanicalMultiMapping<
MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> >
> ;

#endif
#ifndef SOFA_DOUBLE
extern  template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping<
MultiMapping<
MechanicalState<Vec3fTypes>, MechanicalState< Vec3fTypes >
>
>;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMultiMapping<
MechanicalMultiMapping<
MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes>
>
>;

#endif
#endif
}
}
}

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
