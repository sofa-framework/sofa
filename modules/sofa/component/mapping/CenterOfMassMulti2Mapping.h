#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_H
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_H


#include <sofa/core/Multi2Mapping.h>
#include <sofa/core/behavior/MechanicalMulti2Mapping.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/VecId.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/component.h>

namespace sofa
{
namespace component
{
namespace mapping
{

template < class BasicMulti2Mapping >
class CenterOfMassMulti2Mapping : public BasicMulti2Mapping
{
public :
    SOFA_CLASS(SOFA_TEMPLATE(CenterOfMassMulti2Mapping,BasicMulti2Mapping), BasicMulti2Mapping);

    typedef BasicMulti2Mapping     Inherit;
    typedef typename Inherit::In1  In1;
    typedef typename In1::Coord    In1Coord;
    typedef typename In1::Deriv    In1Deriv;

    typedef typename Inherit::In2  In2;
    typedef typename In2::Coord    In2Coord;
    typedef typename In2::Deriv    In2Deriv;

    typedef typename Inherit::Out Out;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename OutCoord::value_type Real;

    CenterOfMassMulti2Mapping():Inherit() {};

    virtual ~CenterOfMassMulti2Mapping()
    {};
    virtual void apply(const helper::vector<typename Out::VecCoord*>& OutPos, const helper::vector<const typename In1::VecCoord*>& InPos1 , const helper::vector<const typename In2::VecCoord*>& InPos2 );
    virtual void applyJ(const helper::vector< typename Out::VecDeriv*>& OutDeriv, const helper::vector<const typename In1::VecDeriv*>& InDeriv1, const helper::vector<const typename In2::VecDeriv*>& InDeriv2);
    virtual void applyJT( const helper::vector<typename In1::VecDeriv*>& OutDeriv1 ,const helper::vector<typename In2::VecDeriv*>& OutDeriv2 , const helper::vector<const typename Out::VecDeriv*>& InDeriv );

    virtual void init();
    void draw();



protected:

    helper::vector<const core::behavior::BaseMass*> inputBaseMass1;
    helper::vector<typename In1::Coord> inputWeightedCOM1;
    helper::vector<typename In1::Deriv> inputWeightedForce1;
    helper::vector<double> inputTotalMass1;

    helper::vector<const core::behavior::BaseMass*> inputBaseMass2;
    helper::vector<typename In2::Coord> inputWeightedCOM2;
    helper::vector<typename In2::Deriv> inputWeightedForce2;
    helper::vector<double> inputTotalMass2;

    double invTotalMass;
};


using namespace core::behavior;
using namespace sofa::defaulttype;
using namespace sofa::core;
#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CenterOfMassMulti2Mapping_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping<
MechanicalMulti2Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> >
> ;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping<
MechanicalMulti2Mapping< MechanicalState<Vec3dTypes>, MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> >
> ;

#endif
#ifndef SOFA_DOUBLE
extern  template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping<
MechanicalMulti2Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> >
>;
extern template class SOFA_COMPONENT_MAPPING_API CenterOfMassMulti2Mapping<
MechanicalMulti2Mapping< MechanicalState<Vec3fTypes>, MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> >
>;

#endif
#endif
}
}
}

#endif //SOFA_COMPONENT_MAPPING_CenterOfMassMulti2Mapping_H
