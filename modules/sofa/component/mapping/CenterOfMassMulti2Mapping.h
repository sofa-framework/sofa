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
    typedef typename In1::VecCoord In1VecCoord;
    typedef typename In1::VecDeriv In1VecDeriv;
    typedef typename In1::DataTypes In1DataTypes;

    typedef typename Inherit::In2  In2;
    typedef typename In2::Coord    In2Coord;
    typedef typename In2::Deriv    In2Deriv;
    typedef typename In2::VecCoord In2VecCoord;
    typedef typename In2::VecDeriv In2VecDeriv;
    typedef typename In2::DataTypes In2DataTypes;

    typedef typename Inherit::Out Out;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename OutCoord::value_type Real;

    CenterOfMassMulti2Mapping():Inherit() {}

    virtual ~CenterOfMassMulti2Mapping()
    {}
    virtual void apply(const helper::vector<OutVecCoord*>& outPos, const helper::vector<const In1VecCoord*>& inPos1 , const helper::vector<const In2VecCoord*>& inPos2 );
    virtual void applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const In1VecDeriv*>& inDeriv1, const helper::vector<const In2VecDeriv*>& inDeriv2);
    virtual void applyJT( const helper::vector<In1VecDeriv*>& outDeriv1 ,const helper::vector<In2VecDeriv*>& outDeriv2 , const helper::vector<const OutVecDeriv*>& inDeriv );

    virtual void init();
    void draw();



protected:

    helper::vector<const core::behavior::BaseMass*> inputBaseMass1;
    helper::vector<In1Coord> inputWeightedCOM1;
    helper::vector<In1Deriv> inputWeightedForce1;
    helper::vector<double> inputTotalMass1;

    helper::vector<const core::behavior::BaseMass*> inputBaseMass2;
    helper::vector<In2Coord> inputWeightedCOM2;
    helper::vector<In2Deriv> inputWeightedForce2;
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

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTI2MAPPING_H
