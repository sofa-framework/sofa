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
    typedef typename In::DataTypes InDataTypes;
    typedef typename In::Coord    InCoord;
    typedef typename In::Deriv    InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename Out::DataTypes OutDataTypes;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename OutCoord::value_type Real;

    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;
    typedef typename helper::vector<const InVecCoord*> vecConstInVecCoord;

    CenterOfMassMultiMapping() {}
    virtual ~CenterOfMassMultiMapping() {}

    virtual void apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos );
    virtual void applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv);
    virtual void applyJT( const helper::vector<InVecDeriv*>& outDeriv , const helper::vector<const OutVecDeriv*>& inDeriv );

    virtual void init();
    void draw();



protected:

    helper::vector<const core::behavior::BaseMass*> inputBaseMass;
    InVecCoord inputWeightedCOM;
    InVecDeriv inputWeightedForce;
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

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
