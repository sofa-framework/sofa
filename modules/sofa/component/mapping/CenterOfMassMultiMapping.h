#ifndef SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
#define SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H


#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMass.h>

#include <sofa/defaulttype/Vec3Types.h>

#include <sofa/component/component.h>

namespace sofa
{

namespace component
{

namespace mapping
{


template <class TIn, class TOut>
class CenterOfMassMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CenterOfMassMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef In InDataTypes;
    typedef typename In::Coord    InCoord;
    typedef typename In::Deriv    InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef Out OutDataTypes;
    typedef typename Out::Coord   OutCoord;
    typedef typename Out::Deriv   OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename OutCoord::value_type Real;

    typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;
    typedef typename helper::vector<const InVecCoord*> vecConstInVecCoord;

    virtual void apply(const vecOutVecCoord& outPos, const vecConstInVecCoord& inPos );
    virtual void applyJ(const helper::vector< OutVecDeriv*>& outDeriv, const helper::vector<const InVecDeriv*>& inDeriv);
    virtual void applyJT( const helper::vector<InVecDeriv*>& outDeriv , const helper::vector<const OutVecDeriv*>& inDeriv );
    virtual void applyDJT(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) {}

    virtual void init();
    void draw(const core::visual::VisualParams* vparams);

protected:

    CenterOfMassMultiMapping()
        : Inherit()
    {
    }

    CenterOfMassMultiMapping(helper::vector< core::State<In>* > in, helper::vector< core::State<Out>* > out)
        : Inherit(in, out)
    {
    }

    virtual ~CenterOfMassMultiMapping() {}

    helper::vector<const core::behavior::BaseMass*> inputBaseMass;
    InVecCoord inputWeightedCOM;
    InVecDeriv inputWeightedForce;
    helper::vector<double> inputTotalMass;
    double invTotalMass;
};


using namespace sofa::defaulttype;

#if defined(WIN32) && !defined(SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Vec3dTypes, Vec3dTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3dTypes, Rigid3dTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3dTypes, Vec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Vec3fTypes, Vec3fTypes >;
extern template class SOFA_MISC_MAPPING_API CenterOfMassMultiMapping< Rigid3fTypes, Rigid3fTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_MAPPING_CENTEROFMASSMULTIMAPPING_H
