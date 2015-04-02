#ifndef SOFA_COMPONENT_ENGINE_NORMENGINE_INL
#define SOFA_COMPONENT_ENGINE_NORMENGINE_INL


#include <SofaEngine/NormEngine.h>


namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
NormEngine<DataTypes>::NormEngine()
    : d_input ( initData (&d_input, "input", "input array of 3d points") )
    , d_output( initData (&d_output, "output", "output array of scalar norms") )
    , d_normType( initData (&d_normType, 2, "normType", "The type of norm. Use a negative value for the infinite norm.") )
{

}

template <class DataType>
void NormEngine<DataType>::init()
{
    addInput(&d_input);
    addOutput(&d_output);
    addInput(&d_normType);
    setDirtyValue();
}

template <class DataType>
void NormEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void NormEngine<DataType>::update()
{
    cleanDirty();

    helper::ReadAccessor<Data<VecData> > in = d_input;
    helper::WriteAccessor<Data<VecReal> > out = d_output;
    int l = d_normType.getValue();

    out.resize( in.size() );

    for( size_t i=0 ; i<in.size() ; ++i )
        out[i] = in[i].lNorm(l);
}

} // namespace engine

} // namespace component

} // namespace sofa


#endif
