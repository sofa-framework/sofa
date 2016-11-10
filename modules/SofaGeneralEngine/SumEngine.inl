#ifndef SOFA_COMPONENT_ENGINE_SumEngine_INL
#define SOFA_COMPONENT_ENGINE_SumEngine_INL


#include "SumEngine.h"
#include <sofa/helper/logging/Messaging.h>
#include <numeric>


namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
SumEngine<DataTypes>::SumEngine()
    : d_input ( initData (&d_input, "input", "input vector") )
    , d_output( initData (&d_output, "output", "output sum") )
{

}

template <class DataType>
void SumEngine<DataType>::init()
{
    addInput(&d_input);
    addOutput(&d_output);
    setDirtyValue();
}

template <class DataType>
void SumEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void SumEngine<DataType>::update()
{

    helper::ReadAccessor<Data<VecData> > in = d_input;

    cleanDirty();

    helper::WriteOnlyAccessor<Data<DataType> > out = d_output;
    out.wref() = std::accumulate(in.begin(), in.end(), DataType() );
}

} // namespace engine

} // namespace component

} // namespace sofa


#endif
