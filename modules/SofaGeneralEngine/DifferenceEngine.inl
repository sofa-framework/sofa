#ifndef SOFA_COMPONENT_ENGINE_DifferenceEngine_INL
#define SOFA_COMPONENT_ENGINE_DifferenceEngine_INL


#include "DifferenceEngine.h"
#include <sofa/helper/logging/Messaging.h>


namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
DifferenceEngine<DataTypes>::DifferenceEngine()
    : d_input ( initData (&d_input, "input", "input vector") )
    , d_substractor ( initData (&d_substractor, "substractor", "vector to substract to input") )
    , d_output( initData (&d_output, "output", "output vector = input-substractor") )
{

}

template <class DataType>
void DifferenceEngine<DataType>::init()
{
    addInput(&d_input);
    addInput(&d_substractor);
    addOutput(&d_output);
    setDirtyValue();
}

template <class DataType>
void DifferenceEngine<DataType>::reinit()
{
    update();
}

template <class DataType>
void DifferenceEngine<DataType>::update()
{
    helper::ReadAccessor<Data<VecData> > in = d_input;
    helper::ReadAccessor<Data<VecData> > sub = d_substractor;

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecData> > out = d_output;

    if(in.size() != sub.size())
    {
        msg_warning(this) << "Input vector and vector to substract should have same size. Abort.";
        return;
    }

    out.resize( in.size() );

    for( size_t i=0 ; i<in.size() ; ++i )
        out[i] = in[i] - sub[i];
}

} // namespace engine

} // namespace component

} // namespace sofa


#endif
