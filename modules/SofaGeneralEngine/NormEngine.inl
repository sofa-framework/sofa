/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ENGINE_NORMENGINE_INL
#define SOFA_COMPONENT_ENGINE_NORMENGINE_INL


#include <SofaGeneralEngine/NormEngine.h>


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
    helper::ReadAccessor<Data<VecData> > in = d_input;
    int l = d_normType.getValue();

    cleanDirty();

    helper::WriteOnlyAccessor<Data<VecReal> > out = d_output;

    out.resize( in.size() );

    for( size_t i=0 ; i<in.size() ; ++i )
        out[i] = in[i].lNorm(l);


}

} // namespace engine

} // namespace component

} // namespace sofa


#endif
