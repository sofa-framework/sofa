/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MERGEVECTORS_INL
#define SOFA_COMPONENT_ENGINE_MERGEVECTORS_INL

#include <SofaGeneralEngine/MergeVectors.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class VecT>
MergeVectors<VecT>::MergeVectors()
    : f_nbInputs( initData(&f_nbInputs, (unsigned)2, "nbInputs", "Number of input vectors") )
    , vf_inputs( this, "input", "Input vector", helper::DataEngineInput )
    , f_output( initData(&f_output , "output", "Output vector") )
{
}

template <class VecT>
MergeVectors<VecT>::~MergeVectors()
{
}

template <class VecT>
void MergeVectors<VecT>::parse( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    vf_inputs.parseSizeData(arg, f_nbInputs);
    Inherit1::parse(arg);
}

template <class VecT>
void MergeVectors<VecT>::parseFields( const std::map<std::string,std::string*>& str )
{
    vf_inputs.parseFieldsSizeData(str, f_nbInputs);
    Inherit1::parseFields(str);
}

template <class VecT>
void MergeVectors<VecT>::init()
{
    addInput(&f_nbInputs);

    vf_inputs.resize( f_nbInputs.getValue() );

    addOutput(&f_output);

    setDirtyValue();
}

template <class VecT>
void MergeVectors<VecT>::reinit()
{
    vf_inputs.resize( f_nbInputs.getValue() );
    update();
}

template <class VecT>
void MergeVectors<VecT>::update()
{
    unsigned int nb = f_nbInputs.getValue();

    for (unsigned int i=0; i<nb; ++i)
        vf_inputs[i]->updateIfDirty();

    cleanDirty();

    helper::vectorData<VecValue>::merge( f_output, vf_inputs );
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
