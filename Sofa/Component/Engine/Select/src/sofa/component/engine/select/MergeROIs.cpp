/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define MergeROIs_CPP_

#include <sofa/component/engine/select/MergeROIs.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::engine::select
{

using namespace sofa;

void MergeROIs::init()
{
    addInput(&d_nbROIs);
    f_indices.resize(d_nbROIs.getValue());
    addOutput(&d_outputIndices);
    setDirtyValue();
}

void MergeROIs::reinit()
{
    f_indices.resize(d_nbROIs.getValue());
    update();
}

void MergeROIs::parse ( core::objectmodel::BaseObjectDescription* arg )
{
    f_indices.parseSizeData(arg, d_nbROIs);
    Inherit1::parse(arg);
}

void MergeROIs::parseFields ( const std::map<std::string,std::string*>& str )
{
    f_indices.parseFieldsSizeData(str, d_nbROIs);
    Inherit1::parseFields(str);
}

void MergeROIs::doUpdate()
{
    const size_t nb = d_nbROIs.getValue();
    f_indices.resize(nb);
    if(!nb) return;

    helper::WriteOnlyAccessor< Data< type::vector<type::SVector<sofa::Index> > > > outputIndices = d_outputIndices;
    outputIndices.resize(nb);

    for(size_t j=0; j<nb;j++)
    {
        helper::ReadAccessor< Data< type::vector<sofa::Index> > > indices = f_indices[j];
        outputIndices[j].resize(indices.size());
        for(size_t i=0 ; i<indices.size() ; i++) outputIndices[j][i]=indices[i];
    }
}
int MergeROIsClass = core::RegisterObject("Merge a list of ROIs (vector<Indices>) into a single Data (vector<svector<Indices>>)")
        .add< MergeROIs >(true)
        ;


} //namespace sofa::component::engine::select
