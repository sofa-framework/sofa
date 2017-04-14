/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_ENGINE_SMOOTHMESHENGINE_INL
#define SOFA_COMPONENT_ENGINE_SMOOTHMESHENGINE_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include "SmoothMeshEngine.h"
#include <sofa/helper/gl/template.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
SmoothMeshEngine<DataTypes>::SmoothMeshEngine()
    : l_topology( initLink( "topology", "Link to a BaseTopology component"))
    , input_position( initData (&input_position, "input_position", "Input position") )
    , input_indices( initData (&input_indices, "input_indices", "Position indices that need to be smoothed, leave empty for all positions") )
    , output_position( initData (&output_position, "output_position", "Output position") )
    , nb_iterations( initData (&nb_iterations, (unsigned int)1, "nb_iterations", "Number of iterations of laplacian smoothing") )
{
    this->addAlias(&input_position,"inputPosition");
    this->addAlias(&output_position,"outputPosition");
    this->addAlias(&nb_iterations,"iterations");
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::init()
{
    if( !l_topology )
    {
        l_topology = this->getContext()->getMeshTopology();
        if (!l_topology)
            serr << "requires a mesh topology" << sendl;
    }

    addInput(&input_position);
    addInput(&input_indices);
    addOutput(&output_position);

    setDirtyValue();
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void SmoothMeshEngine<DataTypes>::update()
{
    using sofa::core::topology::BaseMeshTopology;

    helper::ReadAccessor< Data<VecCoord> > in(input_position);
    helper::ReadAccessor< Data<helper::vector <unsigned int > > > indices(input_indices);

    cleanDirty();

    if (!l_topology) return;

    helper::WriteOnlyAccessor< Data<VecCoord> > out(output_position);

    out.resize(in.size());
    VecCoord t( out.size() ); // temp

    for (unsigned int i =0; i<in.size();i++) t[i] = out[i] = in[i];

    
    for (unsigned int n=0; n < nb_iterations.getValue(); n++)
    {
        if( indices.empty() )
        {
            for (unsigned int i = 0; i < out.size(); i++)
            {
                BaseMeshTopology::VerticesAroundVertex v = l_topology->getVerticesAroundVertex(i);
                for (unsigned int j = 0; j < v.size(); j++)
                    t[i] += out[v[j]];
                t[i] /= (v.size()+1);
            }
            for (unsigned int i=0 ; i<in.size() ; i++ ) out[i] = t[i];
        }
        else
        {
            for(unsigned int i = 0; i < indices.size(); i++)
            {
                BaseMeshTopology::VerticesAroundVertex v = l_topology->getVerticesAroundVertex(indices[i]);
                for (unsigned int j = 0; j < v.size(); j++)
                    t[indices[i]] += out[v[j]];
                t[indices[i]] /= (v.size()+1);
            }
            for(unsigned int i = 0; i < indices.size(); i++) out[indices[i]] = t[indices[i]];
        }
    }

}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
