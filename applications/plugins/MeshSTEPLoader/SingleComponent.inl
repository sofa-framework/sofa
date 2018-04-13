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

#ifndef SOFA_COMPONENT_ENGINE_SINGLECOMPONENT_INL
#define SOFA_COMPONENT_ENGINE_SINGLECOMPONENT_INL

#include "SingleComponent.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/SetDirectory.h>

#include <iostream>
#include <fstream>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;
using namespace std;

template <class DataTypes>
SingleComponent<DataTypes>::SingleComponent()
    : _positionsI(initData(&_positionsI, "positionsI", "input: vertices position of whole mesh"))
    , _positionsO(initData(&_positionsO, "positionsO", "output: vertices position of the component"))
    , _edgesI(initData(&_edgesI, "edgesI", "input: edges of whole mesh"))
    , _edgesO(initData(&_edgesO, "edgesO", "output: edges of the component"))
    , _trianglesI(initData(&_trianglesI, "trianglesI", "input: triangles of whole mesh"))
    , _trianglesO(initData(&_trianglesO, "trianglesO", "output: triangles of the component"))
    , _normalsI(initData(&_normalsI, "normalsI", "input: normals of the whole mesh"))
    , _normalsO(initData(&_normalsO, "normalsO", "output: normals of the component"))
    , _uvI(initData(&_uvI, "uvI", "input: UV coordinates of the whole mesh"))
    , _uvO(initData(&_uvO, "uvO", "output: UV coordinates of the component"))
    , _indicesComponents(initData(&_indicesComponents, "indicesComponents", "Shape # | number of nodes | number of triangles"))
    , _numberShape(initData(&_numberShape, "numberShape", "Shape number to be loaded (see Outputs tab of STEPLoader for a description of the shapes)"))
{
    addAlias(&_positionsO,"position");
    addAlias(&_trianglesO,"triangles");
    addAlias(&_normalsO,"normals");
    addAlias(&_uvO,"uv");
}

template <class DataTypes>
void SingleComponent<DataTypes>::init()
{
    addInput(&_positionsI);
    addInput(&_trianglesI);
    addInput(&_normalsI);
    addInput(&_indicesComponents);
    addInput(&_numberShape);
    addInput(&_uvI);

    setDirtyValue();

    reinit();
}

template <class DataTypes>
void SingleComponent<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void SingleComponent<DataTypes>::update()
{
    cleanDirty();
    loadMesh();
}

template <class DataTypes>
void SingleComponent<DataTypes>::loadMesh()
{
    const helper::vector<sofa::defaulttype::Vector3>& positionsI = _positionsI.getValue();
    const helper::vector<helper::fixed_array <unsigned int,3> >& trianglesI = _trianglesI.getValue();
    const helper::vector<sofa::defaulttype::Vector3>& normalsI = _normalsI.getValue();
    const helper::vector<sofa::defaulttype::Vector2>& uvI = _uvI.getValue();

    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(_positionsO.beginEdit());
    helper::vector<helper::fixed_array <unsigned int,3> >& my_triangles = *(_trianglesO.beginEdit());
    helper::vector<sofa::defaulttype::Vector3>& my_normals = *(_normalsO.beginEdit());
    helper::vector<sofa::defaulttype::Vector2>& my_uv = *(_uvO.beginEdit());

    my_positions.clear();
    my_triangles.clear();
    my_normals.clear();
    my_uv.clear();

    const helper::vector<helper::fixed_array <unsigned int,3> >& my_indicesComponents = _indicesComponents.getValue();

    unsigned int my_numberShape = _numberShape.getValue();

    if (my_numberShape >= my_indicesComponents.size())
    {
        serr << "Number of the shape not valid" << sendl;
    }
    else
    {
        unsigned int numNodes = 0, numTriangles = 0;
        for (unsigned int i=0; i<my_indicesComponents.size(); ++i)
        {
            if (my_indicesComponents[i][0] == my_numberShape)
            {
                if(positionsI.size()>0 )
                {
                    for (unsigned int j=0; j<my_indicesComponents[i][1]; ++j)
                    {
                        my_positions.push_back(positionsI[j+numNodes]);
                        my_uv.push_back(uvI[j+numNodes]);
                    }
                }

                if(trianglesI.size() > 0 )
                {
                    for (unsigned int j=0; j<my_indicesComponents[i][2]; ++j)
                    {
                        helper::fixed_array <unsigned int,3> triangleTemp(trianglesI[j+numTriangles][0]-numNodes, trianglesI[j+numTriangles][1]-numNodes, trianglesI[j+numTriangles][2]-numNodes);
                        my_triangles.push_back(triangleTemp);
                    }
                }

                if(normalsI.size() > 0 )
                {
                    for (unsigned int j=0; j<my_indicesComponents[i][1]; ++j)
                    {
                        my_normals.push_back(normalsI[j+numNodes]);
                    }
                }
                break;
            }
            numNodes += my_indicesComponents[i][1];
            numTriangles += my_indicesComponents[i][2];
        }
    }

    _positionsO.endEdit();
    _trianglesO.endEdit();
    _normalsO.endEdit();
    _uvO.endEdit();
}

}

}

}

#endif
