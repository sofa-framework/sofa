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
#pragma once
#include <sofa/component/engine/select/MeshSubsetEngine.h>

namespace sofa::component::engine::select
{

template <class DataTypes>
MeshSubsetEngine<DataTypes>::MeshSubsetEngine()
    : Inherited()
    , d_inputPosition(initData(&d_inputPosition,"inputPosition","input vertices"))
    , d_inputEdges(initData(&d_inputEdges,"inputEdges","input edges"))
    , d_inputTriangles(initData(&d_inputTriangles,"inputTriangles","input triangles"))
    , d_inputQuads(initData(&d_inputQuads,"inputQuads","input quads"))
    , d_inputTetrahedra(initData(&d_inputTetrahedra,"inputTetrahedra","input tetrahedra"))
    , d_inputHexahedra(initData(&d_inputHexahedra,"inputHexahedra","input hexahedra"))
    , d_indices(initData(&d_indices,"indices","Index lists of the selected vertices"))
    , d_position(initData(&d_position,"position","Vertices of mesh subset"))
    , d_edges(initData(&d_edges,"edges","edges of mesh subset"))
    , d_triangles(initData(&d_triangles,"triangles","Triangles of mesh subset"))
    , d_quads(initData(&d_quads,"quads","Quads of mesh subset"))
    , d_tetrahedra(initData(&d_tetrahedra,"tetrahedra","Tetrahedra of mesh subset"))
    , d_hexahedra(initData(&d_hexahedra,"hexahedra","Hexahedra of mesh subset"))
{
    addInput(&d_inputPosition);
    addInput(&d_inputEdges);
    addInput(&d_inputTriangles);
    addInput(&d_inputQuads);
    addInput(&d_inputTetrahedra);
    addInput(&d_inputHexahedra);
    addInput(&d_indices);
    addOutput(&d_position);
    addOutput(&d_edges);
    addOutput(&d_triangles);
    addOutput(&d_quads);
    addOutput(&d_tetrahedra);
    addOutput(&d_hexahedra);

    inputPosition.setOriginalData(&d_inputPosition);
    inputEdges.setOriginalData(&d_inputEdges);
    inputTriangles.setOriginalData(&d_inputTriangles);
    inputQuads.setOriginalData(&d_inputQuads);
    indices.setOriginalData(&d_indices);
    position.setOriginalData(&d_position);
    edges.setOriginalData(&d_edges);
    triangles.setOriginalData(&d_triangles);
    quads.setOriginalData(&d_quads);
}

template <class DataTypes>
MeshSubsetEngine<DataTypes>::~MeshSubsetEngine()
{
}

template <class ElementType>
void extractElements(
    const std::map<core::topology::BaseMeshTopology::PointID, core::topology::BaseMeshTopology::PointID>& indexMapping,
    const sofa::type::vector<ElementType>& elements,
    sofa::type::vector<ElementType>& subsetElements)
{
    subsetElements.clear();

    for (const auto& element : elements)
    {
        bool inside = true;
        ElementType newElement;
        for (size_t j = 0; j < ElementType::NumberOfNodes; j++)
        {
            auto it = indexMapping.find(element[j]);
            if (it == indexMapping.end())
            {
                inside = false;
                break;
            }
            newElement[j] = it->second;
        }
        if (inside)
        {
            subsetElements.push_back(newElement);
        }
    }
}

template <class DataTypes>
void MeshSubsetEngine<DataTypes>::doUpdate()
{
    helper::ReadAccessor<Data< SeqPositions > > pos(this->inputPosition);
    const helper::ReadAccessor<Data< SetIndices > >  ind(this->indices);
    helper::WriteOnlyAccessor<Data< SeqPositions > > opos(this->position);

    opos.resize(ind.size());
    std::map<PointID, PointID> FtoS;
    for (size_t i = 0; i < ind.size(); i++)
    {
        opos[i] = pos[ind[i]];
        FtoS[ind[i]] = i;
    }

    const auto computeSubset = [&FtoS](const auto& inputElements, auto& subsetElements)
    {
        auto accessor = sofa::helper::getWriteOnlyAccessor(subsetElements);
        extractElements(FtoS, inputElements.getValue(), accessor.wref());
    };

    computeSubset(d_inputEdges, d_edges);
    computeSubset(d_inputTriangles, d_triangles);
    computeSubset(d_inputQuads, d_quads);
    computeSubset(d_inputTetrahedra, d_tetrahedra);
    computeSubset(d_inputHexahedra, d_hexahedra);
}


} //namespace sofa::component::engine::select
