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
#ifndef SOFA_COMPONENT_ENGINE_MERGEMESHES_INL
#define SOFA_COMPONENT_ENGINE_MERGEMESHES_INL

#include <SofaGeneralEngine/MergeMeshes.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
MergeMeshes<DataTypes>::MergeMeshes()
    : f_nbMeshes( initData (&f_nbMeshes, (unsigned)2, "nbMeshes", "number of meshes to merge") )
    , f_output_npoints( initData (&f_output_npoints, (unsigned)0, "npoints", "Number Of out points") )
    , f_output_positions(initData(&f_output_positions,"position","Output Vertices of the merged mesh"))
    , f_output_edges(initData(&f_output_edges,"edges","Output Edges of the merged mesh"))
    , f_output_triangles(initData(&f_output_triangles,"triangles","Output Triangles of the merged mesh"))
    , f_output_quads(initData(&f_output_quads,"quads","Output Quads of the merged mesh"))
    , f_output_polygons(initData(&f_output_polygons,"polygons","Output Polygons of the merged mesh"))
    , f_output_tetrahedra(initData(&f_output_tetrahedra,"tetrahedra","Output Tetrahedra of the merged mesh"))
    , f_output_hexahedra(initData(&f_output_hexahedra,"hexahedra","Output Hexahedra of the merged mesh"))
{
    createInputMeshesData();
}

template <class DataTypes>
MergeMeshes<DataTypes>::~MergeMeshes()
{
    deleteInputDataVector(vf_positions);
    deleteInputDataVector(vf_edges);
    deleteInputDataVector(vf_triangles);
    deleteInputDataVector(vf_quads);
    deleteInputDataVector(vf_tetrahedra);
    deleteInputDataVector(vf_hexahedra);
}


template <class DataTypes>
void MergeMeshes<DataTypes>::createInputMeshesData(int nb)
{
    unsigned int n = (nb < 0) ? f_nbMeshes.getValue() : (unsigned int)nb;

    createInputDataVector(n, vf_positions, "position", "input positions for mesh ");
    createInputDataVector(n, vf_edges, "edges", "input edges for mesh ");
    createInputDataVector(n, vf_triangles, "triangles", "input triangles for mesh ");
    createInputDataVector(n, vf_quads, "quads", "input quads for mesh ");
    createInputDataVector(n, vf_tetrahedra, "tetrahedra", "input tetrahedra for mesh ");
    createInputDataVector(n, vf_hexahedra, "hexahedra", "input hexahedra for mesh ");
    if (n != f_nbMeshes.getValue())
        f_nbMeshes.setValue(n);
}


/// Parse the given description to assign values to this object's fields and potentially other parameters
template <class DataTypes>
void MergeMeshes<DataTypes>::parse ( sofa::core::objectmodel::BaseObjectDescription* arg )
{
    const char* p = arg->getAttribute(f_nbMeshes.getName().c_str());
    if (p)
    {
        std::string nbStr = p;
        sout << "parse: setting nbMeshes="<<nbStr<<sendl;
        f_nbMeshes.read(nbStr);
        createInputMeshesData();
    }
    Inherit1::parse(arg);
}

/// Assign the field values stored in the given map of name -> value pairs
template <class DataTypes>
void MergeMeshes<DataTypes>::parseFields ( const std::map<std::string,std::string*>& str )
{
    std::map<std::string,std::string*>::const_iterator it = str.find(f_nbMeshes.getName());
    if (it != str.end() && it->second)
    {
        std::string nbStr = *it->second;
        sout << "parseFields: setting nbMeshes="<<nbStr<<sendl;
        f_nbMeshes.read(nbStr);
        createInputMeshesData();
    }
    Inherit1::parseFields(str);
}

template <class DataTypes>
void MergeMeshes<DataTypes>::init()
{
    addInput(&f_nbMeshes);
    createInputMeshesData();

    addOutput(&f_output_npoints);
    addOutput(&f_output_positions);
    addOutput(&f_output_edges);
    addOutput(&f_output_triangles);
    addOutput(&f_output_quads);
    addOutput(&f_output_tetrahedra);
    addOutput(&f_output_hexahedra);

    setDirtyValue();
}

template <class DataTypes>
void MergeMeshes<DataTypes>::reinit()
{
    createInputMeshesData();

    update();
}

template <class DataTypes>
void MergeMeshes<DataTypes>::update()
{
//    createInputMeshesData();

    unsigned int nb = f_nbMeshes.getValue();

    for (unsigned int i=0; i<nb; ++i)
    {
        vf_positions[i]->updateIfDirty();
        vf_edges[i]->updateIfDirty();
        vf_triangles[i]->updateIfDirty();
        vf_quads[i]->updateIfDirty();
        vf_tetrahedra[i]->updateIfDirty();
        vf_hexahedra[i]->updateIfDirty();
    }

    cleanDirty();

    mergeInputDataVector(nb, f_output_positions, vf_positions);
    mergeInputDataVector(nb, f_output_edges, vf_edges, vf_positions);
    mergeInputDataVector(nb, f_output_triangles, vf_triangles, vf_positions);
    mergeInputDataVector(nb, f_output_quads, vf_quads, vf_positions);
    mergeInputDataVector(nb, f_output_tetrahedra, vf_tetrahedra, vf_positions);
    mergeInputDataVector(nb, f_output_hexahedra, vf_hexahedra, vf_positions);


    unsigned & npoints = *f_output_npoints.beginWriteOnly();
    npoints = (unsigned) f_output_positions.getValue().size();
    f_output_npoints.endEdit();

    sout << "Created merged mesh: "
            << f_output_positions.getValue().size() << " points, ";
    if (f_output_edges.getValue().size() > 0)
        sout << f_output_edges.getValue().size() << " edges, ";
    if (f_output_triangles.getValue().size() > 0)
        sout << f_output_triangles.getValue().size() << " triangles, ";
    if (f_output_quads.getValue().size() > 0)
        sout << f_output_quads.getValue().size() << " quads, ";
    if (f_output_polygons.getValue().size() > 0)
        sout << f_output_polygons.getValue().size() << " polygons, ";
    if (f_output_tetrahedra.getValue().size() > 0)
        sout << f_output_tetrahedra.getValue().size() << " tetrahedra, ";
    if (f_output_hexahedra.getValue().size() > 0)
        sout << f_output_hexahedra.getValue().size() << " hexahedra, ";
    sout << " from " << nb << " input meshes." << sendl;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
