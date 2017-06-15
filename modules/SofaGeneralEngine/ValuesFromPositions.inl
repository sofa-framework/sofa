/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_INL
#define SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/ValuesFromPositions.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/gl/template.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
ValuesFromPositions<DataTypes>::ValuesFromPositions()
    : f_inputValues( initData(&f_inputValues, "inputValues", "Input values") )
    , f_direction( initData(&f_direction, CPos(0,1,0), "direction", "Direction along which the values are interpolated") )
    , f_X0( initData (&f_X0, "position", "Rest position coordinates of the degrees of freedom") )
    , f_edges(initData (&f_edges, "edges", "Edge Topology") )
    , f_triangles(initData (&f_triangles, "triangles", "Triangle Topology") )
    , f_tetrahedra(initData (&f_tetrahedra, "tetrahedra", "Tetrahedron Topology") )
    , f_values( initData(&f_values,"values","Values of the points contained in the ROI") )
    , f_edgeValues( initData(&f_edgeValues,"edgeValues","Values of the edges contained in the ROI") )
    , f_triangleValues( initData(&f_triangleValues,"triangleValues","Values of the triangles contained in the ROI") )
    , f_tetrahedronValues( initData(&f_tetrahedronValues,"tetrahedronValues","Values of the tetrahedra contained in the ROI") )
    , f_pointVectors( initData(&f_pointVectors,"pointVectors","Vectors of the points contained in the ROI") )
    , f_edgeVectors( initData(&f_edgeVectors,"edgeVectors","Vectors of the edges contained in the ROI") )
    , f_triangleVectors( initData(&f_triangleVectors,"triangleVectors","Vectors of the triangles contained in the ROI") )
    , f_tetrahedronVectors( initData(&f_tetrahedronVectors,"tetrahedronVectors","Vectors of the tetrahedra contained in the ROI") )
    , p_fieldType(initData(&p_fieldType, "fieldType", "field type of output elements"))
    , p_drawVectors(initData(&p_drawVectors, false, "drawVectors", "draw vectors line"))
    , p_vectorLength (initData(&p_vectorLength, (float)10, "drawVectorLength", "vector length visualisation. "))
{
    sofa::helper::OptionsGroup m_newoptiongroup(2,"Scalar","Vector");
    m_newoptiongroup.setSelectedItem("Scalar");
    p_fieldType.setValue(m_newoptiongroup);

    addAlias(&f_X0,"rest_position");
}

template <class DataTypes>
void ValuesFromPositions<DataTypes>::init()
{
    using sofa::core::objectmodel::BaseData;
    using sofa::core::topology::BaseMeshTopology;

    if (!f_X0.isSet())
    {
        sofa::core::behavior::MechanicalState<DataTypes>* mstate;
        this->getContext()->get(mstate);
        if (mstate)
        {
            BaseData* parent = mstate->findData("rest_position");
            if (parent)
            {
                f_X0.setParent(parent);
                f_X0.setReadOnly(true);
            }
        }
        else
        {
            core::loader::MeshLoader* loader = NULL;
            this->getContext()->get(loader);
            if (loader)
            {
                BaseData* parent = loader->findData("position");
                if (parent)
                {
                    f_X0.setParent(parent);
                    f_X0.setReadOnly(true);
                }
            }
        }
    }
    if (!f_edges.isSet() || !f_triangles.isSet() || !f_tetrahedra.isSet())
    {
        BaseMeshTopology* topology;
        this->getContext()->get(topology);
        if (topology)
        {
            if (!f_edges.isSet())
            {
                BaseData* eparent = topology->findData("edges");
                if (eparent)
                {
                    f_edges.setParent(eparent);
                    f_edges.setReadOnly(true);
                }
            }
            if (!f_triangles.isSet())
            {
                BaseData* tparent = topology->findData("triangles");
                if (tparent)
                {
                    f_triangles.setParent(tparent);
                    f_triangles.setReadOnly(true);
                }
            }
            if (!f_tetrahedra.isSet())
            {
                BaseData* tparent = topology->findData("tetrahedra");
                if (tparent)
                {
                    f_tetrahedra.setParent(tparent);
                    f_tetrahedra.setReadOnly(true);
                }
            }
        }
    }

    addInput(&f_inputValues);
    addInput(&f_direction);
    addInput(&f_X0);
    addInput(&f_edges);
    addInput(&f_triangles);
    addInput(&f_tetrahedra);

    addOutput(&f_values);
    addOutput(&f_edgeValues);
    addOutput(&f_triangleValues);
    addOutput(&f_tetrahedronValues);

    addOutput(&f_pointVectors);
    addOutput(&f_edgeVectors);
    addOutput(&f_triangleVectors);
    addOutput(&f_tetrahedronVectors);
    setDirtyValue();
}

template <class DataTypes>
void ValuesFromPositions<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Real ValuesFromPositions<DataTypes>::valueFromPosition(const CPos& p, const TempData& data)
{
    int nbv = data.inputValues.size();

    if (nbv == 0) return 0;
    else if (nbv == 1) return data.inputValues[0];
    Real coef = dot(p,data.dir);
    coef = (coef - data.bmin) / (data.bmax - data.bmin);
    coef *= (nbv-1);
    int v = (int)floor(coef);
    if (v < 0) return data.inputValues[0];
    else if (v >= nbv-1) return data.inputValues[nbv-1];
    coef -= v;
    return data.inputValues[v] * (1-coef) + data.inputValues[v+1] * coef;
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Real ValuesFromPositions<DataTypes>::valueFromPoint(const PointID& pid, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( valueFromPosition(p,data) );
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Real ValuesFromPositions<DataTypes>::valueFromEdge(const Edge& e, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
    CPos c = (p1+p0)*0.5;

    return valueFromPosition(c,data);
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Real ValuesFromPositions<DataTypes>::valueFromTriangle(const Triangle& t, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (valueFromPosition(c,data));
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Real ValuesFromPositions<DataTypes>::valueFromTetrahedron(const Tetra &t, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (valueFromPosition(c,data));
}


template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Vec3 ValuesFromPositions<DataTypes>::vectorFromPosition(const CPos& p, const TempData& data)
{
    (void)p;
    return data.dir;
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Vec3 ValuesFromPositions<DataTypes>::vectorFromPoint(const PointID& pid, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p =  DataTypes::getCPos((*x0)[pid]);
    return ( vectorFromPosition(p,data) );
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Vec3 ValuesFromPositions<DataTypes>::vectorFromEdge(const Edge& e, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[e[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[e[1]]);
    CPos c = (p1+p0)*0.5;

    return vectorFromPosition(c,data);
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Vec3 ValuesFromPositions<DataTypes>::vectorFromTriangle(const Triangle& t, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos c = (p2+p1+p0)/3.0;

    return (vectorFromPosition(c,data));
}

template <class DataTypes>
typename ValuesFromPositions<DataTypes>::Vec3 ValuesFromPositions<DataTypes>::vectorFromTetrahedron(const Tetra &t, const TempData& data)
{
    const VecCoord* x0 = data.x0;
    CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
    CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
    CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
    CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
    CPos c = (p3+p2+p1+p0)/4.0;

    return (vectorFromPosition(c,data));
}



template <class DataTypes>
void ValuesFromPositions<DataTypes>::update()
{
    updateAllInputsIfDirty(); // the easy way to make sure every inputs are up-to-date

    cleanDirty();

    TempData data;
    data.dir = f_direction.getValue();
    data.inputValues = f_inputValues.getValue();
    const VecCoord* x0 = &f_X0.getValue();
    data.x0 = x0;

    // Compute min and max of BB
    sofa::defaulttype::Vec<3, SReal> sceneMinBBox, sceneMaxBBox;
    sofa::simulation::Node* context = dynamic_cast<sofa::simulation::Node*>(this->getContext());
    sofa::simulation::getSimulation()->computeBBox((sofa::simulation::Node*)context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
    data.bmin = (Real)*sceneMinBBox.ptr(); /// @todo: shouldn't this be dot(sceneMinBBox,data.dir) ?
    data.bmax = (Real)*sceneMaxBBox.ptr(); /// @todo: shouldn't this be dot(sceneMaxBBox,data.dir) ?

    if (p_fieldType.getValue().getSelectedId() == 0)
        this->updateValues(data);
    else
        this->updateVectors(data);
}


template <class DataTypes>
void ValuesFromPositions<DataTypes>::updateValues(TempData &_data)
{
    // Read accessor for input topology
    const VecCoord* x0 = &f_X0.getValue();
    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;

    // Write accessor for topological element values
    helper::WriteOnlyAccessor< Data<VecReal> > values = f_values;
    helper::WriteOnlyAccessor< Data<VecReal> > edgeValues = f_edgeValues;
    helper::WriteOnlyAccessor< Data<VecReal> > triangleValues = f_triangleValues;
    helper::WriteOnlyAccessor< Data<VecReal> > tetrahedronValues = f_tetrahedronValues;

    // Clear lists
    values.clear();
    edgeValues.clear();
    triangleValues.clear();
    tetrahedronValues.clear();

    //Points
    for( unsigned int i=0; i<x0->size(); ++i )
    {
        Real v = valueFromPoint(i, _data);
        values.push_back(v);
    }

    //Edges
    for(unsigned int i=0 ; i<edges.size() ; i++)
    {
        Edge e = edges[i];
        Real v = valueFromEdge(e, _data);
        edgeValues.push_back(v);
    }

    //Triangles
    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        Triangle t = triangles[i];
        Real v = valueFromTriangle(t, _data);
        triangleValues.push_back(v);
    }

    //Tetrahedra
    for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
    {
        Tetra t = tetrahedra[i];
        Real v = valueFromTetrahedron(t, _data);
        tetrahedronValues.push_back(v);
    }
}


template <class DataTypes>
void ValuesFromPositions<DataTypes>::updateVectors(TempData &_data)
{
    // Read accessor for input topology
    const VecCoord* x0 = &f_X0.getValue();
    helper::ReadAccessor< Data<helper::vector<Edge> > > edges = f_edges;
    helper::ReadAccessor< Data<helper::vector<Triangle> > > triangles = f_triangles;
    helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;

    // Write accessor for topological element values
    helper::WriteAccessor< Data<sofa::helper::vector<Vec3> > > pointVectors = f_pointVectors;
    helper::WriteAccessor< Data<sofa::helper::vector<Vec3> > > edgeVectors = f_edgeVectors;
    helper::WriteAccessor< Data<sofa::helper::vector<Vec3> > > triangleVectors = f_triangleVectors;
    helper::WriteAccessor< Data<sofa::helper::vector<Vec3> > > tetrahedronVectors = f_tetrahedronVectors;

    // Clear lists
    pointVectors.clear();
    edgeVectors.clear();
    triangleVectors.clear();
    tetrahedronVectors.clear();

    //Points
    for( unsigned int i=0; i<x0->size(); ++i )
    {
        Vec3 v3 = vectorFromPoint(i, _data);
        pointVectors.push_back(v3);
    }

    //Edges
    for(unsigned int i=0 ; i<edges.size() ; i++)
    {
        Edge e = edges[i];
        Vec3 v3 = vectorFromEdge(e, _data);
        edgeVectors.push_back(v3);
    }

    //Triangles
    for(unsigned int i=0 ; i<triangles.size() ; i++)
    {
        Triangle t = triangles[i];
        Vec3 v3 = vectorFromTriangle(t, _data);
        triangleVectors.push_back(v3);
    }

    //Tetrahedra
    for(unsigned int i=0 ; i<tetrahedra.size() ; i++)
    {
        Tetra t = tetrahedra[i];
        Vec3 v3 = vectorFromTetrahedron(t, _data);
        tetrahedronVectors.push_back(v3);
    }
}


template <class DataTypes>
void ValuesFromPositions<DataTypes>::draw(const core::visual::VisualParams* )
{
#ifndef SOFA_NO_OPENGL
    if (p_drawVectors.getValue())
    {
        glDisable(GL_LIGHTING);
        const VecCoord* x0 = &f_X0.getValue();
        helper::ReadAccessor< Data<helper::vector<Tetra> > > tetrahedra = f_tetrahedra;
        helper::WriteAccessor< Data<sofa::helper::vector<Vec3> > > tetrahedronVectors = f_tetrahedronVectors;

        CPos point2, point1;
        sofa::defaulttype::Vec<3,float> colors(0,0,1);

        float vectorLength = p_vectorLength.getValue();
        glBegin(GL_LINES);

        for (unsigned int i =0; i<tetrahedronVectors.size(); i++)
        {
            Tetra t = tetrahedra[i];
            CPos p0 =  DataTypes::getCPos((*x0)[t[0]]);
            CPos p1 =  DataTypes::getCPos((*x0)[t[1]]);
            CPos p2 =  DataTypes::getCPos((*x0)[t[2]]);
            CPos p3 =  DataTypes::getCPos((*x0)[t[3]]);
            point1 = (p3+p2+p1+p0)/4.0;
            point2 = point1 + tetrahedronVectors[i]*vectorLength;

            for(unsigned int j=0; j<3; j++)
                colors[j] = (float)fabs (tetrahedronVectors[i][j]);

            glColor3f (colors[0], colors[1], colors[2]);

            glVertex3d(point1[0], point1[1], point1[2]);
            glVertex3d(point2[0], point2[1], point2[2]);
        }
        glEnd();

    }
#endif /* SOFA_NO_OPENGL */
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
