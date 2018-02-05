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
#ifndef SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_INL
#define SOFA_COMPONENT_ENGINE_TEXTUREINTERPOLATION_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/TextureInterpolation.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
TextureInterpolation<DataTypes>::TextureInterpolation()
    : _inputField (initData (&_inputField, "input_states", "input array of state values."))
    , _inputCoords (initData (&_inputCoords, "input_coordinates", "input array of coordinates values."))
    , _outputCoord (initData (&_outputCoord, "output_coordinates", "output array of texture coordinates."))
    , _scalarField (initData (&_scalarField, (bool) true, "scalarField", "To interpolate only the first dimension of input field (useful if this component need to be templated in higher dimension)."))
    , _minVal (initData (&_minVal, (Real)0, "min_value", "minimum value of state value for interpolation."))
    , _maxVal (initData (&_maxVal, (Real)0, "max_value", "maximum value of state value for interpolation."))
    , _changeScale (initData (&_changeScale, false, "manual_scale", "compute texture interpolation on manually scale defined above."))
    , drawPotentiels (initData (&drawPotentiels, (bool) false, "drawPotentiels", "Debug: view state values."))
    , showIndicesScale (initData(&showIndicesScale, (float) 0.0001, "showIndicesScale", "Debug : scale of state values displayed."))
    ,_vertexPloted (initData(&_vertexPloted, (unsigned int)0, "vertexPloted", "Vertex index of values display in graph for each iteration."))
    ,f_graph( initData(&f_graph,"graph","Vertex state value per iteration") )
{
    f_graph.setWidget("graph");
//     f_graph.setReadOnly(true);
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::init()
{
    addInput(&_inputField);
    addOutput(&_outputCoord);
    setDirtyValue();

    if (!_inputField.isSet())
        this->standardLinearInterpolation();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::reinit()
{
    if (!_inputField.isSet())
        this->standardLinearInterpolation();

    if (_vertexPloted.isDirty())
        this->resetGraph();

    update();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::update()
{
    if (!_inputField.isSet())
        return;

    const sofa::helper::vector <Coord>& realInputs = _inputField.getValue();

    cleanDirty();

    ResizableExtVector2D& outputs = *(_outputCoord.beginWriteOnly());
    outputs.resize (realInputs.size());

    if (realInputs.size() == 0)
    {
        _outputCoord.endEdit();
        return;
    }

    sofa::helper::vector < Coord1D > inputs;
    inputs.resize(realInputs.size());

    if (!_scalarField.getValue() && (realInputs[0].size() > 1)) // > 1D template and use all dim.
    {
        for (unsigned int i = 0; i<inputs.size(); ++i)
        {
            Real tmp = 0;
            for (unsigned int j = 0; j<realInputs[0].size(); ++j)
                tmp += realInputs[i][j];

            inputs[i] = tmp/realInputs[0].size();
        }
    }
    else
    {
        for (unsigned int i = 0; i<inputs.size(); ++i)
            inputs[i] = realInputs[i][0];
    }

    Real minVal = 0;
    Real maxVal = 0;

    // Check min and max values:
    if(_changeScale.getValue())
    {
        if( _minVal.getValue() < _maxVal.getValue() )
        {
            minVal = _minVal.getValue();
            maxVal = _maxVal.getValue();
        }
        else
            serr << "Error: in scale for TextureInterpolation, min_value is not smaller than max_value." << sendl;
    }
    else
    {
        Coord1D min_tmp = *(std::min_element (inputs.begin(), inputs.end()));
        Coord1D max_tmp = *(std::max_element (inputs.begin(), inputs.end()));

        minVal = min_tmp[0];
        maxVal = max_tmp[0];
    }

    Real scale = 1/(maxVal - minVal);

    for (unsigned int i=0; i<inputs.size(); ++i)
    {
        outputs[i][0] = (inputs[i][0] - minVal)*scale;
        outputs[i][1] = 0.0;
    }

    this->updateGraph();

    _outputCoord.endEdit();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::updateGraph()
{
    std::map < std::string, sofa::helper::vector<Real> >& graph = *f_graph.beginEdit();
    const sofa::helper::vector <Coord>& realInputs = _inputField.getValue();

    sofa::helper::vector<Real>& graph_val1 = graph["potential1"];
    graph_val1.push_back( realInputs[ _vertexPloted.getValue()][0] +1); //+1 kick hack for scale resizing probleme with 0.000.. values.

    if (realInputs[0].size() > 1) // Hack for 2D models
    {
        sofa::helper::vector<Real>& graph_val2 = graph["potential2"];
        graph_val2.push_back( realInputs[ _vertexPloted.getValue() ][1] +1 );
    }

    f_graph.endEdit();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::resetGraph()
{
    std::map < std::string, sofa::helper::vector<Real> >& graph = *f_graph.beginEdit();

    sofa::helper::vector<Real>& graph_val1 = graph["potential1"];
    graph_val1.clear();

    Coord test;

    if (test.size() > 1) // Hack for 2D models
    {
        sofa::helper::vector<Real>& graph_val2 = graph["potential2"];
        graph_val2.clear();
    }

    f_graph.endEdit();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::standardLinearInterpolation()
{
    const VecCoord3D& coords = _inputCoords.getValue();
    ResizableExtVector2D& outputs = *(_outputCoord.beginEdit());

    outputs.clear();
    outputs.resize(coords.size());

    SReal Cmin[3]; Cmin[0] = 100000, Cmin[1] = 100000, Cmin[2] = 100000;
    SReal Cmax[3]; Cmax[0] = -100000, Cmax[1] = -100000, Cmax[2] = -100000;

    // creating BB
    for (unsigned int i=0; i<coords.size(); ++i)
    {
        const Coord3D& p0 = coords[i];
        for (unsigned int j=0; j<3; ++j)
        {
            if (p0[j] < Cmin[j]) Cmin[j] = p0[j];
            if (p0[j] > Cmax[j]) Cmax[j] = p0[j];
        }
    }

    unsigned int nullAxis = 0;
    for (unsigned int j=0; j<3; ++j)
        if (Cmin[j] == Cmax[j])
            nullAxis = j;

    unsigned int axe1 = 0, axe2 = 1;
    if (nullAxis == 0)
    {
        axe1 = 1; axe2 = 2;
    }
    else if (nullAxis == 1)
    {
        axe1 = 2; axe2 = 0;
    }

    SReal Uscale = 1/(SReal)(Cmax[axe1]-Cmin[axe1]);
    SReal Vscale = 1/(SReal)(Cmax[axe2]-Cmin[axe2]);

    for (unsigned int i=0; i<coords.size(); ++i)
    {
        const Coord3D& p0 = coords[i];
        Coord2D& textC = outputs[i];
        textC[0] = (Real)((p0[axe1] - Cmin[axe1])*Uscale);
        textC[1] = (Real)((p0[axe2] - Cmin[axe2])*Vscale);
    }

    _outputCoord.endEdit();
}


template <class DataTypes>
void TextureInterpolation<DataTypes>::draw(const core::visual::VisualParams* vparams )
{
#ifndef SOFA_NO_OPENGL
    // to force update. getX() must have call to endEdit()
    _outputCoord.getValue();

    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S, GL_CLAMP);

    if (drawPotentiels.getValue())
    {
        sofa::defaulttype::Mat<4,4, GLfloat> modelviewM;
        sofa::defaulttype::Vec<3, SReal> sceneMinBBox, sceneMaxBBox;

        const VecCoord& realPotentiels = _inputField.getValue();
        const VecCoord3D& coords = _inputCoords.getValue();

        if (realPotentiels.size() == 0)
            return;

        sofa::helper::vector< Coord1D > potentiels;
        potentiels.resize(realPotentiels.size());

        if (!_scalarField.getValue() && (realPotentiels[0].size() > 1)) // > 1D template and use all dim.
        {
            for (unsigned int i = 0; i<potentiels.size(); ++i)
            {
                Real tmp = 0;
                for (unsigned int j = 0; j<potentiels[0].size(); ++j)
                    tmp += realPotentiels[i][j];

                potentiels[i][0] = tmp/potentiels[0].size();
            }
        }
        else
        {
            for (unsigned int i = 0; i<potentiels.size(); ++i)
                potentiels[i] = realPotentiels[i][0];
        }

        if(potentiels.size() != coords.size())
        {
            msg_error() << "Vector sizes differ." ;
            return;
        }
        unsigned int nbr = potentiels.size();


        sofa::simulation::Node* context = static_cast<sofa::simulation::Node*>(this->getContext());
        sofa::simulation::getSimulation()->computeBBox(context, sceneMinBBox.ptr(), sceneMaxBBox.ptr());

        if (sceneMinBBox[0] > 10000000) // hack when BB is not found
        {
            sceneMaxBBox.assign(1);
            sceneMinBBox.assign(0);
        }

        // Recompute, in case Box has moved.
        float scale = (float)(sceneMaxBBox - sceneMinBBox).norm() * showIndicesScale.getValue();
        sofa::defaulttype::Vector4 color4(1.0,1.0,1.0,1.0);
        for (unsigned int i = 0; i<nbr; i++)
        {
            std::ostringstream oss;
            oss << (float)potentiels[i][0];
            std::string tmp = oss.str();
            const char* s = tmp.c_str();

            vparams->drawTool()->draw3DText(coords[i], scale, color4, s);

        }
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
