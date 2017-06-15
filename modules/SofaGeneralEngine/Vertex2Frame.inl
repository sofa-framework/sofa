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
#ifndef SOFA_COMPONENT_ENGINE_VERTEX2FRAME_INL
#define SOFA_COMPONENT_ENGINE_VERTEX2FRAME_INL

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SofaGeneralEngine/Vertex2Frame.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Quat.h>

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
Vertex2Frame<DataTypes>::Vertex2Frame():
    vertices(initData(&vertices,"position","Vertices of the mesh loaded"))
    , texCoords(initData(&texCoords,"texCoords","TexCoords of the mesh loaded"))
    , normals(initData(&normals,"normals","Normals of the mesh loaded"))
    , frames( initData (&frames, "frames", "Frames at output") )
	, useNormals( initData (&useNormals, true, "useNormals", "Use normals to compute the orientations; if disabled the direction of the x axisof a vertice is the one from this vertice to the next one") )
    , invertNormals( initData (&invertNormals, false, "invertNormals", "Swap normals") )
    , rotation( initData (&rotation, 0, "rotation", "Apply a local rotation on the frames. If 0 a x-axis rotation is applied. If 1 a y-axis rotation is applied, If 2 a z-axis rotation is applied.") )
    , rotationAngle( initData (&rotationAngle, 0.0, "rotationAngle", "Angle rotation") )
{
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::init()
{
    addInput(&vertices);
    addInput(&texCoords);
    addInput(&normals);
    addInput(&rotation);
    addInput(&rotationAngle);

    addOutput(&frames);

    setDirtyValue();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::update()
{
    using namespace sofa::defaulttype;

    const helper::vector<Vector3>& fVertices = vertices.getValue();
    const helper::vector<Vector3>& fNormals = normals.getValue();
    unsigned int nbVertices = fVertices.size();

    if (nbVertices <= 0 || fNormals.size() <=0)
    {
        serr << "Vertex2Frame : no vertices or normals found..." << sendl;
        return ;
    }

    texCoords.updateIfDirty();
    rotation.updateIfDirty();
    rotationAngle.updateIfDirty();

    cleanDirty();

    VecCoord& fFrames = *(frames.beginEdit());
    fFrames.resize(nbVertices);

	if(useNormals.getValue()) {
		for (unsigned int i=0 ; i<nbVertices ; i++)
		{
			Quat q, q2;
			Vector3 zAxis = (!invertNormals.getValue()) ? fNormals[i] : -fNormals[i];
			zAxis.normalize();
			Vector3 xAxis;
			Vector3 yAxis(1.0, 0.0, 0.0);
			//if ( fabs(dot(yAxis, zAxis)) > 0.7)
			//	yAxis = Vector3(0.0, 0.0, 1.0);

			xAxis = yAxis.cross(zAxis);
			xAxis.normalize();
			yAxis = zAxis.cross(xAxis);
			yAxis.normalize();

			// compute frame rotation
			Vector3 rotationAxis;
			switch(rotation.getValue())
			{
			case 0 :
				rotationAxis = xAxis;
				break;
			case 1 :
				rotationAxis = yAxis;
				break;
			case 2 :
				rotationAxis = zAxis;
				break;
			default:
				break;
			}
			q2 = q2.axisToQuat(rotationAxis, (rotationAngle.getValue()*M_PI)/180);
			// frame rotation computed

			fFrames[i].getOrientation() = q2*q.createQuaterFromFrame(xAxis, yAxis, zAxis);
			fFrames[i].getCenter() = fVertices[i];
		}
		frames.endEdit();
	} 
	else {
		if (nbVertices <= 1)
		{
			serr << "Vertex2Frame : no enough vertices to compute the orientations..." << sendl;
			return ;
		}

		for (unsigned int i=0 ; i<(nbVertices-1) ; i++)
		{
			Quat q, q2;
			Vector3 xAxis = fVertices[i+1] - fVertices[i];
			xAxis.normalize();
			Vector3 yAxis;
			Vector3 zAxis(1.0, 0.0, 0.0);
			//if ( fabs(dot(zAxis, xAxis)) > 0.707)
			//	zAxis = Vector3(0.0, 0.0, 1.0);

			yAxis = zAxis.cross(xAxis);
			yAxis.normalize();
			zAxis = xAxis.cross(yAxis);
			zAxis.normalize();

			// compute frame rotation
			Vector3 rotationAxis;
			switch(rotation.getValue())
			{
			case 0 :
				rotationAxis = xAxis;
				break;
			case 1 :
				rotationAxis = yAxis;
				break;
			case 2 :
				rotationAxis = zAxis;
				break;
			default:
				break;
			}
			q2 = q2.axisToQuat(rotationAxis, (rotationAngle.getValue()*M_PI)/180);
			// frame rotation computed

			fFrames[i].getOrientation() = q2*q.createQuaterFromFrame(xAxis, yAxis, zAxis);
			fFrames[i].getCenter() = fVertices[i];
		}
		fFrames[nbVertices-1].getOrientation() = fFrames[nbVertices-2].getOrientation();
		fFrames[nbVertices-1].getCenter() = fVertices[nbVertices-1];
		frames.endEdit(); 
	}
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
