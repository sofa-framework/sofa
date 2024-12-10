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
#include <sofa/component/engine/transform/Vertex2Frame.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/Quat.h>

namespace sofa::component::engine::transform
{

template <class DataTypes>
const Real_t<DataTypes> Vertex2Frame<DataTypes>::EPSILON = std::numeric_limits<Real_t<DataTypes>>::epsilon();

template <class DataTypes>
Vertex2Frame<DataTypes>::Vertex2Frame():
    d_vertices(initData(&d_vertices,"position","Vertices of the mesh loaded"))
    , d_texCoords(initData(&d_texCoords,"texCoords","TexCoords of the mesh loaded"))
    , d_normals(initData(&d_normals,"normals","Normals of the mesh loaded"))
    , d_frames( initData (&d_frames, "frames", "Frames at output") )
    , d_useNormals( initData (&d_useNormals, true, "useNormals", "Use normals to compute the orientations; if disabled the direction of the x axisof a vertice is the one from this vertice to the next one") )
    , d_invertNormals( initData (&d_invertNormals, false, "invertNormals", "Swap normals") )
    , d_rotation( initData (&d_rotation, 0, "rotation", "Apply a local rotation on the frames. If 0 a x-axis rotation is applied. If 1 a y-axis rotation is applied, If 2 a z-axis rotation is applied.") )
    , d_rotationAngle( initData (&d_rotationAngle, 0.0, "rotationAngle", "Angle rotation") )
{
    addInput(&d_vertices);
    addInput(&d_texCoords);
    addInput(&d_normals);
    addInput(&d_rotation);
    addInput(&d_rotationAngle);

    addOutput(&d_frames);
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::init()
{
    setDirtyValue();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void Vertex2Frame<DataTypes>::doUpdate()
{
    const type::vector<CPos>& fVertices = d_vertices.getValue();
    const type::vector<CPos>& fNormals = d_normals.getValue();
    unsigned int nbVertices = fVertices.size();

    if (nbVertices <= 0)
    {
        /// Here I set the message to info because there is scenario of usage that initially
        /// does not have vertices and after few frame, vertices pop-up and we don't want
        /// to have message in that case. Using msg_info() allow user to control what it displayed.
        /// with the printLog='true/false' attribute.
        msg_info(this) << "Vertex2Frame : no vertices found. Component will not compute anything";
        return ;
    }

    VecCoord& fFrames = *(d_frames.beginEdit());
    fFrames.resize(nbVertices);

    if(d_useNormals.getValue()) {
        if (fNormals.size() <=0)
        {
            msg_error(this) << "Vertex2Frame : no normals found. Component will not compute anything";
            return;
        }

        for (unsigned int i=0 ; i<nbVertices ; i++)
        {
            CPos zAxis = (!d_invertNormals.getValue()) ? fNormals[i] : -fNormals[i];
            zAxis.normalize();

            CPos xAxis;
            CPos yAxis(0.0, 1.0, 0.0);
            if ( 1.0f - fabs(dot(yAxis, zAxis)) <= EPSILON)
                yAxis = CPos(0.0, 0.0, 1.0);

            xAxis = yAxis.cross(zAxis);
            xAxis.normalize();
            yAxis = zAxis.cross(xAxis);
            yAxis.normalize();


            fFrames[i].getOrientation() = computeOrientation(xAxis, yAxis, zAxis);
            fFrames[i].getCenter() = fVertices[i];
        }
        d_frames.endEdit();
    } 
    else {
        if (nbVertices <= 1)
        {
            msg_error(this) << "Vertex2Frame : no enough vertices to compute the orientations. Component will not compute anything" ;
            return ;
        }

        for (unsigned int i=0 ; i<(nbVertices-1) ; i++)
        {
            CPos xAxis = fVertices[i+1] - fVertices[i];
            xAxis.normalize();

            CPos yAxis;
            CPos zAxis(1.0, 0.0, 0.0);
            if ( 1.0f - fabs(dot(zAxis, xAxis)) <= EPSILON)
                zAxis = CPos(0.0, 0.0, 1.0);

            yAxis = zAxis.cross(xAxis);
            yAxis.normalize();
            zAxis = xAxis.cross(yAxis);
            zAxis.normalize();

            fFrames[i].getOrientation() = computeOrientation(xAxis, yAxis, zAxis);
            fFrames[i].getCenter() = fVertices[i];
        }
        fFrames[nbVertices-1].getOrientation() = fFrames[nbVertices-2].getOrientation();
        fFrames[nbVertices-1].getCenter() = fVertices[nbVertices-1];
        d_frames.endEdit();
    }
}

template <class DataTypes>
type::Quat<SReal>  Vertex2Frame<DataTypes>::computeOrientation(const CPos &xAxis, const CPos &yAxis, const CPos &zAxis)
{
    sofa::type::Quat<SReal> q, q2;

    // compute frame rotation
    CPos rotationAxis;
    switch(d_rotation.getValue())
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
    q2 = q2.axisToQuat(rotationAxis, (d_rotationAngle.getValue()*M_PI)/180);

    return q2*q.createQuaterFromFrame(xAxis, yAxis, zAxis);
}


} //namespace sofa::component::engine::transform
