/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include "SofaPhysicsAPI.h"

#include <sofa/component/visual/VisualModelImpl.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/core/visual/Shader.h>

class SOFA_SOFAPHYSICSAPI_API SofaPhysicsOutputMesh::Impl
{
public:

    Impl();
    ~Impl();

    const std::string& getNameStr() const; ///< return SOFA output mesh component name
    const char* getName(); ///< (non-unique) name of this object
    ID          getID();   ///< unique ID of this object

    unsigned int getNbVertices(); ///< number of vertices
    const Real* getVPositions();  ///< vertices positions (Vec3)
    int getVPositions(Real* values); ///< get the positions/vertices of this mesh inside ouput @param values, of type Real[ 3*nbVertices ]
    const Real* getVNormals();    ///< vertices normals   (Vec3)
    int getVNormals(Real* values); ///< get the normals per vertex of this mesh inside ouput @param values, of type Real[ 3*nbVertices ]
    const Real* getVTexCoords();  ///< vertices UVs       (Vec2)
    int getVTexCoords(Real* values); ///< get the texture coordinates (UV) per vertex of this mesh inside ouput @param values, of type Real[ 2*nbVertices ]
    int getTexCoordRevision();    ///< changes each time texture coord data are updated
    int getVerticesRevision();    ///< changes each time vertices data are updated
    
    unsigned int getNbVAttributes();                    ///< number of vertices attributes
    unsigned int getNbAttributes(int index);            ///< number of the attributes in specified vertex attribute 
    const char*  getVAttributeName(int index);          ///< vertices attribute name
    int          getVAttributeSizePerVertex(int index); ///< vertices attribute #
    const Real*  getVAttributeValue(int index);         ///< vertices attribute (Vec#)
    int          getVAttributeRevision(int index);      ///< changes each time vertices attribute is updated

    unsigned int getNbLines(); ///< number of lines
    const Index* getLines();   ///< lines topology (2 indices / line)
    int getLinesRevision();    ///< changes each time lines data is updated

    unsigned int getNbTriangles(); ///< number of triangles
    const Index* getTriangles();   ///< triangles topology (3 indices / triangle)
    int getTriangles(int* values); ///< get the triangle topology inside ouput @param values, of type int[ 3*nbTriangles ]
    int getTrianglesRevision();    ///< changes each time triangles data is updated

    unsigned int getNbQuads(); ///< number of quads
    const Index* getQuads();   ///< quads topology (4 indices / quad)
    int getQuads(int* values); ///< get the quad topology inside ouput @param values, of type int[ 4*nbQuads ]
    int getQuadsRevision();    ///< changes each time quads data is updated

    typedef sofa::core::visual::VisualModel SofaVisualOutputMesh;
    typedef sofa::component::visual::VisualModelImpl SofaOutputMesh;
    typedef SofaOutputMesh::DataTypes DataTypes;
    typedef SofaOutputMesh::Coord Coord;
    typedef SofaOutputMesh::Deriv Deriv;
    typedef SofaOutputMesh::TexCoord TexCoord;
    typedef SofaOutputMesh::VisualTriangle Triangle;
    typedef SofaOutputMesh::VisualQuad Quad;
    typedef sofa::core::visual::ShaderElement SofaVAttribute;

protected:
    SofaOutputMesh::SPtr sObj;
    sofa::type::vector<SofaVAttribute::SPtr> sVA;

    /// Default static name in case component creation failed
    std::string defaultName = "None";

public:
    SofaOutputMesh* getObject() { return sObj.get(); }
    void setObject(SofaOutputMesh* o);
};
