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

#ifndef MESHSTEPLOADER_MESHSTEPLOADER_H
#define MESHSTEPLOADER_MESHSTEPLOADER_H


#ifdef __linux__
//#include <config.h>
#endif
#include <BRep_Tool.hxx>
#include <vector>
#include <iostream>
#include <stdlib.h>

#include <BRepLib.hxx>

#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>

#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <TopExp_Explorer.hxx>

#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_CompSolid.hxx>
#include <TopoDS_Solid.hxx>

#include <TopTools_HSequenceOfShape.hxx>

#include <STEPControl_Reader.hxx>
#include <XSControl_WorkSession.hxx>
#include <XSControl_TransferReader.hxx>
#include <StepRepr_RepresentationItem.hxx>
#include <TCollection_HAsciiString.hxx>

#include <TopTools_DataMapOfIntegerShape.hxx>
#include <BRepTools.hxx>
#include <BRepMesh.hxx>
#include <Poly_Triangulation.hxx>
#include <Poly_PolygonOnTriangulation.hxx>
#include <Poly_Array1OfTriangle.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <ShapeAnalysis_Surface.hxx>

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/helper/SVector.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

class MeshSTEPLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshSTEPLoader,sofa::core::loader::MeshLoader);

    MeshSTEPLoader();

    virtual bool load();

    template <class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return BaseLoader::canCreate(obj, context, arg);
    }

protected:
    // Read STEP file and classify the type of object
    bool readSTEP(const char* fileName);

    // Tesselate shape in the case there is one component in the file
    void tesselateShape(const TopoDS_Shape& aShape);

    // Tesselate shape in the case there are several components in the file
    void tesselateMultiShape(const TopoDS_Shape& aShape, const std::vector<TopoDS_Solid>& vshape);

    // Read the name of the shape
    std::string readSolidName(const TopoDS_Solid& aSolid, STEPControl_Reader* aReader);

public:
    // UV point coordinates
    Data<helper::vector<sofa::defaulttype::Vector2> > _uv; ///< UV coordinates

    // Deflection parameter for tesselation
    Data<double> _aDeflection; ///< Deflection parameter for tesselation

    // Boolean for debug mode (display information)
    Data<bool> _debug; ///< if true, print information for debug mode

    // Boolean for keeping duplicated vertices (as vertices are read per face, there are many duplicated vertices)
    // If _keepDuplicate is true, keep the original list of vertices, else remove all duplicated vertices
    Data<bool> _keepDuplicate; ///< if true, keep duplicated vertices

    // Shape number, number of vertices and of triangles of the shape
    Data<helper::vector<helper::fixed_array <unsigned int,3> > > _indicesComponents; ///< Shape # | number of nodes | number of triangles
};

}

}

}

#endif
