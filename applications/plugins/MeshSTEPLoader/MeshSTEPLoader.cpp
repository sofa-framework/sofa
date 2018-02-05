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

#include "MeshSTEPLoader.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/SetDirectory.h>

namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;
using namespace sofa::core::loader;

SOFA_DECL_CLASS(MeshSTEPLoader)

int MeshSTEPLoaderClass = core::RegisterObject("Specific mesh loader for STEP file format (see PluginMeshSTEPLoader.txt for further information).")
        .add< MeshSTEPLoader >();

MeshSTEPLoader::MeshSTEPLoader():MeshLoader()
    , _uv(initData(&_uv, "uv", "UV coordinates"))
    , _aDeflection(initData(&_aDeflection, "deflection", "Deflection parameter for tesselation"))
    , _debug(initData(&_debug, "debug", "if true, print information for debug mode"))
    , _keepDuplicate(initData(&_keepDuplicate, "keepDuplicate", "if true, keep duplicated vertices"))
    , _indicesComponents(initData(&_indicesComponents, "indicesComponents", "Shape # | number of nodes | number of triangles"))
{
    _uv.setPersistent(false);
    _aDeflection.setValue(0.1);
    _debug.setValue(false);
    _keepDuplicate.setValue(true);
    _indicesComponents.setPersistent(false);
}

bool MeshSTEPLoader::load()
{
    sout << "Loading STEP file: " << m_filename << sendl;

    bool fileRead = false;

    // Loading file
    const char* filename = m_filename.getFullPath().c_str();
    std::ifstream file(filename);

    if (!file.good())
    {
        serr << "Error: MeshSTEPLoader: Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    // Reading file
    fileRead = this->readSTEP(filename);
    file.close();

    return fileRead;
}

bool MeshSTEPLoader::readSTEP(const char* fileName)
{
    sout << "MeshSTEPLoader::readSTEP" << sendl;

    STEPControl_Reader * aReader = new STEPControl_Reader;

    // ----------------
    // Import from STEP
    // ----------------

    Handle(TopTools_HSequenceOfShape) aSequence;

    IFSelect_ReturnStatus status = aReader->ReadFile((Standard_CString)fileName);

    const Handle(XSControl_WorkSession)& theSession = aReader->WS();
    const Handle(XSControl_TransferReader)& aTransferReader = theSession->TransferReader();

    if (status == IFSelect_RetDone)
    {
        bool failsonly = true;
        aReader->PrintCheckLoad(failsonly, IFSelect_ItemsByEntity);

        int nbr = aReader->NbRootsForTransfer();
        aReader->PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity);
        for (Standard_Integer n=1; n<=nbr; ++n)
        {
            aReader->TransferRoot(n);
            int nbs = aReader->NbShapes();
            if (nbs > 0)
            {
                aSequence = new TopTools_HSequenceOfShape();
                for (int i=1; i<=nbs; ++i)
                {
                    const TopoDS_Shape& theShape = aReader->Shape(i);
                    aSequence->Append(theShape);

                    Handle(Standard_Transient) anEntity = aTransferReader->EntityFromShapeResult(theShape, 1);

                    if (anEntity.IsNull())
                    {
                        // as just mapped
                        anEntity = aTransferReader->EntityFromShapeResult(theShape, -1);
                    }

                    if (anEntity.IsNull())
                    {
                        // as anything
                        anEntity = aTransferReader->EntityFromShapeResult(theShape, 4);
                    }

                    if (anEntity.IsNull())
                    {
                        sout << "Warning[OpenCascade]: XSInterface_STEPReader::ReadAttributes()\nentity not found" << sendl;
                    }
                    /*else
                    {
                    Handle(StepRepr_RepresentationItem) aReprItem = Handle(StepRepr_RepresentationItem)::DownCast(anEntity);

                    if (aReprItem.IsNull())
                    {
                    serr <<"Error[OpenCascade]: STEPReader::ReadAttributes():\nStepRepr_RepresentationItem Is NULL" << sendl;
                    }
                    else
                    {
                    sout << "Name = " << aReprItem->Name()->ToCString() << sendl;
                    }
                    }*/
                }
            }
        }
    }

    // ----------------
    // Determine shape
    // ----------------

    if (aSequence.IsNull() || !aSequence->Length())
        return false;

    cout << endl << "ATTENTION: Depending on the size of the mesh, loading can take time. Please be patient !" << endl;

    // Boolean to tell if type of shape found (useful in the case of several components in a STEP file)
    bool faceBool = false;

    for (int i=1; i<=aSequence->Length(); ++i)
    {
        TopoDS_Face aFace; aFace.Nullify();
        TopoDS_Edge anEdge; anEdge.Nullify();
        TopoDS_Vertex aVertex; aVertex.Nullify();
        TopoDS_CompSolid aCompSolid; aCompSolid.Nullify();
        TopoDS_Solid aSolid; aSolid.Nullify();
        TopoDS_Wire aWire; aWire.Nullify();
        TopoDS_Compound aCompound; aCompound.Nullify();

        try
        {
            aFace = TopoDS::Face(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aFace.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Face" << sendl;
        }

        try
        {
            aWire = TopoDS::Wire(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aWire.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Wire" << sendl;
        }

        try
        {
            anEdge = TopoDS::Edge(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!anEdge.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Edge" << sendl;
        }

        try
        {
            aVertex = TopoDS::Vertex(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aVertex.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Vertex" << sendl;
        }

        try
        {
            aSolid = TopoDS::Solid(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aSolid.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Solid" << sendl;

            std::string nameShape = readSolidName(aSolid, aReader);
            sout << "============================" << sendl;
            sout << "Name of the shape\t\t -> shape #" << sendl;
            sout << "--------------------------------------------------------" << sendl;
            sout << nameShape << "\t\t -> 0" << sendl;
            sout << "============================" << sendl;

            tesselateShape(aSolid);
        }

        try
        {
            aCompSolid = TopoDS::CompSolid(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aCompSolid.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: CompSolid" << sendl;
        }

        try
        {
            aCompound = TopoDS::Compound(aSequence->Value(i));
        }
        catch(Standard_Failure)
        {
        }
        if (!aCompound.IsNull() && !faceBool)
        {
            faceBool = true;
            sout << "[" << i << "]: Compound" << sendl;

            sout << "============================" << sendl;
            sout << "Name of the shape\t\t -> shape #" << sendl;
            sout << "--------------------------------------------------------" << sendl;

            TopExp_Explorer aExpSolid;
            std::vector<TopoDS_Solid> vshape;
            int numShape = 0;
            for (aExpSolid.Init(aCompound, TopAbs_SOLID); aExpSolid.More(); aExpSolid.Next())
            {
                const TopoDS_Solid& solid = TopoDS::Solid(aExpSolid.Current());
                vshape.push_back(solid);

                std::string nameShape = readSolidName(solid, aReader);
                sout << nameShape << "\t\t -> " << numShape << sendl;
                ++numShape;
            }

            sout << "============================" << sendl;

            tesselateMultiShape(aCompound, vshape);
        }
    }

    return true;
}

void MeshSTEPLoader::tesselateShape(const TopoDS_Shape& aShape)
{
    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());

    helper::vector< Edge >& my_edges = *(edges.beginEdit());
    helper::vector< Triangle >& my_triangles = *(triangles.beginEdit());

    helper::vector<sofa::defaulttype::Vector2>& my_uv = *(_uv.beginEdit());

    helper::vector<helper::fixed_array <unsigned int,3> >& my_indicesComponents = *(_indicesComponents.beginEdit());

    BRepTools::Clean(aShape);
    BRepMesh::Mesh(aShape, _aDeflection.getValue());

    Standard_Integer aCount = 0;
    Standard_Integer aNumOfNodes = 0;
    Standard_Integer aNumOfTriangles = 0;

    TopExp_Explorer aExpFace, aExpEdge;

    int aCount2 = 0;

    // Correspondant vertex indices between the list with duplicated vertices kept and the list with duplicated vertices removed
    std::map<int,int> nodeIndex;

    for (aExpFace.Init(aShape, TopAbs_FACE); aExpFace.More(); aExpFace.Next())
    {
        ++aCount;

        const TopoDS_Face& aFace = TopoDS::Face(aExpFace.Current());
        TopLoc_Location aLocation;

        // See Docs_tutorials/OpenCascade/ModelingData/html/classBRep__Tool.html
        const Handle_Poly_Triangulation& aTr = BRep_Tool::Triangulation(aFace, aLocation);

        if (!aTr.IsNull())
        {
            const TColgp_Array1OfPnt& aNodes = aTr->Nodes();
            Standard_Integer aNbOfNodesOfFace = aTr->NbNodes();
            Standard_Integer aLower = aNodes.Lower();
            const Poly_Array1OfTriangle& triangles = aTr->Triangles();
            Standard_Integer aNbOfTrianglesOfFace = aTr->NbTriangles();

            // Point coordinates
            for (Standard_Integer i=1; i<aNbOfNodesOfFace+1; ++i)
            {
                const gp_Pnt& aPnt = aNodes(i).Transformed(aLocation);
                my_positions.push_back(Vector3(aPnt.X(), aPnt.Y(), aPnt.Z()));

                // Remove duplicated vertex
                if (!_keepDuplicate.getValue())
                {
                    for (size_t k=0; k<my_positions.size()-1; ++k)
                    {
                        nodeIndex[aCount2] = my_positions.size()-1;
                        if (my_positions[k] == Vector3(aPnt.X(), aPnt.Y(), aPnt.Z()))
                        {
                            my_positions.pop_back();
                            nodeIndex[aCount2] = k;
                            break;
                        }
                    }
                }

                ++aCount2;
            }

            for (int i=1; i<aNbOfNodesOfFace+1; ++i)
            {
                const gp_Pnt& aPnt = aNodes(i).Transformed(aLocation);

                // get face as surface
                const Handle(Geom_Surface)& surface = BRep_Tool::Surface(aFace);
                // create shape analysis object
                ShapeAnalysis_Surface sas(surface);
                // get UV of point on surface
                const gp_Pnt2d& uv = sas.ValueOfUV(aPnt, 0.01);
                my_uv.push_back(Vector2(uv.X(), uv.Y()));
            }

            // if(aCount == aNumOfFace)
            {
                for (aExpEdge.Init(aFace, TopAbs_EDGE); aExpEdge.More(); aExpEdge.Next())
                {
                    const TopoDS_Edge& aEdge = TopoDS::Edge(aExpEdge.Current());

                    if (!aEdge.IsNull())
                    {
                        const Handle_Poly_PolygonOnTriangulation& aPol = BRep_Tool::PolygonOnTriangulation(aEdge, aTr, aEdge.Location());

                        if (!aPol.IsNull())
                        {
                            const TColStd_Array1OfInteger& aNodesOfPol = aPol->Nodes();

                            if (_debug.getValue())
                            {
                                Standard_Integer aNbOfNodesOfEdge = aPol->NbNodes();
                                sout << "Number of nodes of the edge = " << aNbOfNodesOfEdge << sendl;
                                sout << "Number of nodes of the face = " << aNbOfNodesOfFace << sendl;
                                sout << "Number of triangles of the face = " << aNbOfTrianglesOfFace << sendl;
                            }

                            // Edge
                            Standard_Integer aLower = aNodesOfPol.Lower(), anUpper = aNodesOfPol.Upper();
                            for (int i=aLower; i<anUpper ; ++i)
                            {
                                int nodesOfPol_1 = aNodesOfPol(i) - aLower + aNumOfNodes, nodesOfPol_2 = aNodesOfPol(i+1) - aLower + aNumOfNodes;
                                if (!_keepDuplicate.getValue())
                                    addEdge(&my_edges, Edge(nodeIndex[nodesOfPol_1], nodeIndex[nodesOfPol_2]));
                                else
                                    addEdge(&my_edges, Edge(nodesOfPol_1, nodesOfPol_2));
                            }
                        }
                    }
                }
            }

            // Triangle
            Standard_Integer n1, n2, n3;

            const TopAbs_Orientation& orientation = aFace.Orientation();
            for (Standard_Integer nt=1 ; nt<aNbOfTrianglesOfFace+1 ; ++nt)
            {
                // Check orientation of the triangle
                if (orientation == TopAbs_FORWARD)
                {
                    triangles(nt).Get(n1, n2, n3);
                }
                else
                {
                    triangles(nt).Get(n2, n1, n3);
                }
                n1 -= aLower - aNumOfNodes; n2 -= aLower - aNumOfNodes; n3 -= aLower - aNumOfNodes;
                if (!_keepDuplicate.getValue())
                    addTriangle(&my_triangles, Triangle(nodeIndex[n1], nodeIndex[n2], nodeIndex[n3]));
                else
                    addTriangle(&my_triangles, Triangle(n1, n2, n3));
            }

            aNumOfNodes += aNbOfNodesOfFace;
            aNumOfTriangles += aNbOfTrianglesOfFace;
        }
        else
        {
            serr << "Can't compute a triangulation on face " << aCount << sendl;
        }
    }

    my_indicesComponents.push_back(helper::fixed_array <unsigned int,3>(0, aNumOfNodes, aNumOfTriangles));

    cout << "Finished loading mesh" << endl;

    sout << "Number of nodes of the shape = " << aNumOfNodes << sendl;
    sout << "Number of triangles of the shape " << aNumOfTriangles << sendl;

    positions.endEdit();
    edges.endEdit();
    triangles.endEdit();

    _uv.endEdit();

    _indicesComponents.endEdit();
}

void MeshSTEPLoader::tesselateMultiShape(const TopoDS_Shape& aShape, const std::vector<TopoDS_Solid>& vshape)
{
    helper::vector<sofa::defaulttype::Vector3>& my_positions = *(positions.beginEdit());

    helper::vector< Edge >& my_edges = *(edges.beginEdit());
    helper::vector< Triangle >& my_triangles = *(triangles.beginEdit());

    helper::vector<sofa::defaulttype::Vector2>& my_uv = *(_uv.beginEdit());

    helper::vector<helper::fixed_array <unsigned int,3> >& my_indicesComponents = *(_indicesComponents.beginEdit());

    BRepTools::Clean(aShape);
    BRepMesh::Mesh(aShape, _aDeflection.getValue());

    Standard_Integer aCount = 0;
    Standard_Integer aNumOfNodes = 0;
    Standard_Integer aNumOfTriangles = 0;

    TopExp_Explorer aExpFace, aExpEdge;

    // Tesselate each component
    for (size_t numShape=0; numShape<vshape.size(); ++numShape)
    {
        Standard_Integer aNumOfNodesShape = 0;
        Standard_Integer aNumOfTrianglesShape = 0;

        for (aExpFace.Init(vshape[numShape], TopAbs_FACE); aExpFace.More(); aExpFace.Next())
        {
            ++aCount;

            const TopoDS_Face& aFace = TopoDS::Face(aExpFace.Current());
            TopLoc_Location aLocation;

            // See Docs_tutorials/OpenCascade/ModelingData/html/classBRep__Tool.html
            const Handle_Poly_Triangulation& aTr = BRep_Tool::Triangulation(aFace, aLocation);

            if (!aTr.IsNull())
            {
                const TColgp_Array1OfPnt& aNodes = aTr->Nodes();
                Standard_Integer aNbOfNodesOfFace = aTr->NbNodes();
                Standard_Integer aLower = aNodes.Lower();
                const Poly_Array1OfTriangle& triangles = aTr->Triangles();
                Standard_Integer aNbOfTrianglesOfFace = aTr->NbTriangles();

                // Point coordinates
                for (Standard_Integer i=1; i<aNbOfNodesOfFace+1; ++i)
                {
                    const gp_Pnt& aPnt = aNodes(i).Transformed(aLocation);
                    my_positions.push_back(Vector3(aPnt.X(), aPnt.Y(), aPnt.Z()));
                }

                for (int i=1; i<aNbOfNodesOfFace+1; ++i)
                {
                    const gp_Pnt& aPnt = aNodes(i).Transformed(aLocation);

                    // get face as surface
                    const Handle(Geom_Surface)& surface = BRep_Tool::Surface(aFace);
                    // create shape analysis object
                    ShapeAnalysis_Surface sas(surface);
                    // get UV of point on surface
                    const gp_Pnt2d& uv = sas.ValueOfUV(aPnt, 0.01);
                    my_uv.push_back(Vector2(uv.X(), uv.Y()));
                }

                //if (aCount == aNumOfFace)
                {
                    for (aExpEdge.Init(aFace, TopAbs_EDGE); aExpEdge.More(); aExpEdge.Next())
                    {
                        const TopoDS_Edge& aEdge = TopoDS::Edge(aExpEdge.Current());

                        if (!aEdge.IsNull())
                        {
                            const Handle_Poly_PolygonOnTriangulation& aPol = BRep_Tool::PolygonOnTriangulation(aEdge, aTr, aEdge.Location());

                            if (!aPol.IsNull())
                            {
                                const TColStd_Array1OfInteger& aNodesOfPol = aPol->Nodes();

                                if (_debug.getValue())
                                {
                                    Standard_Integer aNbOfNodesOfEdge = aPol->NbNodes();
                                    sout << "Number of nodes of the edge = " << aNbOfNodesOfEdge << sendl;
                                    sout << "Number of nodes of the face = " << aNbOfNodesOfFace << sendl;
                                    sout << "Number of triangles of the face = " << aNbOfTrianglesOfFace << sendl;
                                }

                                // Edge
                                Standard_Integer aLower = aNodesOfPol.Lower(), anUpper = aNodesOfPol.Upper();
                                for (int i=aLower; i<anUpper ; ++i)
                                {
                                    int nodesOfPol_1 = aNodesOfPol(i) - aLower + aNumOfNodes, nodesOfPol_2 = aNodesOfPol(i+1) - aLower + aNumOfNodes;
                                    addEdge(&my_edges, Edge(nodesOfPol_1, nodesOfPol_2));
                                }
                            }
                        }
                    }
                }

                // Triangle
                Standard_Integer n1, n2, n3;

                const TopAbs_Orientation& orientation = aFace.Orientation();
                for (Standard_Integer nt=1 ; nt<aNbOfTrianglesOfFace+1 ; ++nt)
                {
                    // Check orientation of the triangle
                    if (orientation == TopAbs_FORWARD)
                    {
                        triangles(nt).Get(n1, n2, n3);
                    }
                    else
                    {
                        triangles(nt).Get(n2, n1, n3);
                    }
                    n1 -= aLower - aNumOfNodes; n2 -= aLower - aNumOfNodes; n3 -= aLower - aNumOfNodes;
                    addTriangle(&my_triangles, Triangle(n1, n2, n3));
                }

                aNumOfNodes += aNbOfNodesOfFace;
                aNumOfTriangles += aNbOfTrianglesOfFace;

                aNumOfNodesShape += aNbOfNodesOfFace;
                aNumOfTrianglesShape += aNbOfTrianglesOfFace;
            }
            else
            {
                serr << "Can't compute a triangulation on face " << aCount << sendl;
            }
        }

        my_indicesComponents.push_back(helper::fixed_array <unsigned int,3>(numShape, aNumOfNodesShape, aNumOfTrianglesShape));
    }

    cout << "Finished loading mesh" << endl;

    sout << "Number of nodes of the shape = " << aNumOfNodes << sendl;
    sout << "Number of triangles of the shape " << aNumOfTriangles << sendl;

    positions.endEdit();
    edges.endEdit();
    triangles.endEdit();

    _uv.endEdit();

    _indicesComponents.endEdit();
}

std::string MeshSTEPLoader::readSolidName(const TopoDS_Solid& aSolid, STEPControl_Reader* aReader)
{
    const Handle(XSControl_WorkSession)& theSession = aReader->WS();
    const Handle(XSControl_TransferReader)& aTransferReader = theSession->TransferReader();

    Handle(Standard_Transient) anEntity = aTransferReader->EntityFromShapeResult(aSolid, 1);

    if (anEntity.IsNull())
    {
        // as just mapped
        anEntity = aTransferReader->EntityFromShapeResult(aSolid, -1);
    }

    if (anEntity.IsNull())
    {
        // as anything
        anEntity = aTransferReader->EntityFromShapeResult(aSolid, 4);
    }

    if (anEntity.IsNull())
    {
        sout << "Warning[OpenCascade]: XSInterface_STEPReader::ReadAttributes()\nentity not found" << sendl;
    }
    else
    {
        const Handle(StepRepr_RepresentationItem)& aReprItem = Handle(StepRepr_RepresentationItem)::DownCast(anEntity);

        if (aReprItem.IsNull())
        {
            serr << "Error[OpenCascade]: STEPReader::ReadAttributes():\nStepRepr_RepresentationItem Is NULL" << sendl;
        }
        else
        {
            return aReprItem->Name()->ToCString();
        }
    }
    return "";
}

}

}

}
