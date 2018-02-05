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
/*
 * VTKExporter.cpp
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#include "VTKExporter.h"

#include <sstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(VTKExporter)

int VTKExporterClass = core::RegisterObject("Save State vectors from file at each timestep")
        .add< VTKExporter >();

VTKExporter::VTKExporter()
    : stepCounter(0), outfile(NULL)
    , vtkFilename( initData(&vtkFilename, "filename", "output VTK file name"))
    , fileFormat( initData(&fileFormat, (bool) true, "XMLformat", "Set to true to use XML format"))
    , position( initData(&position, "position", "points position (will use points from topology or mechanical state if this is empty)"))
    , writeEdges( initData(&writeEdges, (bool) true, "edges", "write edge topology"))
    , writeTriangles( initData(&writeTriangles, (bool) false, "triangles", "write triangle topology"))
    , writeQuads( initData(&writeQuads, (bool) false, "quads", "write quad topology"))
    , writeTetras( initData(&writeTetras, (bool) false, "tetras", "write tetra topology"))
    , writeHexas( initData(&writeHexas, (bool) false, "hexas", "write hexa topology"))
    , dPointsDataFields( initData(&dPointsDataFields, "pointsDataFields", "Data to visualize (on points)"))
    , dCellsDataFields( initData(&dCellsDataFields, "cellsDataFields", "Data to visualize (on cells)"))
    , exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
    , exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
    , exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
    , overwrite( initData(&overwrite, false, "overwrite", "overwrite the file, otherwise create a new file at each export, with suffix in the filename"))
{
}

VTKExporter::~VTKExporter()
{
    if (outfile)
        delete outfile;
}

void VTKExporter::init()
{    
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topology);
    context->get(mstate);

    // if not set, set the printLog to true to read the msg_info()
    if(!this->f_printLog.isSet())
        f_printLog.setValue(true);

    if (!topology)
    {
        msg_error() << "VTKExporter : error, no topology ." ;
        return;
    }
    else
    {
        msg_info() << "VTKExporter: found topology " << topology->getName() ;
    }

    nbFiles = 0;

    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    if (!pointsData.empty())
    {
        fetchDataFields(pointsData, pointsDataObject, pointsDataField, pointsDataName);
    }
    if (!cellsData.empty())
    {
        fetchDataFields(cellsData, cellsDataObject, cellsDataField, cellsDataName);
    }

}

void VTKExporter::fetchDataFields(const helper::vector<std::string>& strData, helper::vector<std::string>& objects, helper::vector<std::string>& fields, helper::vector<std::string>& names)
{
    for (unsigned int i=0 ; i<strData.size() ; i++)
    {
        std::string str = strData[i];
        std::string name, objectName, dataFieldName;
        std::string::size_type loc = str.find_first_of('=');
        if (loc != std::string::npos)
        {
            name = str.substr(0,loc);
            str = str.substr(loc+1);
        }
        if (str.at(0) == '@') // ignore @ prefix
            str = str.substr(1);

        loc = str.find_last_of('.');
        if ( loc != std::string::npos)
        {
            objectName = str.substr(0, loc);
            dataFieldName = str.substr(loc+1);
        }
        else
        {
            msg_error() << "VTKExporter : error while parsing dataField names" ;
            continue;
        }
        if (name.empty()) name = dataFieldName;
        objects.push_back(objectName);
        fields.push_back(dataFieldName);
        names.push_back(name);
    }
}

void VTKExporter::writeData(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        if (obj)
        {
            field = obj->findData(fields[i]);
        }

        if (!obj || !field)
        {
            if (!obj)
                msg_error() << "VTKExporter : error while fetching data field '" << msgendl
                            << fields[i] << "' of object '" << objects[i] << msgendl
                            << "', check object name"  << msgendl;
            else if (!field)
                msg_error() << "VTKExporter : error while fetching data field " << msgendl
                            << fields[i] << " of object '" << objects[i] << msgendl
                            << "', check field name " << msgendl;
        }
        else
        {
            //Scalars
            std::string line;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
            {
                line = "float 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
            {
                line = "double 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2f > >* > (field))
            {
                line = "float 2";
                sizeSeg = 2;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2d > >* >(field))
            {
                line = "double 2";
                sizeSeg = 2;
            }

            //if this is a scalar
            if (!line.empty())
            {
                *outfile << "SCALARS" << " " << names[i] << " ";
                *outfile << line << std::endl;
                *outfile << "LOOKUP_TABLE default" << std::endl;
            }
            else
            {
                //Vectors
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                {
                    line = "float";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                {
                    line = "double";
                    sizeSeg = 3;
                }
                *outfile << "VECTORS" << " " << names[i] << " ";
                *outfile << line << std::endl;
            }

            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << std::endl;


        }
    }
}

void VTKExporter::writeDataArray(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        if (obj)
        {
            field = obj->findData(fields[i]);
        }

        if (!obj || !field)
        {
            if (!obj)
                msg_error() << "VTKExporter : error while fetching data field '" << msgendl
                            << fields[i] << "' of object '" << objects[i] << msgendl
                            << "', check object name" << msgendl;
            else if (!field)
                msg_error()  << "VTKExporter : error while fetching data field " << msgendl
                             << fields[i] << " of object '" << objects[i] << msgendl
                             << "', check field name " << msgendl;
        }
        else
        {
            //Scalars
            std::string type;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<int> >* >(field))
            {
                type = "Int32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<unsigned int> >* >(field))
            {
                type = "UInt32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
            {
                type = "Float32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
            {
                type = "Float64";
                sizeSeg = 1;
            }

            //Vectors
            if (type.empty())
            {
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec1f> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec1d> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2f> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 2;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec2d> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 2;
                }

                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                {
                    type = "Float32";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 3;
                }
            }
            *outfile << "        <DataArray type=\""<< type << "\" Name=\"" << names[i];
            if(sizeSeg > 1)
                *outfile << "\" NumberOfComponents=\"" << sizeSeg;
            *outfile << "\" format=\"ascii\">" << std::endl;
            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << "        </DataArray>" << std::endl;
        }
    }
}


std::string VTKExporter::segmentString(std::string str, unsigned int n)
{
    std::string::size_type loc = 0;
    unsigned int i=0;

    loc = str.find(' ', 0);

    while(loc != std::string::npos )
    {
        i++;
        if (i == n)
        {
            str[loc] = '\n';
            i=0;
        }
        loc = str.find(' ', loc+1);
    }

    return str;
}


void VTKExporter::writeVTKSimple()
{
    std::string filename = vtkFilename.getFullPath();

    std::ostringstream oss;
    oss << "_" << nbFiles;

    if (filename.size() > 3) {
        std::string ext;
        std::string baseName;
        if (filename.substr(filename.size()-4)==".vtu") {
            ext = ".vtu";
            baseName = filename.substr(0, filename.size()-4);
        }

        if (filename.substr(filename.size()-4)==".vtk") {
            ext = ".vtk";
            baseName = filename.substr(0, filename.size()-4);
        }

        /// no extension given => default "vtu"
        if (ext == "") {
            ext = ".vtu";
            baseName = filename;
        }

        if (overwrite.getValue())
            filename = baseName + ext;
        else
            filename = baseName + oss.str() + ext;

    }

    /*if ( filename.size() > 3 && filename.substr(filename.size()-4)==".vtu")
    {
        if (!overwrite.getValue())
            filename = filename.substr(0,filename.size()-4) + oss.str() + ".vtu";
    }
    else
    {
        if (!overwrite.getValue())
            filename += oss.str();
        filename += ".vtu";
    }*/

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        delete outfile;
        outfile = NULL;
        return;
    }

    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = (!pointsPos.empty()) ? pointsPos.size() : topology->getNbPoints();

    //Write header
    *outfile << "# vtk DataFile Version 2.0" << std::endl;

    //write Title
    *outfile << "Exported VTK file" << std::endl;

    //write Data type
    *outfile << "ASCII" << std::endl;

    *outfile << std::endl;

    //write dataset (geometry, unstructured grid)
    *outfile << "DATASET " << "UNSTRUCTURED_GRID" << std::endl;

    *outfile << "POINTS " << nbp << " float" << std::endl;
    //write Points
    if (!pointsPos.empty())
    {
        for (int i=0 ; i<nbp; i++)
        {
            *outfile << pointsPos[i] << std::endl;
        }
    }
    else if (mstate && mstate->getSize() == (size_t)nbp)
    {
        for (size_t i=0 ; i<mstate->getSize() ; i++)
        {
            *outfile << mstate->getPX(i) << " " << mstate->getPY(i) << " " << mstate->getPZ(i) << std::endl;
        }
    }
    else
    {
        for (int i=0 ; i<nbp ; i++)
        {
            *outfile << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
        }
    }

    *outfile << std::endl;

    //Write Cells
    unsigned int numberOfCells, totalSize;
    numberOfCells = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );
    totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );


    *outfile << "CELLS " << numberOfCells << " " << totalSize << std::endl;

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 2 << " " << topology->getEdge(i) << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 3 << " " <<  topology->getTriangle(i) << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 4 << " " << topology->getQuad(i) << std::endl;
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 4 << " " <<  topology->getTetra(i) << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 8 << " " <<  topology->getHexa(i) << std::endl;
    }

    *outfile << std::endl;

    *outfile << "CELL_TYPES " << numberOfCells << std::endl;

    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 3 << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 5 << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 9 << std::endl;
    }

    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 10 << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 12 << std::endl;
    }

    *outfile << std::endl;

    //write dataset attributes
    if (!pointsData.empty())
    {
        *outfile << "POINT_DATA " << nbp << std::endl;
        writeData(pointsDataObject, pointsDataField, pointsDataName);
    }

    if (!cellsData.empty())
    {
        *outfile << "CELL_DATA " << numberOfCells << std::endl;
        writeData(cellsDataObject, cellsDataField, cellsDataName);
    }

    outfile->close();

    ++nbFiles;

    msg_info() << "Export VTK in file " << filename << "  done.";
}

void VTKExporter::writeVTKXML()
{
    std::string filename = vtkFilename.getFullPath();

    std::ostringstream oss;
    oss << nbFiles;

    if ( filename.size() > 3 && filename.substr(filename.size()-4)==".vtu")
    {
        if (!overwrite.getValue())
            filename = filename.substr(0,filename.size()-4) + oss.str() + ".vtu";
    }
    else
    {
        if (!overwrite.getValue())
            filename += oss.str();
        filename += ".vtu";
    }

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        delete outfile;
        outfile = NULL;
        return;
    }
    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = position;

    const int nbp = (!pointsPos.empty()) ? pointsPos.size() : topology->getNbPoints();

    unsigned int numberOfCells;
    numberOfCells = ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? topology->getNbHexas() : 0 );

    msg_info() << "### VTKExporter[" << this->getName() << "] ###" << msgendl
               << "Nb points: " << nbp << msgendl
               << "Nb edges: " << ( (writeEdges.getValue()) ? topology->getNbEdges() : 0 ) << msgendl
               << "Nb triangles: " << ( (writeTriangles.getValue()) ? topology->getNbTriangles() : 0 ) << msgendl
               << "Nb quads: " << ( (writeQuads.getValue()) ? topology->getNbQuads() : 0 ) << msgendl
               << "Nb tetras: " << ( (writeTetras.getValue()) ? topology->getNbTetras() : 0 ) << msgendl
               << "Nb hexas: " << ( (writeHexas.getValue()) ? topology->getNbHexas() : 0 ) << msgendl
               << "### ###" << msgendl
               << "Total nb cells: " << numberOfCells << msgendl;

    //write header
    *outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *outfile << "  <UnstructuredGrid>" << std::endl;

    //write piece
    *outfile << "    <Piece NumberOfPoints=\"" << nbp << "\" NumberOfCells=\""<< numberOfCells << "\">" << std::endl;

    //write point data
    if (!pointsData.empty())
    {
        *outfile << "      <PointData>" << std::endl;
        writeDataArray(pointsDataObject, pointsDataField, pointsDataName);
        *outfile << "      </PointData>" << std::endl;
    }
    //write cell data
    if (!cellsData.empty())
    {
        *outfile << "      <CellData>" << std::endl;
        writeDataArray(cellsDataObject, cellsDataField, cellsDataName);
        *outfile << "      </CellData>" << std::endl;
    }



    //write points
    *outfile << "      <Points>" << std::endl;
    *outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (!pointsPos.empty())
    {
        for (int i = 0 ; i < nbp; i++)
        {
            *outfile << "\t" << pointsPos[i] << std::endl;
        }
    }
    else if (mstate && mstate->getSize() == (size_t)nbp)
    {
        for (size_t i = 0; i < mstate->getSize(); i++)
            *outfile << "          " << mstate->getPX(i) << " " << mstate->getPY(i) << " " << mstate->getPZ(i) << std::endl;
    }
    else
    {
        for (int i = 0; i < nbp; i++)
            *outfile << "          " << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
    }
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Points>" << std::endl;

    //write cells
    *outfile << "      <Cells>" << std::endl;
    //write connectivity
    *outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << "          " << topology->getEdge(i) << std::endl;
    }

    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << "          " <<  topology->getTriangle(i) << std::endl;
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << "          " << topology->getQuad(i) << std::endl;
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << "          " <<  topology->getTetra(i) << std::endl;
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << "          " <<  topology->getHexa(i) << std::endl;
    }
    *outfile << "        </DataArray>" << std::endl;
    //write offsets
    int num = 0;
    *outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
        {
            num += 2;
            *outfile << num << " ";
        }
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
        {
            num += 3;
            *outfile << num << " ";
        }
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
        {
            num += 4;
            *outfile << num << " ";
        }
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
        {
            num += 4;
            *outfile << num << " ";
        }
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
        {
            num += 8;
            *outfile << num << " ";
        }
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    //write types
    *outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
    *outfile << "          ";
    if (writeEdges.getValue())
    {
        for (int i=0 ; i<topology->getNbEdges() ; i++)
            *outfile << 3 << " ";
    }
    if (writeTriangles.getValue())
    {
        for (int i=0 ; i<topology->getNbTriangles() ; i++)
            *outfile << 5 << " ";
    }
    if (writeQuads.getValue())
    {
        for (int i=0 ; i<topology->getNbQuads() ; i++)
            *outfile << 9 << " ";
    }
    if (writeTetras.getValue())
    {
        for (int i=0 ; i<topology->getNbTetras() ; i++)
            *outfile << 10 << " ";
    }
    if (writeHexas.getValue())
    {
        for (int i=0 ; i<topology->getNbHexas() ; i++)
            *outfile << 12 << " ";
    }
    *outfile << std::endl;
    *outfile << "        </DataArray>" << std::endl;
    *outfile << "      </Cells>" << std::endl;

    //write end
    *outfile << "    </Piece>" << std::endl;
    *outfile << "  </UnstructuredGrid>" << std::endl;
    *outfile << "</VTKFile>" << std::endl;
    outfile->close();
    ++nbFiles;

    msg_info() << "Export VTK XML in file " << filename << "  done.";
}

void VTKExporter::writeParallelFile()
{
    std::string filename = vtkFilename.getFullPath();
    filename.insert(0, "P_");
    filename += ".vtk";

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        delete outfile;
        outfile = NULL;
        return;
    }

    *outfile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *outfile << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;

    const helper::vector<std::string>& pointsData = dPointsDataFields.getValue();
    const helper::vector<std::string>& cellsData = dCellsDataFields.getValue();

    //write type of the data
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    if (!pointsData.empty())
    {
        for (unsigned int i=0 ; i<pointsDataObject.size() ; i++)
        {
            core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (pointsDataObject[i]);
            core::objectmodel::BaseData* field = NULL;
            if (obj)
            {
                field = obj->findData(pointsDataField[i]);
            }

            if (!obj || !field)
            {
                if (!obj)
                    msg_error() << "VTKExporter : error while fetching data field '" << msgendl
                                << pointsDataField[i] << "' of object '" << pointsDataObject[i] << msgendl
                                << "', check object name" << msgendl;
                else if (!field)
                    msg_error() << "VTKExporter : error while fetching data field '" << msgendl
                                << pointsDataField[i] << "' of object '" << pointsDataObject[i] << msgendl
                                << "', check field name " << msgendl;
            }
            else
            {
                //Scalars
                std::string type;
                unsigned int sizeSeg=0;
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<int> >* >(field))
                {
                    type = "Int32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<unsigned int> >* >(field))
                {
                    type = "UInt32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                //Vectors
                if (type.empty())
                {
                    if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                    {
                        type = "Float32";
                        sizeSeg = 3;
                    }
                    if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                    {
                        type = "Float64";
                        sizeSeg = 3;
                    }
                }

                *outfile << "    <PPointData>" << std::endl;
                *outfile << "      <PDataArray type=\""<< type << "\" Name=\"" << pointsDataName[i];
                if(sizeSeg > 1)
                    *outfile << "\" NumberOfComponents=\"" << sizeSeg;
                *outfile << "\"/>" << std::endl;
                *outfile << "    </PPointData>" << std::endl;
            }
        }
    }

    if (!cellsData.empty())
    {
        for (unsigned int i=0 ; i<cellsDataObject.size() ; i++)
        {
            core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (cellsDataObject[i]);
            core::objectmodel::BaseData* field = NULL;
            if (obj)
            {
                field = obj->findData(cellsDataField[i]);
            }

            if (!obj || !field)
            {
                if (!obj)
                    msg_error() << "VTKExporter : error while fetching data field '"
                         << cellsDataField[i] << "' of object '" << cellsDataObject[i]
                         << "', check object name" << sendl;
                else if (!field)
                    msg_error() << "VTKExporter : error while fetching data field '" << msgendl
                                << cellsDataField[i] << "' of object '" << cellsDataObject[i] << msgendl
                                << "', check field name " << msgendl;
            }
            else
            {
                //Scalars
                std::string type;
                unsigned int sizeSeg=0;
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<int> >* >(field))
                {
                    type = "Int32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<unsigned int> >* >(field))
                {
                    type = "UInt32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData< helper::vector<float> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector<double> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                //Vectors
                if (type.empty())
                {
                    if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3f > >* > (field))
                    {
                        type = "Float32";
                        sizeSeg = 3;
                    }
                    if (dynamic_cast<sofa::core::objectmodel::TData<helper::vector< defaulttype::Vec3d > >* >(field))
                    {
                        type = "Float64";
                        sizeSeg = 3;
                    }
                }

                *outfile << "    <PCellData>" << std::endl;
                *outfile << "      <PDataArray type=\""<< type << "\" Name=\"" << cellsDataName[i];
                if(sizeSeg > 1)
                    *outfile << "\" NumberOfComponents=\"" << sizeSeg;
                *outfile << "\"/>" << std::endl;
                *outfile << "    </PCellData>" << std::endl;
            }
        }
    }

    *outfile << "    <PPoints>" << std::endl;
    *outfile << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
    *outfile << "    </PPoints>" << std::endl;

    //write piece
    for(int i = 1; i < nbFiles; ++i)
    {
        std::ostringstream oss;
        oss << i;
        *outfile << "    <Piece Source=\"" << vtkFilename.getFullPath() << oss.str() << ".vtu" << "\"/>" << std::endl;
    }

    //write end
    *outfile << "  </PUnstructuredGrid>" << std::endl;
    *outfile << "</VTKFile>" << std::endl;
    outfile->close();

    msg_info() << "Export VTK in file " << filename << "  done.";
}


void VTKExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent* ev = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);

        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            if(fileFormat.getValue())
                writeVTKXML();
            else
                writeVTKSimple();
            break;

        case 'F':
        case 'f':
            if(fileFormat.getValue())
                writeParallelFile();
        }
    }


    if ( /*simulation::AnimateEndEvent* ev =*/ simulation::AnimateEndEvent::checkEventType(event))
    {
        unsigned int maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;

        stepCounter++;
        if(stepCounter >= maxStep)
        {
            stepCounter = 0;
            if(fileFormat.getValue())
                writeVTKXML();
            else
                writeVTKSimple();
        }
    }
}

void VTKExporter::cleanup()
{
    if (exportAtEnd.getValue())
        (fileFormat.getValue()) ? writeVTKXML() : writeVTKSimple();

}

void VTKExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        (fileFormat.getValue()) ? writeVTKXML() : writeVTKSimple();
}

}

}

}
