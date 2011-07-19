/*
 * OBJExporter.cpp
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#include "OBJExporter.h"

#include <sstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/ExportOBJVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(OBJExporter)

int OBJExporterClass = core::RegisterObject("Read State vectors from file at each timestep")
        .add< OBJExporter >();

OBJExporter::OBJExporter()
    : stepCounter(0), outfile(NULL)
    , objFilename( initData(&objFilename, "filename", "output VTK file name"))
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
{
    this->f_listening.setValue(true);
}

OBJExporter::~OBJExporter()
{
    if (outfile)
        delete outfile;
}

void OBJExporter::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(topology);
    context->get(mstate);

    if (!topology)
    {
        serr << "OBJExporter : error, no topology ." << sendl;
        return;
    }

    nbFiles = 0;
// 	const std::string& filename = objFilename.getFullPath();
// //	std::cout << filename << std::endl;
//
// 	outfile = new std::ofstream(filename.c_str());
// 	if( !outfile->is_open() )
// 	{
// 		serr << "Error creating file "<<filename<<sendl;
// 		delete outfile;
// 		outfile = NULL;
// 		return;
// 	}

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

void OBJExporter::fetchDataFields(const helper::vector<std::string>& strData, helper::vector<std::string>& objects, helper::vector<std::string>& fields, helper::vector<std::string>& names)
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
            serr << "OBJExporter : error while parsing dataField names" << sendl;
            continue;
        }
        if (name.empty()) name = dataFieldName;
        objects.push_back(objectName);
        fields.push_back(dataFieldName);
        names.push_back(name);
    }
}

void OBJExporter::writeData(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    //std::cout << "List o: " << objects << std::endl;
    //std::cout << "List f: " << fields << std::endl;

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        //std::cout << objects[i] << std::endl;
        if (obj)
        {
            std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

            for (unsigned int j=0 ; j<f.size() && !field; j++)
            {
                if(fields[i].compare(f[j].first) == 0)
                    field = f[j].second;
            }
        }

        if (!obj || !field)
        {
            serr << "OBJExporter : error while fetching data field, check object name or field name " << sendl;
        }
        else
        {
            //std::cout << "Type: " << field->getValueTypeString() << std::endl;

            //retrieve data file type
//			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
//				std::cout << "Vec3f" << std::endl;
//			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
//				std::cout << "Vec3d" << std::endl;

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
            }
            *outfile << line << std::endl;
            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << std::endl;
        }
    }
}

void OBJExporter::writeDataArray(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names)
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();

    //std::cout << "List o: " << objects << std::endl;
    //std::cout << "List f: " << fields << std::endl;

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = NULL;
        //std::cout << objects[i] << std::endl;
        if (obj)
        {
            std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

            for (unsigned int j=0 ; j<f.size() && !field; j++)
            {
                if(fields[i].compare(f[j].first) == 0)
                    field = f[j].second;
            }
        }

        if (!obj || !field)
        {
            serr << "OBJExporter : error while fetching data field, check object name or field name " << sendl;
        }
        else
        {
            //std::cout << "Type: " << field->getValueTypeString() << std::endl;

            //retrieve data file type
//			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
//				std::cout << "Vec3f" << std::endl;
//			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
//				std::cout << "Vec3d" << std::endl;

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
            *outfile << "        <DataArray type=\""<< type << "\" Name=\"" << names[i];
            if(sizeSeg > 1)
                *outfile << "\" NumberOfComponents=\"" << sizeSeg;
            *outfile << "\" format=\"ascii\">" << std::endl;
            *outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *outfile << "        </DataArray>" << std::endl;
        }
    }
}


std::string OBJExporter::segmentString(std::string str, unsigned int n)
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


void OBJExporter::writeVTKSimple()
{
    std::string filename = objFilename.getFullPath();
    filename += ".vtu";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
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
    else if (mstate && mstate->getSize() == nbp)
    {
        for (int i=0 ; i<mstate->getSize() ; i++)
        {
            *outfile << mstate->getPX(i) << " " << mstate->getPY(i) << " " << mstate->getPZ(i) << std::endl;
        }
    }
    else
    {
        for (int i=0 ; i<nbp ; i++)
        {
            *outfile << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
//		std::cout << topology->getPX(i) << " " << topology->getPY(i) << " " << topology->getPZ(i) << std::endl;
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
    sout << filename << " written" << sendl;
}

void OBJExporter::writeVTKXML()
{
    std::string filename = objFilename.getFullPath();
    std::ostringstream oss;
    oss << nbFiles;
    filename += oss.str() + ".vtu";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
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
//	unsigned int totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
// 				   +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
// 				   +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
// 				   +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
// 				   +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );

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
    if (mstate && mstate->getSize() == nbp)
    {
        for (int i = 0; i < mstate->getSize(); i++)
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
            num += 6;
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
    sout << filename << " written" << sendl;
    ++nbFiles;
}

void OBJExporter::writeParallelFile()
{
    std::string filename = objFilename.getFullPath();
    filename.insert(0, "P_");
    filename += ".vtk";
//	std::cout << filename << std::endl;

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file "<<filename<<sendl;
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
            //std::cout << objects[i] << std::endl;
            if (obj)
            {
                std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

                for (unsigned int j=0 ; j<f.size() && !field; j++)
                {
                    if(pointsDataField[i].compare(f[j].first) == 0)
                        field = f[j].second;
                }
            }

            if (!obj || !field)
            {
                serr << "OBJExporter : error while fetching data field, check object name or field name " << sendl;
            }
            else
            {
                //std::cout << "Type: " << field->getValueTypeString() << std::endl;

                //retrieve data file type
                //			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
                //				std::cout << "Vec3f" << std::endl;
                //			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
                //				std::cout << "Vec3d" << std::endl;

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
            //std::cout << objects[i] << std::endl;
            if (obj)
            {
                std::vector< std::pair<std::string, core::objectmodel::BaseData*> > f = obj->getFields();

                for (unsigned int j=0 ; j<f.size() && !field; j++)
                {
                    if(cellsDataField[i].compare(f[j].first) == 0)
                        field = f[j].second;
                }
            }

            if (!obj || !field)
            {
                serr << "OBJExporter : error while fetching data field, check object name or field name " << sendl;
            }
            else
            {
                //std::cout << "Type: " << field->getValueTypeString() << std::endl;

                //retrieve data file type
                //			if (dynamic_cast<Data< defaulttype::Vec3f >* >(field))
                //				std::cout << "Vec3f" << std::endl;
                //			if (dynamic_cast<Data< defaulttype::Vec3d >* >(field))
                //				std::cout << "Vec3d" << std::endl;

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
        *outfile << "    <Piece Source=\"" << objFilename.getFullPath() << oss.str() << ".vtu" << "\"/>" << std::endl;
    }

    //write end
    *outfile << "  </PUnstructuredGrid>" << std::endl;
    *outfile << "</VTKFile>" << std::endl;
    outfile->close();
    sout << "parallel file " << filename << " written" << sendl;
}


void OBJExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        std::cout << "key pressed " << std::endl;
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
        {
            std::string filename = objFilename.getFullPath();
            filename += ".obj";
            outfile = new std::ofstream(filename.c_str());
            filename = objFilename.getFullPath();
            filename += ".mtl";
            mtlfile = new std::ofstream(filename.c_str());

            sofa::simulation::ExportOBJVisitor exportOBJ(core::ExecParams::defaultInstance(),outfile, mtlfile);
            sofa::core::objectmodel::BaseContext* context = this->getContext();
            context->executeVisitor(&exportOBJ);

            break;
        }
        }
    }


    if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        unsigned int maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;

        stepCounter++;
        if(stepCounter > maxStep)
        {
            stepCounter = 0;

            std::string filename = objFilename.getFullPath();
            std::ostringstream oss;
            oss << stepCounter;
            filename += oss.str() + ".obj";
            outfile = new std::ofstream(filename.c_str());

            std::string mtlfilename = objFilename.getFullPath();
            std::ostringstream mtloss;
            mtloss << stepCounter;
            mtlfilename += mtloss.str() + ".mtl";
            mtlfile = new std::ofstream(mtlfilename.c_str());

            sofa::simulation::ExportOBJVisitor exportOBJ(core::ExecParams::defaultInstance(),outfile, mtlfile);
            sofa::core::objectmodel::BaseContext* context = this->getContext();
            context->executeVisitor(&exportOBJ);
        }
    }
}

void OBJExporter::cleanup()
{
    if (exportAtEnd.getValue())
        (fileFormat.getValue()) ? writeVTKXML() : writeVTKSimple();

}

void OBJExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        (fileFormat.getValue()) ? writeVTKXML() : writeVTKSimple();
}

}

}

}
