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
#include <sofa/component/io/mesh/VTKExporter.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/events/SimulationInitDoneEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

namespace sofa::component::_vtkexporter_
{

VTKExporter::VTKExporter() 
    : sofa::simulation::BaseSimulationExporter()
    , m_topology(initLink("topology", "topology to export"))
    , m_mstate(initLink("mstate", "mechanical state to export"))
    , m_outfile(nullptr)
    , d_fileFormat(initData(&d_fileFormat, (bool) true, "XMLformat", "Set to true to use XML format"))
    , d_position(initData(&d_position, "position", "points position (will use points from topology or mechanical state if this is empty)"))
    , d_writeEdges(initData(&d_writeEdges, (bool) true, "edges", "write edge topology"))
    , d_writeTriangles(initData(&d_writeTriangles, (bool) false, "triangles", "write triangle topology"))
    , d_writeQuads(initData(&d_writeQuads, (bool) false, "quads", "write quad topology"))
    , d_writeTetras(initData(&d_writeTetras, (bool) false, "tetras", "write tetra topology"))
    , d_writeHexas(initData(&d_writeHexas, (bool) false, "hexas", "write hexa topology"))
    , d_dPointsDataFields(initData(&d_dPointsDataFields, "pointsDataFields", "Data to visualize (on points)"))
    , d_dCellsDataFields(initData(&d_dCellsDataFields, "cellsDataFields", "Data to visualize (on cells)"))
    , d_overwrite(initData(&d_overwrite, false, "overwrite", "overwrite the file, otherwise create a new file at each export, with suffix in the filename"))
{
}

VTKExporter::~VTKExporter(){}

void VTKExporter::doInit() 
{ 
    const sofa::core::objectmodel::BaseContext* context = this->getContext();

    if (!m_mstate.get())
    {
        m_mstate.set(context->getMechanicalState());
    }

    if (m_mstate && !d_position.isSet())
    {
        if (core::BaseData* data = m_mstate->findData("position"))
        {
            msg_info() << "found position data in mechanical state";
            d_position.setParent(data);
        }
    }

    if (!m_topology.get())
    {
        m_topology.set(context->getMeshTopology());
    }

    if (!m_topology)
    {
        msg_error() << "no topology.";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    msg_info() << "found topology " << m_topology->getName();

    if (!d_position.isSet())
    {
        if (core::BaseData* data = m_topology->findData("position"))
        {
            msg_info() << "found position data in topology";
            d_position.setParent(data);
        }
    }

    nbFiles = 0;

    const type::vector<std::string>& pointsData = d_dPointsDataFields.getValue();
    const type::vector<std::string>& cellsData = d_dCellsDataFields.getValue();

    if (!pointsData.empty())
    {
        fetchDataFields(pointsData, pointsDataObject, pointsDataField, pointsDataName);
    }
    if (!cellsData.empty())
    {
        fetchDataFields(cellsData, cellsDataObject, cellsDataField, cellsDataName);
    }

    /// Activate the listening to the event in order to be able to export file at first step and/or the nth-step
    if(d_exportEveryNbSteps.getValue() != 0 || d_exportAtBegin.getValue())
        this->f_listening.setValue(true);

    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid) ;
}

void VTKExporter::doReInit()
{
    doInit();
}

void VTKExporter::fetchDataFields(const type::vector<std::string>& strData, type::vector<std::string>& objects, type::vector<std::string>& fields, type::vector<std::string>& names)
{
    for (auto str : strData)
    {
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
            msg_error() << "error while parsing dataField names" ;
            continue;
        }
        if (name.empty()) name = dataFieldName;
        objects.push_back(objectName);
        fields.push_back(dataFieldName);
        names.push_back(name);
    }
}

void VTKExporter::writeData(const type::vector<std::string>& objects, const type::vector<std::string>& fields, const type::vector<std::string>& names)
{
    const sofa::core::objectmodel::BaseContext* context = this->getContext();

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        const core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = nullptr;
        if (obj)
        {
            field = obj->findData(fields[i]);
        }

        if (!obj || !field)
        {
            if (!obj)
                msg_error() << "error while fetching data field '" << msgendl
                            << fields[i] << "' of object '" << objects[i] << msgendl
                            << "', check object name"  << msgendl;
            else if (!field)
                msg_error() << "error while fetching data field " << msgendl
                            << fields[i] << " of object '" << objects[i] << msgendl
                            << "', check field name " << msgendl;
        }
        else
        {
            //Scalars
            std::string line;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<float> >* >(field))
            {
                line = "float 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data<type::vector<double> >* >(field))
            {
                line = "double 1";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec2f > >* > (field))
            {
                line = "float 2";
                sizeSeg = 2;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec2d > >* >(field))
            {
                line = "double 2";
                sizeSeg = 2;
            }

            //if this is a scalar
            if (!line.empty())
            {
                *m_outfile << "SCALARS" << " " << names[i] << " ";
                *m_outfile << line << std::endl;
                *m_outfile << "LOOKUP_TABLE default" << std::endl;
            }
            else
            {
                //Vectors
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3f > >* > (field))
                {
                    line = "float";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3d > >* >(field))
                {
                    line = "double";
                    sizeSeg = 3;
                }
                *m_outfile << "VECTORS" << " " << names[i] << " ";
                *m_outfile << line << std::endl;
            }

            *m_outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *m_outfile << std::endl;


        }
    }
}

void VTKExporter::writeDataArray(const type::vector<std::string>& objects, const type::vector<std::string>& fields, const type::vector<std::string>& names)
{
    const sofa::core::objectmodel::BaseContext* context = this->getContext();

    for (unsigned int i=0 ; i<objects.size() ; i++)
    {
        const core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (objects[i]);
        core::objectmodel::BaseData* field = nullptr;
        if (obj)
        {
            field = obj->findData(fields[i]);
        }

        if (!obj || !field)
        {
            if (!obj)
                msg_error() << "error while fetching data field '" << msgendl
                            << fields[i] << "' of object '" << objects[i] << msgendl
                            << "', check object name" << msgendl;
            else if (!field)
                msg_error()  << "error while fetching data field " << msgendl
                             << fields[i] << " of object '" << objects[i] << msgendl
                             << "', check field name " << msgendl;
        }
        else
        {
            //Scalars
            std::string type;
            unsigned int sizeSeg=0;
            if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<int> >* >(field))
            {
                type = "Int32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<unsigned int> >* >(field))
            {
                type = "UInt32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<float> >* >(field))
            {
                type = "Float32";
                sizeSeg = 1;
            }
            if (dynamic_cast<sofa::core::objectmodel::Data<type::vector<double> >* >(field))
            {
                type = "Float64";
                sizeSeg = 1;
            }

            //Vectors
            if (type.empty())
            {
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec1f> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec1d> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec2f> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 2;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec2d> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 2;
                }

                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3f > >* > (field))
                {
                    type = "Float32";
                    sizeSeg = 3;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3d > >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 3;
                }
            }
            *m_outfile << "        <DataArray type=\""<< type << "\" Name=\"" << names[i];
            if(sizeSeg > 1)
                *m_outfile << "\" NumberOfComponents=\"" << sizeSeg;
            *m_outfile << "\" format=\"ascii\">" << std::endl;
            *m_outfile << segmentString(field->getValueString(),sizeSeg) << std::endl;
            *m_outfile << "        </DataArray>" << std::endl;
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


bool VTKExporter::write()
{ 
    if (!this->isComponentStateValid()) 
        return false;
    
    if (d_fileFormat.getValue())
        return writeVTKXML();
    return writeVTKSimple();
}


bool VTKExporter::writeVTKSimple()
{
    std::string filename = d_filename.getFullPath();

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

        if (d_overwrite.getValue())
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

    m_outfile.reset(new std::ofstream(filename.c_str()));
    if( !m_outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        m_outfile.reset();
        return false;
    }

    const type::vector<std::string>& pointsData = d_dPointsDataFields.getValue();
    const type::vector<std::string>& cellsData = d_dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const size_t nbp = (!pointsPos.empty()) ? pointsPos.size() : m_topology->getNbPoints();

    //Write header
    *m_outfile << "# vtk DataFile Version 2.0" << std::endl;

    //write Title
    *m_outfile << "Exported VTK file" << std::endl;

    //write Data type
    *m_outfile << "ASCII" << std::endl;

    *m_outfile << std::endl;

    //write dataset (geometry, unstructured grid)
    *m_outfile << "DATASET " << "UNSTRUCTURED_GRID" << std::endl;

    *m_outfile << "POINTS " << nbp << " float" << std::endl;
    //write Points
    if (!pointsPos.empty())
    {
        for (size_t i=0 ; i<nbp; i++)
        {
            *m_outfile << pointsPos[i] << std::endl;
        }
    }

    *m_outfile << std::endl;

    //Write Cells
    size_t numberOfCells, totalSize;
    numberOfCells = ((d_writeEdges.getValue()) ? m_topology->getNbEdges() : 0 )
            +((d_writeTriangles.getValue()) ? m_topology->getNbTriangles() : 0 )
            +((d_writeQuads.getValue()) ? m_topology->getNbQuads() : 0 )
            +((d_writeTetras.getValue()) ? m_topology->getNbTetras() : 0 )
            +((d_writeHexas.getValue()) ? m_topology->getNbHexas() : 0 );
    totalSize =     ((d_writeEdges.getValue()) ? 3 * m_topology->getNbEdges() : 0 )
            +((d_writeTriangles.getValue()) ? 4 * m_topology->getNbTriangles() : 0 )
            +((d_writeQuads.getValue()) ? 5 * m_topology->getNbQuads() : 0 )
            +((d_writeTetras.getValue()) ? 5 * m_topology->getNbTetras() : 0 )
            +((d_writeHexas.getValue()) ? 9 * m_topology->getNbHexas() : 0 );


    *m_outfile << "CELLS " << numberOfCells << " " << totalSize << std::endl;

    if (d_writeEdges.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbEdges() ; i++)
            *m_outfile << 2 << " " << m_topology->getEdge(i) << std::endl;
    }

    if (d_writeTriangles.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTriangles() ; i++)
            *m_outfile << 3 << " " <<  m_topology->getTriangle(i) << std::endl;
    }
    if (d_writeQuads.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbQuads() ; i++)
            *m_outfile << 4 << " " << m_topology->getQuad(i) << std::endl;
    }

    if (d_writeTetras.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTetras() ; i++)
            *m_outfile << 4 << " " <<  m_topology->getTetra(i) << std::endl;
    }
    if (d_writeHexas.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbHexas() ; i++)
            *m_outfile << 8 << " " <<  m_topology->getHexa(i) << std::endl;
    }

    *m_outfile << std::endl;

    *m_outfile << "CELL_TYPES " << numberOfCells << std::endl;

    if (d_writeEdges.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbEdges() ; i++)
            *m_outfile << 3 << std::endl;
    }

    if (d_writeTriangles.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTriangles() ; i++)
            *m_outfile << 5 << std::endl;
    }
    if (d_writeQuads.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbQuads() ; i++)
            *m_outfile << 9 << std::endl;
    }

    if (d_writeTetras.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTetras() ; i++)
            *m_outfile << 10 << std::endl;
    }
    if (d_writeHexas.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbHexas() ; i++)
            *m_outfile << 12 << std::endl;
    }

    *m_outfile << std::endl;

    //write dataset attributes
    if (!pointsData.empty())
    {
        *m_outfile << "POINT_DATA " << nbp << std::endl;
        writeData(pointsDataObject, pointsDataField, pointsDataName);
    }

    if (!cellsData.empty())
    {
        *m_outfile << "CELL_DATA " << numberOfCells << std::endl;
        writeData(cellsDataObject, cellsDataField, cellsDataName);
    }

    m_outfile->close();

    ++nbFiles;

    msg_info() << "Export VTK in file " << filename << "  done.";

    return true;
}

bool VTKExporter::writeVTKXML()
{
    std::string filename = d_filename.getFullPath();

    std::ostringstream oss;
    oss << nbFiles;

    if ( filename.size() > 3 && filename.substr(filename.size()-4)==".vtu")
    {
        if (!d_overwrite.getValue())
            filename = filename.substr(0,filename.size()-4) + oss.str() + ".vtu";
    }
    else
    {
        if (!d_overwrite.getValue())
            filename += oss.str();
        filename += ".vtu";
    }

    m_outfile.reset(new std::ofstream(filename.c_str()));
    if( !m_outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        m_outfile.reset();
        return false;
    }
    const type::vector<std::string>& pointsData = d_dPointsDataFields.getValue();
    const type::vector<std::string>& cellsData = d_dCellsDataFields.getValue();

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const size_t nbp = (!pointsPos.empty()) ? pointsPos.size() : m_topology->getNbPoints();

    size_t numberOfCells;
    numberOfCells = ((d_writeEdges.getValue()) ? m_topology->getNbEdges() : 0 )
            +((d_writeTriangles.getValue()) ? m_topology->getNbTriangles() : 0 )
            +((d_writeQuads.getValue()) ? m_topology->getNbQuads() : 0 )
            +((d_writeTetras.getValue()) ? m_topology->getNbTetras() : 0 )
            +((d_writeHexas.getValue()) ? m_topology->getNbHexas() : 0 );

    msg_info() << "### VTKExporter[" << this->getName() << "] ###" << msgendl
               << "Nb points: " << nbp << msgendl
               << "Nb edges: " << ((d_writeEdges.getValue()) ? m_topology->getNbEdges() : 0 ) << msgendl
               << "Nb triangles: " << ((d_writeTriangles.getValue()) ? m_topology->getNbTriangles() : 0 ) << msgendl
               << "Nb quads: " << ((d_writeQuads.getValue()) ? m_topology->getNbQuads() : 0 ) << msgendl
               << "Nb tetras: " << ((d_writeTetras.getValue()) ? m_topology->getNbTetras() : 0 ) << msgendl
               << "Nb hexas: " << ((d_writeHexas.getValue()) ? m_topology->getNbHexas() : 0 ) << msgendl
               << "### ###" << msgendl
               << "Total nb cells: " << numberOfCells << msgendl;

    //write header
    *m_outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *m_outfile << "  <UnstructuredGrid>" << std::endl;

    //write piece
    *m_outfile << "    <Piece NumberOfPoints=\"" << nbp << "\" NumberOfCells=\""<< numberOfCells << "\">" << std::endl;

    //write point data
    if (!pointsData.empty())
    {
        *m_outfile << "      <PointData>" << std::endl;
        writeDataArray(pointsDataObject, pointsDataField, pointsDataName);
        *m_outfile << "      </PointData>" << std::endl;
    }
    //write cell data
    if (!cellsData.empty())
    {
        *m_outfile << "      <CellData>" << std::endl;
        writeDataArray(cellsDataObject, cellsDataField, cellsDataName);
        *m_outfile << "      </CellData>" << std::endl;
    }



    //write points
    *m_outfile << "      <Points>" << std::endl;
    *m_outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;
    if (!pointsPos.empty())
    {
        for (size_t i = 0 ; i < nbp; i++)
        {
            *m_outfile << "\t" << pointsPos[i] << std::endl;
        }
    }

    *m_outfile << "        </DataArray>" << std::endl;
    *m_outfile << "      </Points>" << std::endl;

    //write cells
    *m_outfile << "      <Cells>" << std::endl;
    //write connectivity
    *m_outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << std::endl;
    if (d_writeEdges.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbEdges() ; i++)
            *m_outfile << "          " << m_topology->getEdge(i) << std::endl;
    }

    if (d_writeTriangles.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTriangles() ; i++)
            *m_outfile << "          " <<  m_topology->getTriangle(i) << std::endl;
    }
    if (d_writeQuads.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbQuads() ; i++)
            *m_outfile << "          " << m_topology->getQuad(i) << std::endl;
    }
    if (d_writeTetras.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTetras() ; i++)
            *m_outfile << "          " <<  m_topology->getTetra(i) << std::endl;
    }
    if (d_writeHexas.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbHexas() ; i++)
            *m_outfile << "          " <<  m_topology->getHexa(i) << std::endl;
    }
    *m_outfile << "        </DataArray>" << std::endl;
    //write offsets
    int num = 0;
    *m_outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << std::endl;
    *m_outfile << "          ";
    if (d_writeEdges.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbEdges() ; i++)
        {
            num += 2;
            *m_outfile << num << " ";
        }
    }
    if (d_writeTriangles.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTriangles() ; i++)
        {
            num += 3;
            *m_outfile << num << " ";
        }
    }
    if (d_writeQuads.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbQuads() ; i++)
        {
            num += 4;
            *m_outfile << num << " ";
        }
    }
    if (d_writeTetras.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTetras() ; i++)
        {
            num += 4;
            *m_outfile << num << " ";
        }
    }
    if (d_writeHexas.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbHexas() ; i++)
        {
            num += 8;
            *m_outfile << num << " ";
        }
    }
    *m_outfile << std::endl;
    *m_outfile << "        </DataArray>" << std::endl;
    //write types
    *m_outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << std::endl;
    *m_outfile << "          ";
    if (d_writeEdges.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbEdges() ; i++)
            *m_outfile << 3 << " ";
    }
    if (d_writeTriangles.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTriangles() ; i++)
            *m_outfile << 5 << " ";
    }
    if (d_writeQuads.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbQuads() ; i++)
            *m_outfile << 9 << " ";
    }
    if (d_writeTetras.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbTetras() ; i++)
            *m_outfile << 10 << " ";
    }
    if (d_writeHexas.getValue())
    {
        for (unsigned int i=0 ; i<m_topology->getNbHexas() ; i++)
            *m_outfile << 12 << " ";
    }
    *m_outfile << std::endl;
    *m_outfile << "        </DataArray>" << std::endl;
    *m_outfile << "      </Cells>" << std::endl;

    //write end
    *m_outfile << "    </Piece>" << std::endl;
    *m_outfile << "  </UnstructuredGrid>" << std::endl;
    *m_outfile << "</VTKFile>" << std::endl;
    m_outfile->close();
    ++nbFiles;

    msg_info() << "Export VTK XML in file " << filename << "  done.";

    return true;
}

void VTKExporter::writeParallelFile()
{
    std::string filename = d_filename.getFullPath();
    filename.insert(0, "P_");
    filename += ".vtk";

    m_outfile.reset(new std::ofstream(filename.c_str()));
    if( !m_outfile->is_open() )
    {
        msg_error() << "Error creating file "<<filename;
        m_outfile.reset();
        return;
    }

    *m_outfile << "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">" << std::endl;
    *m_outfile << "  <PUnstructuredGrid GhostLevel=\"0\">" << std::endl;

    const type::vector<std::string>& pointsData = d_dPointsDataFields.getValue();
    const type::vector<std::string>& cellsData = d_dCellsDataFields.getValue();

    //write type of the data
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    if (!pointsData.empty())
    {
        for (unsigned int i=0 ; i<pointsDataObject.size() ; i++)
        {
            core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (pointsDataObject[i]);
            core::objectmodel::BaseData* field = nullptr;
            if (obj)
            {
                field = obj->findData(pointsDataField[i]);
            }

            if (!obj || !field)
            {
                if (!obj)
                    msg_error() << "error while fetching data field '" << msgendl
                                << pointsDataField[i] << "' of object '" << pointsDataObject[i] << msgendl
                                << "', check object name" << msgendl;
                else if (!field)
                    msg_error() << "error while fetching data field '" << msgendl
                                << pointsDataField[i] << "' of object '" << pointsDataObject[i] << msgendl
                                << "', check field name " << msgendl;
            }
            else
            {
                //Scalars
                std::string type;
                unsigned int sizeSeg=0;
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<int> >* >(field))
                {
                    type = "Int32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<unsigned int> >* >(field))
                {
                    type = "UInt32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<float> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector<double> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                //Vectors
                if (type.empty())
                {
                    if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3f > >* > (field))
                    {
                        type = "Float32";
                        sizeSeg = 3;
                    }
                    if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3d > >* >(field))
                    {
                        type = "Float64";
                        sizeSeg = 3;
                    }
                }

                *m_outfile << "    <PPointData>" << std::endl;
                *m_outfile << "      <PDataArray type=\""<< type << "\" Name=\"" << pointsDataName[i];
                if(sizeSeg > 1)
                    *m_outfile << "\" NumberOfComponents=\"" << sizeSeg;
                *m_outfile << "\"/>" << std::endl;
                *m_outfile << "    </PPointData>" << std::endl;
            }
        }
    }

    if (!cellsData.empty())
    {
        for (unsigned int i=0 ; i<cellsDataObject.size() ; i++)
        {
            core::objectmodel::BaseObject* obj = context->get<core::objectmodel::BaseObject> (cellsDataObject[i]);
            core::objectmodel::BaseData* field = nullptr;
            if (obj)
            {
                field = obj->findData(cellsDataField[i]);
            }

            if (!obj || !field)
            {
                if (!obj)
                    msg_error() << "error while fetching data field '"
                         << cellsDataField[i] << "' of object '" << cellsDataObject[i]
                            << "', check object name" << msgendl;
                else if (!field)
                    msg_error() << "error while fetching data field '" << msgendl
                                << cellsDataField[i] << "' of object '" << cellsDataObject[i] << msgendl
                                << "', check field name " << msgendl;
            }
            else
            {
                //Scalars
                std::string type;
                unsigned int sizeSeg=0;
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<int> >* >(field))
                {
                    type = "Int32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<unsigned int> >* >(field))
                {
                    type = "UInt32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data< type::vector<float> >* >(field))
                {
                    type = "Float32";
                    sizeSeg = 1;
                }
                if (dynamic_cast<sofa::core::objectmodel::Data<type::vector<double> >* >(field))
                {
                    type = "Float64";
                    sizeSeg = 1;
                }

                //Vectors
                if (type.empty())
                {
                    if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3f > >* > (field))
                    {
                        type = "Float32";
                        sizeSeg = 3;
                    }
                    if (dynamic_cast<sofa::core::objectmodel::Data<type::vector< type::Vec3d > >* >(field))
                    {
                        type = "Float64";
                        sizeSeg = 3;
                    }
                }

                *m_outfile << "    <PCellData>" << std::endl;
                *m_outfile << "      <PDataArray type=\""<< type << "\" Name=\"" << cellsDataName[i];
                if(sizeSeg > 1)
                    *m_outfile << "\" NumberOfComponents=\"" << sizeSeg;
                *m_outfile << "\"/>" << std::endl;
                *m_outfile << "    </PCellData>" << std::endl;
            }
        }
    }

    *m_outfile << "    <PPoints>" << std::endl;
    *m_outfile << "      <PDataArray type=\"Float32\" NumberOfComponents=\"3\"/>" << std::endl;
    *m_outfile << "    </PPoints>" << std::endl;

    //write piece
    for(int i = 1; i < nbFiles; ++i)
    {
        std::ostringstream oss;
        oss << i;
        *m_outfile << "    <Piece Source=\"" << d_filename.getFullPath() << oss.str() << ".vtu" << "\"/>" << std::endl;
    }

    //write end
    *m_outfile << "  </PUnstructuredGrid>" << std::endl;
    *m_outfile << "</VTKFile>" << std::endl;
    m_outfile->close();

    msg_info() << "Export VTK in file " << filename << "  done.";
}


void VTKExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        const sofa::core::objectmodel::KeypressedEvent* ev = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);

        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            write();
            break;

        case 'F':
        case 'f':
            if(d_fileFormat.getValue())
                writeParallelFile();
        }
    }
    else
    {
        BaseSimulationExporter::handleEvent(event);
    }
}

} // namespace sofa::component::_vtkexporter_

namespace sofa::component::io::mesh
{

void registerVTKExporter(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Export a given mesh in a VTK file.")
        .add< VTKExporter >());
}

} // namespace sofa::component::io::mesh
