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
#include "MeshExporter.h"

#include <sstream>
#include <iomanip>
#include <fstream>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>

using sofa::core::objectmodel::ComponentState ;

namespace sofa
{

namespace component
{

namespace _meshexporter_
{

SOFA_DECL_CLASS(MeshExporter)

int MeshExporterClass = core::RegisterObject("Export topology and positions into file. " msgendl
                                             "Supported format are: " msgendl
                                             "- vtkxml" msgendl
                                             "- vtk" msgendl
                                             "- netgen" msgendl
                                             "- teten" msgendl
                                             "- gmsh" msgendl)
        .add< MeshExporter >();

MeshExporter::MeshExporter()
    : d_fileFormat( initData(&d_fileFormat, sofa::helper::OptionsGroup(6,"ALL","vtkxml","vtk","netgen","tetgen","gmsh"), "format", "File format to use"))
    , d_position( initData(&d_position, "position", "points position (will use points from topology or mechanical state if this is empty)"))
    , d_writeEdges( initData(&d_writeEdges, (bool) true, "edges", "write edge topology"))
    , d_writeTriangles( initData(&d_writeTriangles, (bool) true, "triangles", "write triangle topology"))
    , d_writeQuads( initData(&d_writeQuads, (bool) true, "quads", "write quad topology"))
    , d_writeTetras( initData(&d_writeTetras, (bool) true, "tetras", "write tetra topology"))
    , d_writeHexas( initData(&d_writeHexas, (bool) true, "hexas", "write hexa topology"))
{
}

MeshExporter::~MeshExporter()
{
}

void MeshExporter::doInit()
{
    doReInit() ;
}

void MeshExporter::doReInit()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(m_inputtopology);
    context->get(m_inputmstate);

    if (!m_inputtopology)
    {
        msg_error() << "Error, no topology." << sendl;
        m_componentstate = ComponentState::Invalid ;
        return;
    }

    if (!d_position.isSet() && m_inputmstate)
    {
        sofa::core::objectmodel::BaseData* parent = NULL;
        if (!parent && m_inputmstate) parent = m_inputmstate->findData("position");
        if (!parent && m_inputtopology) parent = m_inputmstate->findData("topology");
        if (parent)
        {
            d_position.setParent(parent);
            d_position.setReadOnly(true);
        }
    }

    m_componentstate = ComponentState::Valid ;
}

bool MeshExporter::write()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;
    return writeMesh() ;
}

bool MeshExporter::writeMesh()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    const unsigned int format = d_fileFormat.getValue().getSelectedId();

    const bool all = (format == 0);
    const bool vtkxml = all || (format == 1);
    const bool vtk    = all || (format == 2);
    const bool netgen = all || (format == 3);
    const bool tetgen = all || (format == 4);
    const bool gmsh   = all || (format == 5);
    msg_info() << "Exporting a mesh in '" << getMeshFilename("") << "'" << msgendl
               << "-" << d_position.getValue().size() << " points" << msgendl
               << "-" << m_inputtopology->getNbEdges() << " edges" << msgendl
               << "-" << m_inputtopology->getNbTriangles() << " triangles" << msgendl
               << "-" << m_inputtopology->getNbQuads() << " quads" << msgendl
               << "-" << m_inputtopology->getNbTetras() << " tetras" << msgendl
               << "-" << m_inputtopology->getNbHexas() << " hexas";

    bool res = false ;
    if (vtkxml)
        res = writeMeshVTKXML();
    if (vtk)
        res = writeMeshVTK();
    if (netgen)
        res = writeMeshNetgen();
    if (tetgen)
        res = writeMeshTetgen();
    if (gmsh)
        res = writeMeshGmsh();

    return res ;
}

std::string MeshExporter::getMeshFilename(const char* ext)
{
    int nbp = d_position.getValue().size();
    unsigned int nbce;
    nbce = ( (d_writeEdges.getValue()) ? m_inputtopology->getNbEdges() : 0 )
            + ( (d_writeTriangles.getValue()) ? m_inputtopology->getNbTriangles() : 0 )
            + ( (d_writeQuads.getValue()) ? m_inputtopology->getNbQuads() : 0 )
            + ( (d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0 )
            + ( (d_writeHexas.getValue()) ? m_inputtopology->getNbHexas() : 0 );
    unsigned int nbe = 0;
    nbe = ( (d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0 )
            + ( (d_writeHexas.getValue()) ? m_inputtopology->getNbHexas() : 0 );
    if (!nbe)
        nbe = ( (d_writeTriangles.getValue()) ? m_inputtopology->getNbTriangles() : 0 )
                + ( (d_writeQuads.getValue()) ? m_inputtopology->getNbQuads() : 0 );
    if (!nbe)
        nbe = ( (d_writeEdges.getValue()) ? m_inputtopology->getNbEdges() : 0 );
    if (!nbe)
        nbe = nbp;

    std::ostringstream oss;
    std::string filename = d_filename.getFullPath();
    std::size_t pos = 0;
    while (pos != std::string::npos)
    {
        std::size_t newpos = filename.find('%',pos);
        oss << filename.substr(pos, (newpos == std::string::npos) ? std::string::npos : newpos-pos);
        pos = newpos;
        if(pos != std::string::npos)
        {
            ++pos;
            char c = filename[pos];
            ++pos;
            switch (c)
            {
            case 'n' : oss << m_stepCounter; break;
            case 'p' : oss << nbp; break;
            case 'E' : oss << nbce; break;
            case 'e' : oss << nbe; break;
            case '%' : oss << '%';
            default:
                msg_error() << "Invalid special character %" << c << " in filename.";
            }
        }
    }
    return getOrCreateTargetPath(oss.str(), d_exportEveryNbSteps.getValue()) + ext;
}

bool MeshExporter::writeMeshVTKXML()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    std::string filename = getMeshFilename(".vtu");

    std::ofstream outfile(filename.c_str());
    if (!outfile.is_open())
    {
        msg_error() << "Unable to create file '"<<filename << "'";
        return false;
    }

    outfile << std::setprecision (9);

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const int nbp = pointsPos.size();

    unsigned int numberOfCells;
    numberOfCells = ( (d_writeEdges.getValue()) ? m_inputtopology->getNbEdges() : 0 )
            +( (d_writeTriangles.getValue()) ? m_inputtopology->getNbTriangles() : 0 )
            +( (d_writeQuads.getValue()) ? m_inputtopology->getNbQuads() : 0 )
            +( (d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0 )
            +( (d_writeHexas.getValue()) ? m_inputtopology->getNbHexas() : 0 );

    //write header
    outfile << "<?xml version=\"1.0\"?>\n";
    outfile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n";
    outfile << "  <UnstructuredGrid>\n";
    //write piece
    outfile << "    <Piece NumberOfPoints=\"" << nbp << "\" NumberOfCells=\""<< numberOfCells << "\">\n";

    //write points
    outfile << "      <Points>\n";
    outfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << "          " << pointsPos[i] << "\n";
    }
    outfile << "        </DataArray>\n";
    outfile << "      </Points>\n";
    //write cells
    outfile << "      <Cells>\n";
    //write connectivity
    outfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
            outfile << "          " << m_inputtopology->getEdge(i) << "\n";
    }

    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            outfile << "          " <<  m_inputtopology->getTriangle(i) << "\n";
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
            outfile << "          " << m_inputtopology->getQuad(i) << "\n";
    }
    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
            outfile << "          " <<  m_inputtopology->getTetra(i) << "\n";
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
            outfile << "          " <<  m_inputtopology->getHexa(i) << "\n";
    }
    outfile << "        </DataArray>\n";
    //write offsets
    int num = 0;
    outfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    outfile << "          ";
    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
        {
            num += 2;
            outfile << num << ' ';
        }
    }
    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
        {
            num += 3;
            outfile << num << ' ';
        }
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
        {
            num += 4;
            outfile << num << ' ';
        }
    }
    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
        {
            num += 4;
            outfile << num << ' ';
        }
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
        {
            num += 6;
            outfile << num << ' ';
        }
    }
    outfile << "\n";
    outfile << "        </DataArray>\n";
    //write types
    outfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    outfile << "          ";
    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
            outfile << 3 << ' ';
    }
    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            outfile << 5 << ' ';
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
            outfile << 9 << ' ';
    }
    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
            outfile << 10 << ' ';
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
            outfile << 12 << ' ';
    }
    outfile << "\n";
    outfile << "        </DataArray>\n";
    outfile << "      </Cells>\n";

    //write end
    outfile << "    </Piece>\n";
    outfile << "  </UnstructuredGrid>\n";
    outfile << "</VTKFile>\n";
    outfile.close();
    msg_info() << filename << " written" ;
    return true;
}

bool MeshExporter::writeMeshVTK()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    std::string filename = getMeshFilename(".vtk");

    std::ofstream outfile(filename.c_str());
    if( !outfile.is_open() )
    {
        msg_error() << "Unable to create file '"<<filename << "'";
        return false;
    }

    outfile << std::setprecision (9);

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const int nbp = pointsPos.size();

    //Write header
    outfile << "# vtk DataFile Version 2.0\n";

    //write Title
    outfile << "SOFA Exported Mesh file\n";

    //write Data type
    outfile << "ASCII\n";

    //write dataset (geometry, unstructured grid)
    outfile << "DATASET " << "UNSTRUCTURED_GRID\n";

    outfile << "POINTS " << nbp << " float\n";

    //Write Points
    for (int i=0 ; i<nbp; i++)
    {
        outfile << pointsPos[i] << "\n";
    }

    //Write Cells
    unsigned int numberOfCells, totalSize;
    numberOfCells = ( (d_writeEdges.getValue()) ? m_inputtopology->getNbEdges() : 0 )
            +( (d_writeTriangles.getValue()) ? m_inputtopology->getNbTriangles() : 0 )
            +( (d_writeQuads.getValue()) ? m_inputtopology->getNbQuads() : 0 )
            +( (d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0 )
            +( (d_writeHexas.getValue()) ? m_inputtopology->getNbHexas() : 0 );
    totalSize =     ( (d_writeEdges.getValue()) ? 3 * m_inputtopology->getNbEdges() : 0 )
            +( (d_writeTriangles.getValue()) ? 4 *m_inputtopology->getNbTriangles() : 0 )
            +( (d_writeQuads.getValue()) ? 5 *m_inputtopology->getNbQuads() : 0 )
            +( (d_writeTetras.getValue()) ? 5 *m_inputtopology->getNbTetras() : 0 )
            +( (d_writeHexas.getValue()) ? 9 *m_inputtopology->getNbHexas() : 0 );


    outfile << "CELLS " << numberOfCells << ' ' << totalSize << "\n";

    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
            outfile << 2 << ' ' << m_inputtopology->getEdge(i) << "\n";
    }

    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            outfile << 3 << ' ' <<  m_inputtopology->getTriangle(i) << "\n";
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
            outfile << 4 << ' ' << m_inputtopology->getQuad(i) << "\n";
    }

    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
            outfile << 4 << ' ' <<  m_inputtopology->getTetra(i) << "\n";
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
            outfile << 8 << ' ' <<  m_inputtopology->getHexa(i) << "\n";
    }

    outfile << "CELL_TYPES " << numberOfCells << "\n";

    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
            outfile << 3 << "\n";
    }

    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            outfile << 5 << "\n";
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
            outfile << 9 << "\n";
    }

    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
            outfile << 10 << "\n";
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
            outfile << 12 << "\n";
    }
    msg_info() << filename << " written. " ;

    return true ;
}

/// http://geuz.org/gmsh/doc/texinfo/gmsh.html#File-formats
bool MeshExporter::writeMeshGmsh()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    std::string filename = getMeshFilename(".gmsh");

    std::ofstream outfile(filename.c_str());
    if( !outfile.is_open() )
    {
        msg_error() << "Unable to create file '"<<filename << "'";
        return false;
    }

    outfile << std::setprecision (9);

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const int nbp = pointsPos.size();

    //Write header
    outfile << "$MeshFormat\n";
    outfile << "2.1" << ' ' << 0 << ' ' << 8 << "\n";
    outfile << "$EndMeshFormat\n";

    //Write Points
    outfile << "$Nodes\n";

    outfile << nbp << "\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << 1+i << ' ' << pointsPos[i] << "\n";
    }

    outfile << "$EndNodes\n";

    //Write Cells
    outfile << "$Elements\n";
    unsigned int numberOfCells/*, totalSize*/;
    numberOfCells = ( (d_writeEdges.getValue()) ? m_inputtopology->getNbEdges() : 0 )
            +( (d_writeTriangles.getValue()) ? m_inputtopology->getNbTriangles() : 0 )
            +( (d_writeQuads.getValue()) ? m_inputtopology->getNbQuads() : 0 )
            +( (d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0 )
            +( (d_writeHexas.getValue()) ? m_inputtopology->getNbHexas() : 0 );
    /*totalSize =     ( (writeEdges.getValue()) ? 3 * topology->getNbEdges() : 0 )
            +( (writeTriangles.getValue()) ? 4 *topology->getNbTriangles() : 0 )
            +( (writeQuads.getValue()) ? 5 *topology->getNbQuads() : 0 )
            +( (writeTetras.getValue()) ? 5 *topology->getNbTetras() : 0 )
            +( (writeHexas.getValue()) ? 9 *topology->getNbHexas() : 0 );*/


    outfile << numberOfCells << "\n";
    unsigned int elem = 0;
    if (d_writeEdges.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbEdges() ; i++)
        {
            outfile << ++elem << ' ' << 1 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Edge t = m_inputtopology->getEdge(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    if (d_writeTriangles.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
        {
            outfile << ++elem << ' ' << 2 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Triangle t = m_inputtopology->getTriangle(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }
    if (d_writeQuads.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbQuads() ; i++)
        {
            outfile << ++elem << ' ' << 3 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Quad t = m_inputtopology->getQuad(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
        {
            outfile << ++elem << ' ' << 4 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Tetra t = m_inputtopology->getTetra(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }
    if (d_writeHexas.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbHexas() ; i++)
        {
            outfile << ++elem << ' ' << 5 << ' ' << 0;
            sofa::core::topology::BaseMeshTopology::Hexa t = m_inputtopology->getHexa(i);
            for (unsigned int j=0; j<t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    outfile << "$EndElements\n";

    msg_info() << filename << " written." ;
    return true ;
}

bool MeshExporter::writeMeshNetgen()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    std::string filename = getMeshFilename(".mesh");

    std::ofstream outfile(filename.c_str());
    if (!outfile.is_open())
    {
        msg_error() << "Unable to create file '"<<filename << "'";
        return false;
    }

    outfile << std::setprecision (9);

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    const int nbp = pointsPos.size();

    //Write Points
    outfile << nbp << "\n";
    for (int i=0 ; i<nbp; i++)
    {
        outfile << pointsPos[i] << "\n";
    }

    //Write Volume Elements
    outfile << ((d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0) << "\n";
    if (d_writeTetras.getValue())
    {
        for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
        {
            sofa::core::topology::BaseMeshTopology::Tetra t = m_inputtopology->getTetra(i);
            outfile << 0; // subdomain
            for (unsigned int j = 0; j < t.size(); ++j)
                outfile << ' ' << 1+t[j];
            outfile << "\n";
        }
    }

    //Write Surface Elements
    int nbtri = 0;
    if (d_writeTriangles.getValue())
    {
        if (m_inputtopology->getNbTetras() == 0)
        {
            nbtri += m_inputtopology->getNbTriangles();
        }
        else
        {
            for (int i=0; i<m_inputtopology->getNbTriangles(); ++i)
            {
                if (m_inputtopology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
            }
        }
    }
    outfile << nbtri << "\n";
    if (d_writeTriangles.getValue())
    {
        if (m_inputtopology->getNbTetras() == 0)
        {
            for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = m_inputtopology->getTriangle(i);
                outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        else
        {
            for (int i=0; i<m_inputtopology->getNbTriangles(); ++i)
            {
                if (m_inputtopology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = m_inputtopology->getTriangle(i);
                outfile << 0; // subdomain
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
    }

    msg_info() << filename << " written." ;
    return true;
}

/// http://tetgen.berlios.de/fformats.html
bool MeshExporter::writeMeshTetgen()
{
    if(m_componentstate!=ComponentState::Valid)
        return false;

    std::string filename = getMeshFilename(".node");

    std::ofstream outfile(filename.c_str());
    if(!outfile.is_open())
    {
        msg_error() << "Unable to create file '"<<filename << "'";
        return false;
    }

    outfile << std::setprecision (9);

    helper::ReadAccessor<Data<defaulttype::Vec3Types::VecCoord> > pointsPos = d_position;

    // Write Points

    const int nbp = pointsPos.size();

    // http://tetgen.berlios.de/fformats.node.html
    // <# of points> <dimension (must be 3)> <# of attributes> <# of boundary markers (0 or 1)>
    outfile << nbp << ' ' << 3 << ' ' << 0 << ' ' << 0 << "\n";
    // <point #> <x> <y> <z> [attributes] [boundary marker]
    for (int i=0 ; i<nbp; i++)
    {
        outfile << i+1 << ' ' << pointsPos[i] << "\n";
    }

    outfile.close();
    sout << filename << " written" << sendl;

    //Write Volume Elements

    if (d_writeTetras.getValue())
    {
        // http://tetgen.berlios.de/fformats.ele.html
        filename = getMeshFilename(".ele");
        std::ofstream outfile(filename.c_str());
        if (!outfile.is_open())
        {
            msg_error() << "Unable to create file '"<<filename << "'";
            return false;
        }
        // <# of tetrahedra> <nodes per tetrahedron> <# of attributes>
        outfile << ((d_writeTetras.getValue()) ? m_inputtopology->getNbTetras() : 0) << ' ' << 4 << ' ' << 0 << "\n";
        // <tetrahedron #> <node> <node> <node> <node> ... [attributes]
        if (d_writeTetras.getValue())
        {
            for (int i=0 ; i<m_inputtopology->getNbTetras() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Tetra t = m_inputtopology->getTetra(i);
                // check tetra inversion
                if (dot(pointsPos[t[1]]-pointsPos[t[0]],cross(pointsPos[t[2]]-pointsPos[t[0]],pointsPos[t[3]]-pointsPos[t[0]])) > 0)
                {
                    //sout << "Inverting tetra " << i << sendl;
                    unsigned int tmp = t[3]; t[3] = t[2]; t[2] = tmp;
                }

                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        outfile.close();
        sout << filename << " written" << sendl;
    }

    //Write Surface Elements
    if (d_writeTriangles.getValue())
    {
        // http://tetgen.berlios.de/fformats.face.html
        filename = getMeshFilename(".face");
        std::ofstream outfile(filename.c_str());
        if (!outfile.is_open())
        {
            msg_error() << "Unable to create file '"<<filename << "'";
            return false;
        }
        int nbtri = 0;
        if (d_writeTriangles.getValue())
        {
            if (m_inputtopology->getNbTetras() == 0)
            {
                nbtri += m_inputtopology->getNbTriangles();
            }
            else
            {
                for (int i=0; i<m_inputtopology->getNbTriangles(); ++i)
                {
                    if (m_inputtopology->getTetrahedraAroundTriangle(i).size() < 2)
                        ++nbtri;
                }
            }
        }
        // <# of faces> <boundary marker (0 or 1)>
        outfile << nbtri << ' ' << 0 << "\n";
        // <face #> <node> <node> <node> [boundary marker]
        if (m_inputtopology->getNbTetras() == 0)
        {
            for (int i=0 ; i<m_inputtopology->getNbTriangles() ; i++)
            {
                sofa::core::topology::BaseMeshTopology::Triangle t = m_inputtopology->getTriangle(i);
                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        else
        {
            for (int i=0; i<m_inputtopology->getNbTriangles(); ++i)
            {
                if (m_inputtopology->getTetrahedraAroundTriangle(i).size() < 2)
                    ++nbtri;
                sofa::core::topology::BaseMeshTopology::Triangle t = m_inputtopology->getTriangle(i);
                outfile << 1+i; // id
                for (unsigned int j = 0; j < t.size(); ++j)
                    outfile << ' ' << 1+t[j];
                outfile << "\n";
            }
        }
        outfile.close();
        msg_info() << filename << " written.";
    }
    return true;
}

void MeshExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *ev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            //todo(18.06) remove the behavior
            msg_deprecated() << "Hard coded interaction behavior in component is now a deprecated behavior."
                                "Scene specific interaction should be implement using an external controller or pythonScriptController."
                                "Please update your scene because this behavior will be removed in Sofa 18.06";
            writeMesh();
            break;
        }
    }
      BaseSimulationExporter::handleEvent(event);
}

} // namespace misc

} // namespace component

} // namespace sofa
