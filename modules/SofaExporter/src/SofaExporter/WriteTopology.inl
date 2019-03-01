/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_WRITETOPOLOGY_INL
#define SOFA_COMPONENT_MISC_WRITETOPOLOGY_INL

#include <SofaExporter/WriteTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <fstream>
#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{



WriteTopology::WriteTopology()
    : f_filename( initData(&f_filename, "filename", "output file name"))
    , f_writeContainers( initData(&f_writeContainers, true, "writeContainers", "flag enabling output of common topology containers."))
    , f_writeShellContainers( initData(&f_writeShellContainers, false, "writeShellContainers", "flag enabling output of specific shell topology containers."))
    , f_interval( initData(&f_interval, 0.0, "interval", "time duration between outputs"))
    , f_time( initData(&f_time, helper::vector<double>(0), "time", "set time to write outputs"))
    , f_period( initData(&f_period, 0.0, "period", "period between outputs"))
    , m_topology(NULL)
    , outfile(NULL)
    #ifdef SOFA_HAVE_ZLIB
    , gzfile(NULL)
    #endif
    , nextTime(0)
    , lastTime(0)
{
    this->f_listening.setValue(true);
}


WriteTopology::~WriteTopology()
{
    if (outfile)
        delete outfile;
#ifdef SOFA_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}


void WriteTopology::init()
{
    m_topology = this->getContext()->getMeshTopology();

    const std::string& filename = f_filename.getFullPath();

    if (filename.empty())
        return;

#ifdef SOFA_HAVE_ZLIB
    if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"wb");
        if( !gzfile )
        {
            msg_error() << "Unable to create the compressed file '"<<filename<<"'.";
        }
        return;
    }
#endif

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        msg_error() << "Unable to create the file "<<filename;
        delete outfile;
        outfile = NULL;
    }
}


void WriteTopology::reset()
{
    nextTime = 0;
    lastTime = 0;
}


void WriteTopology::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (!m_topology) return;
        if (!outfile
        #ifdef SOFA_HAVE_ZLIB
                && !gzfile
        #endif
                )
            return;

        SReal time = getContext()->getTime();

        bool writeCurrent = false;
        if (nextTime<f_time.getValue().size())
        {
            // store the actual time instant
            lastTime = f_time.getValue()[nextTime];
            if (time >= lastTime) // if the time simulation is >= that the actual time instant
            {
                writeCurrent = true;
                nextTime++;
            }
        }
        else
        {
            // write the topology using a period
            if (time >= (lastTime + f_period.getValue()))
            {
                writeCurrent = true;
                lastTime += f_period.getValue();
            }
        }


        if (writeCurrent)
        {
#ifdef SOFA_HAVE_ZLIB
            if (gzfile)
            {
                std::ostringstream str;
                str << "T= "<< time << "\n";

                // write the common containers:
                if (f_writeContainers.getValue())
                {
                    // - Points are already tested by mstate.

                    // - Edges
                    str << "  Edges= ";
                    str << m_topology->getNbEdges();
                    str << "\n";
                    str << m_topology->getEdges();
                    str << "\n";

                    // - Triangles
                    str << "  Triangles= ";
                    str << m_topology->getNbTriangles();
                    str << "\n";
                    str << m_topology->getTriangles();
                    str << "\n";

                    // - Quads
                    str << "  Quads= ";
                    str << m_topology->getNbQuads();
                    str << "\n";
                    str << m_topology->getQuads();
                    str << "\n";

                    // - Tetrahedra
                    str << "  Tetrahedra= ";
                    str << m_topology->getNbTetrahedra();
                    str << "\n";
                    str << m_topology->getTetrahedra();
                    str << "\n";

                    // - Hexahedra
                    str << "  Hexahedra= ";
                    str << m_topology->getNbHexahedra();
                    str << "\n";
                    str << m_topology->getHexahedra();
                    str << "\n";
                }

                // write the shell containers:
                if (f_writeShellContainers.getValue())
                {
                    str << "  Writing shell not handle yet.\n";
                }

                gzputs(gzfile, str.str().c_str());
                gzflush(gzfile, Z_SYNC_FLUSH);
            }
            else
#endif
                if (outfile)
                {
                    (*outfile) << "T= "<< time << "\n";

                    // write the common containers:
                    if (f_writeContainers.getValue())
                    {
                        // - Points are already tested by mstate.

                        // - Edges
                        (*outfile) << "  Edges= ";
                        (*outfile) << m_topology->getNbEdges();
                        (*outfile) << "\n";
                        (*outfile) << m_topology->getEdges();
                        (*outfile) << "\n";

                        // - Triangles
                        (*outfile) << "  Triangles= ";
                        (*outfile) << m_topology->getNbTriangles();
                        (*outfile) << "\n";
                        (*outfile) << m_topology->getTriangles();
                        (*outfile) << "\n";

                        // - Quads
                        (*outfile) << "  Quads= ";
                        (*outfile) << m_topology->getNbQuads();
                        (*outfile) << "\n";
                        (*outfile) << m_topology->getQuads();
                        (*outfile) << "\n";

                        // - Tetrahedra
                        (*outfile) << "  Tetrahedra= ";
                        (*outfile) << m_topology->getNbTetrahedra();
                        (*outfile) << "\n";
                        (*outfile) << m_topology->getTetrahedra();
                        (*outfile) << "\n";

                        // - Hexahedra
                        (*outfile) << "  Hexahedra= ";
                        (*outfile) << m_topology->getNbHexahedra();
                        (*outfile) << "\n";
                        (*outfile) << m_topology->getHexahedra();
                        (*outfile) << "\n";
                    }

                    // write the shell containers:
                    if (f_writeShellContainers.getValue())
                    {
                        (*outfile) << "  Writing shell not handle yet.\n";
                    }

                    outfile->flush();
                }
        }
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
