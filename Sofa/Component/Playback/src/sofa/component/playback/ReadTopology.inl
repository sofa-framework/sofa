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
#include <sofa/component/playback/ReadTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/helper/system/FileSystem.h>

#include <cstring>
#include <sstream>

namespace sofa::component::playback
{

ReadTopology::ReadTopology()
    : d_filename(initData(&d_filename, "filename", "input file name"))
    , d_interval(initData(&d_interval, 0.0, "interval", "time duration between inputs"))
    , d_shift(initData(&d_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , d_loop(initData(&d_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , m_topology(nullptr)
    , infile(nullptr)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , gzfile(nullptr)
#endif
    , nextTime(0.0)
    , lastTime(0.0)
    , loopTime(0.0)
{
    this->f_listening.setValue(true);
    f_filename.setParent(&d_filename);
    f_interval.setOriginalData(&d_interval);
    f_shift.setOriginalData(&d_shift);
    f_loop.setOriginalData(&d_loop);
}

ReadTopology::~ReadTopology()
{
    if (infile)
        delete infile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}

void ReadTopology::init()
{

    reset();
}

void ReadTopology::reset()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
    
    if (infile)
    {
        delete infile;
        infile = nullptr;
    }

#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    if (gzfile)
    {
        gzclose(gzfile);
        gzfile = nullptr;
    }
#endif

    const std::string& filename = d_filename.getFullPath();
    if (filename.empty())
    {
        msg_error() << "Empty filename";
    }
    else if (!sofa::helper::system::FileSystem::exists(filename))
    {
        msg_error() << "Compressed file doesn't exist:" << filename;
    }
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            msg_error() << "Opening compressed file " << filename;
        }
    }
#endif
    else
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            msg_error() << "Opening file " << filename;
            delete infile;
            infile = nullptr;
        }
    }
    nextTime = 0;
}

void ReadTopology::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        processReadTopology();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {

    }
}



void ReadTopology::setTime(double time)
{
    if (time < nextTime) {reset(); nextTime=0.0; loopTime=0.0; }
}

void ReadTopology::processReadTopology(double time)
{
    if (time == lastTime) return;
    setTime(time);
    processReadTopology();
}

bool ReadTopology::readNext(double time, std::vector<std::string>& validLines)
{
    if (!m_topology) return false;
    if (!infile
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
        && !gzfile
#endif
       )
        return false;
    lastTime = time;
    validLines.clear();
    std::string line, cmd;

    const double epsilon = 0.00000001;


    while ((double)nextTime <= (time + epsilon))
    {

#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
        if (gzfile)
        {

            if (gzeof(gzfile))
            {
                if (!d_loop.getValue())
                    break;
                gzrewind(gzfile);
                loopTime = nextTime;
            }
            //getline(gzfile, line);
            line.clear();
            char buf[4097];
            buf[0] = '\0';
            while (gzgets(gzfile,buf,sizeof(buf))!=nullptr && buf[0])
            {
                const size_t l = strlen(buf);
                if (buf[l-1] == '\n')
                {
                    buf[l-1] = '\0';
                    line += buf;
                    break;
                }
                else
                {
                    line += buf;
                    buf[0] = '\0';
                }
            }
        }
        else
#endif
            if (infile)
            {
                if (infile->eof())
                {
                    if (!d_loop.getValue())
                        break;
                    infile->clear();
                    infile->seekg(0);
                    loopTime = nextTime;
                }
                getline(*infile, line);
            }
        std::istringstream str(line);
        str >> cmd;
        if (cmd == "T=")
        {
            str >> nextTime;

            nextTime += loopTime;
            if (nextTime <= time)
                validLines.clear();
        }

        if (nextTime <= time)
            validLines.push_back(line);
    }
    return true;
}

void ReadTopology::processReadTopology()
{
    double time = getContext()->getTime() + d_shift.getValue();

    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;

    unsigned int nbr = 0;
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end();)
    {
        //For one Timestep store all topology data available.

        std::string buff;
        std::istringstream str(*it);

        str >> buff;

        if (buff == "T=")
        {
            //Nothing to do in this case.
            ++it;
            continue;
        }
        else if ( buff == "Edges=")
        {
            //Looking fo the number of edges, if not null, then we store them..
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                type::vector<type::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());
                std::istringstream Sedges(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    type::fixed_array <unsigned int,2> nodes;
                    Sedges >> nodes[0] >> nodes[1];

                    my_edges.push_back (nodes);
                }
                edges.endEdit();
            }

            ++it;
            continue;
        }
        else if ( buff == "Triangles=")
        {
            //Looking fo the number of Triangles, if not null, then we store them..
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                type::vector<type::fixed_array <unsigned int,3> >& my_triangles = *(triangles.beginEdit());
                std::istringstream Stri(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    type::fixed_array <unsigned int,3> nodes;
                    Stri >> nodes[0] >> nodes[1] >> nodes[2];

                    my_triangles.push_back (nodes);
                }
                triangles.endEdit();
            }

            ++it;
            continue;
        }
        else if ( buff == "Quads=")
        {
            //Looking fo the number of Quads, if not null, then we store them..
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                type::vector<type::fixed_array <unsigned int,4> >& my_quads = *(quads.beginEdit());
                std::istringstream Squads(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    type::fixed_array <unsigned int,4> nodes;
                    Squads >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3];

                    my_quads.push_back (nodes);
                }
                quads.endEdit();
            }

            ++it;
            continue;
        }
        else if ( buff == "Tetrahedra=")
        {
            //Looking fo the number of Tetrahedra, if not null, then we store them..
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                type::vector<type::fixed_array <unsigned int,4> >& my_tetrahedra = *(tetrahedra.beginEdit());
                std::istringstream Stetra(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    type::fixed_array <unsigned int,4> nodes;
                    Stetra >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3];

                    my_tetrahedra.push_back (nodes);
                }
                tetrahedra.endEdit();
            }

            ++it;
            continue;
        }
        else if ( buff == "Hexahedra=")
        {
            //Looking fo the number of Hexahedra, if not null, then we store them..
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                type::vector<type::fixed_array <unsigned int,8> >& my_hexahedra = *(hexahedra.beginEdit());
                std::istringstream Shexa(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    type::fixed_array <unsigned int,8> nodes;
                    Shexa >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> nodes[4] >> nodes[5] >> nodes[6] >> nodes[7];

                    my_hexahedra.push_back (nodes);
                }
                hexahedra.endEdit();
            }

            ++it;
            continue;
        }
        else
        {
            ++it;
            continue;
        }

    }

}

} //namespace sofa::component::playback
