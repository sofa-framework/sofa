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
#ifndef SOFA_COMPONENT_MISC_READTOPOLOGY_INL
#define SOFA_COMPONENT_MISC_READTOPOLOGY_INL

#include <SofaGeneralLoader/ReadTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <string.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{

ReadTopology::ReadTopology()
    : f_filename( initData(&f_filename, "filename", "input file name"))
    , f_interval( initData(&f_interval, 0.0, "interval", "time duration between inputs"))
    , f_shift( initData(&f_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , f_loop( initData(&f_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , m_topology(NULL)
    , infile(NULL)
#ifdef SOFA_HAVE_ZLIB
    , gzfile(NULL)
#endif
    , nextTime(0.0)
    , lastTime(0.0)
    , loopTime(0.0)
{
    this->f_listening.setValue(true);
}

ReadTopology::~ReadTopology()
{
    if (infile)
        delete infile;
#ifdef SOFA_HAVE_ZLIB
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
    m_topology = this->getContext()->getMeshTopology();
    if (infile)
    {
        delete infile;
        infile = NULL;
    }
#ifdef SOFA_HAVE_ZLIB
    if (gzfile)
    {
        gzclose(gzfile);
        gzfile = NULL;
    }
#endif

    const std::string& filename = f_filename.getFullPath();
    if (filename.empty())
    {
        serr << "ERROR: empty filename"<<sendl;
    }
#ifdef SOFA_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            serr << "Error opening compressed file "<<filename<<sendl;
        }
    }
#endif
    else
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            serr << "Error opening file "<<filename<<sendl;
            delete infile;
            infile = NULL;
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
#ifdef SOFA_HAVE_ZLIB
        && !gzfile
#endif
       )
        return false;
    lastTime = time;
    validLines.clear();
    std::string line, cmd;

    double epsilon = 0.00000001;


    while ((double)nextTime <= (time + epsilon))
    {

#ifdef SOFA_HAVE_ZLIB
        if (gzfile)
        {

            if (gzeof(gzfile))
            {
                if (!f_loop.getValue())
                    break;
                gzrewind(gzfile);
                loopTime = nextTime;
            }
            //getline(gzfile, line);
            line.clear();
            char buf[4097];
            buf[0] = '\0';
            while (gzgets(gzfile,buf,sizeof(buf))!=NULL && buf[0])
            {
                size_t l = strlen(buf);
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
                    if (!f_loop.getValue())
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
    double time = getContext()->getTime() + f_shift.getValue();

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
                helper::vector<helper::fixed_array <unsigned int,2> >& my_edges = *(edges.beginEdit());
                std::istringstream Sedges(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    helper::fixed_array <unsigned int,2> nodes;
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
                helper::vector<helper::fixed_array <unsigned int,3> >& my_triangles = *(triangles.beginEdit());
                std::istringstream Stri(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    helper::fixed_array <unsigned int,3> nodes;
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
                helper::vector<helper::fixed_array <unsigned int,4> >& my_quads = *(quads.beginEdit());
                std::istringstream Squads(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    helper::fixed_array <unsigned int,4> nodes;
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
                helper::vector<helper::fixed_array <unsigned int,4> >& my_tetrahedra = *(tetrahedra.beginEdit());
                std::istringstream Stetra(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    helper::fixed_array <unsigned int,4> nodes;
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
                helper::vector<helper::fixed_array <unsigned int,8> >& my_hexahedra = *(hexahedra.beginEdit());
                std::istringstream Shexa(*it);
                for (unsigned int i = 0; i<nbr; ++i)
                {
                    helper::fixed_array <unsigned int,8> nodes;
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

} // namespace misc

} // namespace component

} // namespace sofa

#endif
