/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/misc/TopologicalChangeProcessor.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/misc/TopologicalChangeProcessor.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/component/topology/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/TriangleSetTopologyAlgorithms.h>
#include <sofa/component/topology/QuadSetTopologyModifier.h>
#include <sofa/component/topology/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/HexahedronSetTopologyModifier.h>


namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(TopologicalChangeProcessor)

using namespace defaulttype;



int TopologicalChangeProcessorClass = core::RegisterObject("Read topological Changes and process them.")
        .add< TopologicalChangeProcessor >();


TopologicalChangeProcessor::TopologicalChangeProcessor()
    : m_filename( initData(&m_filename, "filename", "input file name for topological changes."))
    , m_listChanges (initData (&m_listChanges, "listChanges", "0 for adding, 1 for removing, 2 for cutting and associated indices."))
    , m_interval( initData(&m_interval, 0.0, "interval", "time duration between outputs"))
    , m_shift( initData(&m_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , m_loop( initData(&m_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , m_topology(NULL)
    , infile(NULL)
#ifdef SOFA_HAVE_ZLIB
    , gzfile(NULL)
#endif
    , nextTime(0)
    , lastTime(0)
    , loopTime(0)
{
    this->f_listening.setValue(true);
}


TopologicalChangeProcessor::~TopologicalChangeProcessor()
{
    if (infile)
        delete infile;
#ifdef SOFA_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}


void TopologicalChangeProcessor::init()
{
    m_topology = dynamic_cast<core::componentmodel::topology::BaseMeshTopology*>(this->getContext()->getMeshTopology());
    //for (unsigned int i = 0 ; i < 243; i++)
    //std::cout << i << " " ;

    reset();
}

void TopologicalChangeProcessor::reinit()
{
    reset();
}



void TopologicalChangeProcessor::reset()
{
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

    const std::string& filename = m_filename.getFullPath();
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
    lastTime = 0;
    loopTime = 0;
}


void TopologicalChangeProcessor::setTime(double time)
{
    if (time < nextTime) {reset();}
}


void TopologicalChangeProcessor::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */ dynamic_cast<simulation::AnimateBeginEvent*>(event))
    {
        processTopologicalChanges();
    }
    if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {

    }
}


void TopologicalChangeProcessor::processTopologicalChanges(double time)
{
    if (time == lastTime) return;
    setTime(time);
    processTopologicalChanges();
}


bool TopologicalChangeProcessor::readNext(double time, std::vector<std::string>& validLines)
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
    while (nextTime <= time)
    {
#ifdef SOFA_HAVE_ZLIB
        if (gzfile)
        {
            if (gzeof(gzfile))
            {
                if (!m_loop.getValue())
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
                int l = strlen(buf);
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
                    if (!m_loop.getValue())
                        break;
                    infile->clear();
                    infile->seekg(0);
                    loopTime = nextTime;
                }
                getline(*infile, line);
            }
        //sout << "line= "<<line<<sendl;
        std::istringstream str(line);
        str >> cmd;
        if (cmd == "T=")
        {
            str >> nextTime;
            nextTime += loopTime;
            //sout << "next time: " << nextTime << sendl;
            if (nextTime <= time)
                validLines.clear();
        }

        if (nextTime <= time)
            validLines.push_back(line);
    }
    return true;
}




void TopologicalChangeProcessor::processTopologicalChanges()
{

    double time = getContext()->getTime() + m_shift.getValue();
    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;

    unsigned int nbr = 0;
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end();)
    {
        //For one Timestep store all topology data available.

        //std::cout <<"Line: " << (*it) << std::endl;

        std::string buff;
        std::istringstream str(*it);

        str >> buff;

        if (buff == "T=")
        {
            //Nothing to do in this case.
            std::cout << "cas T" << std::endl;
            ++it;
            continue;
        }
        else if ( buff == "ADD=")
        {
            /*	//Looking fo the number of element to add:
            str >> nbr;
            ++it;

            sofa::core::componentmodel::topology::TopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);


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
            */

            std::cout << "Adding element not handle yet (buffer size problem) " << std::endl;

            ++it;
            continue;
        }
        else if ( buff == "REMOVE=" )
        {
            //Looking fo the number of element to add:
            str >> nbr;
            ++it;

            sofa::core::componentmodel::topology::TopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);

            helper::vector <unsigned int> vitems;
            vitems.resize (nbr);

            for (unsigned int i = 0; i<nbr; ++i)
                str >> vitems[i];

            topoMod->removeItems(vitems);

            //TODO: check  Cas des mappings volume to surface. Il ny a pas suppression des element surfaceique isole.

            ++it;
            continue;
        }
        else if ( buff == "INCISE=" )
        {

            sofa::component::topology::TriangleSetTopologyModifier* triangleMod;
            m_topology->getContext()->get(triangleMod);

            sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
            m_topology->getContext()->get(triangleAlg);

            sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
            m_topology->getContext()->get(triangleGeo);

            Vector3 a;
            Vector3 b;
            unsigned int ind_ta;
            unsigned int ind_tb;
            unsigned int a_last = -1;
            unsigned int b_last = -1;
            bool firstCut= true;

            str >> nbr;
            ++it;
            std::istringstream Sin(*it);

            Sin >> ind_ta;

            for (unsigned int j = 0; j<3; ++j)
                Sin >> a[j];

            for (unsigned int i =1; i <nbr; ++i)
            {
                Sin >> ind_tb;
                for (unsigned int j = 0; j<3; ++j)
                    Sin >> b[j];


                // Output declarations
                sofa::helper::vector< sofa::core::componentmodel::topology::TopologyObjectType> topoPath_list;
                sofa::helper::vector<unsigned int> indices_list;
                sofa::helper::vector< Vec<3, double> > coords2_list;

                if(firstCut)
                    a_last = (unsigned int)-1;
                else
                {
                    core::componentmodel::behavior::MechanicalState<Vec3Types>* mstate = m_topology->getContext()->get<core::componentmodel::behavior::MechanicalState<Vec3Types> >();
                    helper::vector<Vector3> &v_coords =  *mstate->getX();
                    a = v_coords[a_last];
                }


                bool ok = triangleGeo->computeIntersectedObjectsList(a_last, a, b, ind_ta, ind_tb, topoPath_list, indices_list, coords2_list);


                if (!ok)
                {
                    std::cout << "ERROR in computeIntersectedPointsList" << std::endl;
                    break;
                }

                sofa::helper::vector< unsigned int > new_edges;

                triangleAlg->SplitAlongPath(a_last, a, b_last, b, topoPath_list, indices_list, coords2_list, new_edges, 0.1, 0.25);

                sofa::helper::vector<unsigned int> new_points;
                sofa::helper::vector<unsigned int> end_points;
                bool reachBorder = false;

                triangleAlg->InciseAlongEdgeList(new_edges, new_points, end_points, reachBorder);

                if (!end_points.empty())
                {
                    a_last = end_points.back();
                }
                ind_ta = ind_tb;
                firstCut=false;

                triangleMod->propagateTopologicalChanges();
                // notify the end for the current sequence of topological change events
                triangleMod->notifyEndingEvent();

                triangleMod->propagateTopologicalChanges();
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


    /*
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
    */
}





} // namespace misc

} // namespace component

} // namespace sofa
