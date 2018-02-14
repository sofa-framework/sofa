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
#include <SofaMiscTopology/TopologicalChangeProcessor.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/QuadSetTopologyModifier.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>
#include <SofaBaseTopology/TetrahedronSetTopologyModifier.h>
#include <SofaBaseTopology/HexahedronSetTopologyModifier.h>
#include <SofaBaseTopology/EdgeSetTopologyModifier.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/simulation/Simulation.h>

#include <time.h>

#ifndef NDEBUG
    #define DEBUG_MSG true
#else
    #define DEBUG_MSG false
#endif

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
    , m_interval( initData(&m_interval, 0.0, "interval", "time duration between 2 actions"))
    , m_shift( initData(&m_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , m_loop( initData(&m_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , m_useDataInputs( initData(&m_useDataInputs, false, "useDataInputs", "If true, will perform operation using Data input lists rather than text file."))
    , m_timeToRemove( initData(&m_timeToRemove, 0.0, "timeToRemove", "If using option useDataInputs, time at which will be done the operations. Possibility to use the interval Data also."))
    , m_edgesToRemove (initData (&m_edgesToRemove, "edgesToRemove", "List of edge IDs to be removed."))
    , m_trianglesToRemove (initData (&m_trianglesToRemove, "trianglesToRemove", "List of triangle IDs to be removed."))
    , m_quadsToRemove (initData (&m_quadsToRemove, "quadsToRemove", "List of quad IDs to be removed."))
    , m_tetrahedraToRemove (initData (&m_tetrahedraToRemove, "tetrahedraToRemove", "List of tetrahedron IDs to be removed."))
    , m_hexahedraToRemove (initData (&m_hexahedraToRemove, "hexahedraToRemove", "List of hexahedron IDs to be removed."))
    , m_saveIndicesAtInit( initData(&m_saveIndicesAtInit, false, "saveIndicesAtInit", "set to 'true' to save the incision to do in the init to incise even after a movement"))
    , m_epsilonSnapPath( initData(&m_epsilonSnapPath, (Real)0.1, "epsilonSnapPath", "epsilon snap path"))
    , m_epsilonSnapBorder( initData(&m_epsilonSnapBorder, (Real)0.25, "epsilonSnapBorder", "epsilon snap path"))
    , m_draw( initData(&m_draw, false, "draw", "draw information"))
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
    m_topology = this->getContext()->getMeshTopology();

    if (!m_useDataInputs.getValue())
        this->readDataFile();
}

void TopologicalChangeProcessor::reinit()
{
    if (!m_useDataInputs.getValue())
        this->readDataFile();
}



void TopologicalChangeProcessor::readDataFile()
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
        serr << "TopologicalChangeProcessor: ERROR: empty filename"<<sendl;
    }
#ifdef SOFA_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            serr << "TopologicalChangeProcessor: Error opening compressed file "<<filename<<sendl;
        }
    }
#endif
    else
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            serr << "TopologicalChangeProcessor: Error opening file "<<filename<<sendl;
            delete infile;
            infile = NULL;
        }
    }
    nextTime = 0;
    lastTime = 0;
    loopTime = 0;

    if (m_saveIndicesAtInit.getValue())
        saveIndices();

    return;
}


void TopologicalChangeProcessor::setTime(double time)
{
    if (time < nextTime)
    {
        if (!m_useDataInputs.getValue())
            this->readDataFile();
    }
}


void TopologicalChangeProcessor::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        if (m_useDataInputs.getValue())
            processTopologicalChanges(this->getTime());
        else
            processTopologicalChanges();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {

    }
}


void TopologicalChangeProcessor::processTopologicalChanges(double time)
{
    if (!m_useDataInputs.getValue())
    {
        if (time == lastTime) return;
        setTime(time);
        processTopologicalChanges();
    }
    else
    {
        if (m_timeToRemove.getValue() >= time)
            return;

        // process topological changes
        helper::ReadAccessor< Data<helper::vector<unsigned int> > > edges = m_edgesToRemove;
        helper::ReadAccessor< Data<helper::vector<unsigned int> > > triangles = m_trianglesToRemove;
        helper::ReadAccessor< Data<helper::vector<unsigned int> > > quads = m_quadsToRemove;
        helper::ReadAccessor< Data<helper::vector<unsigned int> > > tetrahedra = m_tetrahedraToRemove;
        helper::ReadAccessor< Data<helper::vector<unsigned int> > > hexahedra = m_hexahedraToRemove;

        if (!hexahedra.empty())
        {
            sofa::component::topology::HexahedronSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            helper::vector <unsigned int> vitems;
            vitems.assign(hexahedra.begin(), hexahedra.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                serr<< "TopologicalChangeProcessor: Error: No HexahedraTopology available" << sendl;
        }

        if (!tetrahedra.empty())
        {
            sofa::component::topology::TetrahedronSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            helper::vector <unsigned int> vitems;
            vitems.assign(tetrahedra.begin(), tetrahedra.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                serr<< "TopologicalChangeProcessor: Error: No TetrahedraTopology available" << sendl;
        }

        if (!quads.empty())
        {
            sofa::component::topology::QuadSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            helper::vector <unsigned int> vitems;
            vitems.assign(quads.begin(), quads.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                serr<< "TopologicalChangeProcessor: Error: No QuadTopology available" << sendl;
        }

        if (!triangles.empty())
        {
            sofa::component::topology::TriangleSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            sofa::helper::vector <unsigned int> vitems;
            vitems.assign(triangles.begin(), triangles.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                serr<< "TopologicalChangeProcessor: Error: No TriangleTopology available" << sendl;
        }

        if (!edges.empty())
        {
            sofa::component::topology::EdgeSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            helper::vector <unsigned int> vitems;
            vitems.assign(edges.begin(), edges.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                serr<< "TopologicalChangeProcessor: Error: No EdgeTopology available" << sendl;
        }

        // iterate, time set to infini if no interval.
        double& newTime = *m_timeToRemove.beginEdit();
        if (m_interval.getValue() != 0.0)
            newTime += m_interval.getValue();
        else
            newTime = (unsigned int)-1;
        m_timeToRemove.endEdit();
    }
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
    double epsilon = 1e-10;
    while (nextTime < time || fabs(nextTime - time) < epsilon )
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
        std::istringstream str(line);
        str >> cmd;
        if (cmd == "T=")
        {
            str >> nextTime;
            nextTime += loopTime;
            if (nextTime < time || fabs(nextTime - time) < epsilon)
                validLines.clear();
        }

        if (nextTime < time || fabs(nextTime - time) < epsilon)
            validLines.push_back(line);
    }
    return true;
}




void TopologicalChangeProcessor::processTopologicalChanges()
{
    double time = getContext()->getTime() + m_shift.getValue();
    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;

    unsigned int nbElements = 0;
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
        else if ( buff == "ADD=")
        {
            //Looking for the type of element.
            std::string EleType;
            str >> EleType;

            //Looking for the number of element to add:
            str >> nbElements;
            ++it;

            std::istringstream Sin(*it);

            if ( EleType == "PointInTriangle" || EleType == "PointsInTriangle")
            {
                sofa::component::topology::TriangleSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if(!topoMod)
                {
                    serr << "No PointSetTopologyModifier available" << sendl;
                    continue;
                }

                helper::vector< Vector2 > baryCoords;
                baryCoords.resize(nbElements);
                helper::vector < unsigned int > triangles;
                triangles.resize(nbElements);

                for(unsigned int i=0;i<nbElements;++i)
                {
                    Sin >> triangles[i];
                    Vector2& baryCoord = baryCoords[i];
                    Sin >> baryCoord[0] >> baryCoord[1];
                }


                helper::vector< helper::vector< unsigned int > > p_ancestors(nbElements);
                sofa::helper::vector< helper::vector< double > > p_baryCoefs(nbElements);
                for(unsigned int i=0; i<nbElements; ++i)
                {
                    helper::vector<unsigned int>& ancestor = p_ancestors[i];
                    ancestor.resize(3);
                    const core::topology::BaseMeshTopology::Triangle& t = m_topology->getTriangle( triangles[i] );
                    ancestor[0] = t[0];
                    ancestor[1] = t[1];
                    ancestor[2] = t[2];
                    helper::vector<double>& baryCoef = p_baryCoefs[i];
                    baryCoef.resize(3);
                    baryCoef[0] = baryCoords[i][0];
                    baryCoef[1] = baryCoords[i][1];
                    baryCoef[2] = 1 - baryCoef[0] - baryCoef[1];
                }
                topoMod->addPoints(nbElements, p_ancestors, p_baryCoefs, true);

            }
            else if ( EleType == "Edge" || EleType == "Edges")
            {
                sofa::component::topology::EdgeSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    serr<< "TopologicalChangeProcessor: Error: No QuadTopology available" << sendl;
                    continue;
                }

                helper::vector<core::topology::Topology::Edge > vitems;
                vitems.resize (nbElements);

                for (unsigned int i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1];

                topoMod->addEdges(vitems);

            }
            else if ( EleType == "Triangle" || EleType == "Triangles")
            {

                helper::vector<helper::vector<unsigned int> >  p_ancestors(nbElements);
                helper::vector<helper::vector<double> >        p_baryCoefs(nbElements);

                if(!str.eof() )
                {
                    std::string token;
                    str >> token;
                    if(token == "Ancestors" || token == "Ancestor")
                    {
                        for(unsigned int i = 0; i<nbElements; ++i)
                        {
                            helper::vector<unsigned int>& ancestor = p_ancestors[i];
                            ancestor.resize(1);
                            str >> ancestor[0];
                        }
                    }
                }

                sofa::component::topology::TriangleSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    serr<< "TopologicalChangeProcessor: Error: No TriangleTopology available" << sendl;
                    continue;
                }

                helper::vector<core::topology::Topology::Triangle > vitems;
                vitems.resize (nbElements);

                for (unsigned int i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2];

                if(!p_ancestors.empty())
                {
                    topoMod->addTriangles(vitems,p_ancestors,p_baryCoefs);
                }
                else
                {
                    topoMod->addTriangles(vitems);
                }
            }
            else if ( EleType == "Quad" || EleType == "Quads")
            {
                sofa::component::topology::QuadSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    serr<< "TopologicalChangeProcessor: Error: No QuadTopology available" << sendl;
                    continue;
                }

                helper::vector<core::topology::Topology::Quad > vitems;
                vitems.resize (nbElements);

                for (unsigned int i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3];

                topoMod->addQuads(vitems);
            }
            else if ( EleType == "Tetrahedron" || EleType == "Tetrahedra")
            {
                sofa::component::topology::TetrahedronSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    serr<< "TopologicalChangeProcessor: Error: No TetrahedraTopology available" << sendl;
                    continue;
                }

                helper::vector<core::topology::Topology::Tetrahedron > vitems;
                vitems.resize (nbElements);

                for (unsigned int i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3];

                topoMod->addTetrahedra(vitems);
            }
            else if ( EleType == "Hexahedron" || EleType == "Hexahedra")
            {
                sofa::component::topology::HexahedronSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    serr<< "TopologicalChangeProcessor: Error: No HexahedraTopology available" << sendl;
                    continue;
                }

                helper::vector<core::topology::Topology::Hexahedron > vitems;
                vitems.resize (nbElements);

                for (unsigned int i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3]
                        >> vitems[i][4] >> vitems[i][5] >> vitems[i][6] >> vitems[i][7];

                topoMod->addHexahedra(vitems);
            }
            else
            {
                serr<< "TopologicalChangeProcessor: Error: keyword: '" << EleType <<"' not expected."<< sendl;
                continue;
            }

            ++it;
            continue;
        }
        else if ( buff == "REMOVE=" )
        {
            //Looking fo the number of element to add:
            str >> nbElements;
            ++it;

            std::istringstream Sin(*it);

            sofa::core::topology::TopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);

            helper::vector <unsigned int> vitems;
            vitems.resize (nbElements);

            for (unsigned int i = 0; i<nbElements; ++i)
                Sin >> vitems[i];

            topoMod->removeItems(vitems);

            //TODO: check  Cas des mappings volume to surface. Il ny a pas suppression des element surfaceique isole.

            ++it;
            continue;
        }
        else if ( buff == "INCISE=" )
        {
            msg_info() << "processTopologicalChanges: about to make a incision with time = " << time ;

            if (m_saveIndicesAtInit.getValue())
            {
                inciseWithSavedIndices();
                ++it; ++it; continue;
            }

            sofa::component::topology::TriangleSetTopologyModifier* triangleMod;
            m_topology->getContext()->get(triangleMod);

            sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
            m_topology->getContext()->get(triangleAlg);

            sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
            m_topology->getContext()->get(triangleGeo);

            Vector3 a;
            Vector3 b;
            int ind_ta;
            int ind_tb;
            unsigned int a_last = core::topology::BaseMeshTopology::InvalidID;
            unsigned int b_last = core::topology::BaseMeshTopology::InvalidID;
            bool firstCut = true;

            //get the number of element
            str >> nbElements;

            ++it;//go to the next line

            //get the values in the current line and put them in a vector
            std::vector<Real> values = getValuesInLine(*it, nbElements);
            bool onlyCoordinates = (values.size() == nbElements * 3);

            std::istringstream Sin(*it);

            if (!onlyCoordinates)
            {
                Sin >> ind_ta;//get the first index of triangle
            }
            else
            {
                nbElements = values.size() / 3;
            }

            for (unsigned int j = 0; j < 3; ++j)
                Sin >> a[j];

            if (onlyCoordinates)
            {
                //find the triangle corresponding to the current coordinates
                findElementIndex(a, ind_ta, -1);
            }

            for (unsigned int i = 1; i < nbElements; ++i)
            {
                if (!onlyCoordinates)
                {
                    Sin >> ind_tb;//get the second index of triangle
                }
                for (unsigned int j = 0; j < 3; ++j)
                    Sin >> b[j];

                if (onlyCoordinates)
                {
                    findElementIndex(b, ind_tb, -1);
                }

                // Output declarations
                sofa::helper::vector<sofa::core::topology::TopologyObjectType>       topoPath_list;
                sofa::helper::vector<unsigned int> indices_list;
                sofa::helper::vector<Vec<3, double> > coords2_list;

                if (firstCut)
                    a_last
                        = core::topology::BaseMeshTopology::InvalidID;
                else
                {
                    core::behavior::MechanicalState<Vec3Types> * mstate =
                        m_topology->getContext()->get< core::behavior::MechanicalState<Vec3Types> > ();
                    //get the coordinates of the mechanical state
                    const helper::vector<Vector3> &v_coords = mstate->read(core::ConstVecCoordId::position())->getValue();
                    a = v_coords[a_last];
                }

                //Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
                unsigned int uInd_ta = (unsigned int) ind_ta;
                unsigned int uInd_tb = (unsigned int) ind_tb;

                bool isPathOk =
                    triangleGeo->computeIntersectedObjectsList(
                            a_last, a, b, uInd_ta, uInd_tb,
                            topoPath_list, indices_list,
                            coords2_list);

                if (!isPathOk)
                {
                    dmsg_error() << "Invalid path in computeIntersectedPointsList" ;
                    break;
                }

                sofa::helper::vector<unsigned int> new_edges;

                //Split triangles to create edges along a path given as a the list of existing edges and triangles crossed by it.
                triangleAlg->SplitAlongPath(a_last, a, b_last, b,
                        topoPath_list, indices_list, coords2_list,
                        new_edges, 0.1, 0.25);

                sofa::helper::vector<unsigned int> new_points;
                sofa::helper::vector<unsigned int> end_points;
                bool reachBorder = false;

                //Duplicates the given edges
                triangleAlg->InciseAlongEdgeList(new_edges,
                        new_points, end_points, reachBorder);

                if (!end_points.empty())
                {
                    a_last = end_points.back();
                }
                ind_ta = ind_tb;
                firstCut = false;

                // notify the end for the current sequence of topological change events
                //triangleMod->notifyEndingEvent();
                //triangleMod->propagateTopologicalChanges();
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

/*
 * Read and scan the file to save the triangles to incise later
 */
void TopologicalChangeProcessor::saveIndices()
{
    triangleIncisionInformation.clear();
    linesAboutIncision.clear();

    std::vector<std::string> listInTheFile;
    listInTheFile.clear();

    while (!infile->eof())
    {
        std::string line;
        getline(*infile, line);
        if (!line.empty())
            listInTheFile.push_back(line);
    }

    if (infile->eof())
    {
        infile->clear();
        infile->seekg(0);
    }

    //filter only the lines about incision and put them in the linesAboutIncision
    for (unsigned int i = 0 ; i < listInTheFile.size() ; i++)
    {
        std::string currentString(listInTheFile[i]);

        if (currentString.empty())
            break;

        size_t found=currentString.find("INCISE");
        if (found!=std::string::npos)
        {
            size_t foundT = listInTheFile[i-1].find("T=");
            if (foundT!=std::string::npos)
            {
                linesAboutIncision.push_back(listInTheFile[i-1]);
                linesAboutIncision.push_back(listInTheFile[i]);
                linesAboutIncision.push_back(listInTheFile[i+1]);
            }
            else
                dmsg_error() << " Error in line " << i << " : " << listInTheFile[i-1] ;
        }
    }

    if (linesAboutIncision.size() % 3)
    {
        dmsg_error() << " Problem (Bug) while saving the lines about incision." ;
    }

    for (std::vector<std::string>::iterator it=linesAboutIncision.begin(); it!=linesAboutIncision.end();)
    {
        TriangleIncisionInformation incisionInfo;

        std::string buff;
        std::istringstream str(*it);

        Real timeToIncise;
        int indexOfTime = -1;

        str >> buff;
        if (buff == "T=")
        {
            str >> timeToIncise;

            indexOfTime = findIndexInListOfTime(timeToIncise);
            if (indexOfTime == -1)
            {
                incisionInfo.timeToIncise = timeToIncise;
            }
            else
            {
                incisionInfo = triangleIncisionInformation[indexOfTime];
            }
        }

        //go to the next line
        ++it;
        std::istringstream str2(*it);

        unsigned int nbElements = 0;
        str2 >> buff;
        if (buff == "INCISE=")
        {
            str2 >> nbElements;
        }

        //go to the next line
        ++it;

        std::vector<Real> values = getValuesInLine(*it, nbElements);

        dmsg_error_when(values.empty())
                <<  "Error while saving the indices. Cannot get the values of line " << *it ;

        bool onlyCoordinates = false;

        if (values.size() == nbElements * 3)
        {
            onlyCoordinates = true;

            if(DEBUG_MSG)
                dmsg_info() << "Use only coordinates. Triangles indices will be computed. " ;
        }

        unsigned int increment = ( onlyCoordinates ) ? 3 : 4; // 3 if only the coordinates, 4 if there is also a triangle index

        for (unsigned int i = 0 ; i < values.size() ; i+=increment)
        {
            Vector3 coord;
            int triangleIndex;
            if (onlyCoordinates)
            {
                coord = Vector3(values[i], values[i+1], values[i+2]);
                findElementIndex(coord, triangleIndex, -1);
            }
            else
            {
                coord = Vector3(values[i+1], values[i+2], values[i+3]);
                triangleIndex = (unsigned int)values[i];
            }

            incisionInfo.triangleIndices.push_back(triangleIndex);

            const Vector3 constCoord = coord;
            const unsigned int triInd = triangleIndex;
            sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
            m_topology->getContext()->get(triangleGeo);

            sofa::helper::vector< double > baryCoef = triangleGeo->computeTriangleBarycoefs( triInd, constCoord);

            Vector3 barycentricCoordinates(baryCoef[0], baryCoef[1], baryCoef[2]);
            Vec3Types::Coord aCoord[3];
            triangleGeo->getTriangleVertexCoordinates(triInd, aCoord);

            incisionInfo.barycentricCoordinates.push_back(barycentricCoordinates);
        }

        if (indexOfTime > -1)
        {
            triangleIncisionInformation[indexOfTime] = incisionInfo;
        }
        else if (indexOfTime == -1)
        {
            triangleIncisionInformation.push_back(incisionInfo);
        }

        //go to the next line
        ++it;
    }


    /*HACK:
     * The topological changes crash the simulation when a incision line begins exactly where the previous ends
     * The hack consists in gently modify the first point of the line if both the points are equal (within epsilon)
     * Note : the crash is due to the computeIntersectedObjectsList algorithm in TriangleSetGeometryAlgorithm
     * */
    for ( unsigned int i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        triangleIncisionInformation[i].computeCoordinates(m_topology);
        if ( i )
        {
            Real epsilon = 1e-5;

            bool equal = true;

            for (unsigned int j = 0 ; j < 3 ; j++)
            {
                equal &= ( fabs( triangleIncisionInformation[i].coordinates.front()[j] - triangleIncisionInformation[i-1].coordinates.back()[j]) < epsilon );
            }

            if (equal &&  triangleIncisionInformation[i].coordinates.size() > 1)
            {
                if(DEBUG_MSG)
                    dmsg_warning() << "Two consecutives values are equal" ;

                Vector3 direction =  triangleIncisionInformation[i].coordinates[1] - triangleIncisionInformation[i].coordinates[0];
                direction *= epsilon;

                const Vector3 newPosition = triangleIncisionInformation[i].coordinates[0] + direction;

                sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
                m_topology->getContext()->get(triangleGeo);

                int triIndex;
                findElementIndex(Vector3(newPosition), triIndex, -1);

                msg_error_when( (triIndex==-1) )
                        << "Error while searching triangle index." ;

                triangleIncisionInformation[i].triangleIndices[0] = (unsigned int) triIndex;

                sofa::helper::vector< double > newBaryCoef = triangleGeo->computeTriangleBarycoefs( triangleIncisionInformation[i].triangleIndices[0], newPosition);

                for (unsigned int j = 0 ; j < 3 ; j++)
                    triangleIncisionInformation[i].barycentricCoordinates.front()[j] = newBaryCoef[j];

                triangleIncisionInformation[i].computeCoordinates(m_topology);
            }

        }
    }
}

int TopologicalChangeProcessor::findIndexInListOfTime(Real time)
{
    double epsilon = 1e-10;
    for (unsigned int i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        if ( fabs(time - triangleIncisionInformation[i].timeToIncise) < epsilon )
        {
            return (int)i;
        }
    }
    return -1;
}

std::vector<Real> TopologicalChangeProcessor::getValuesInLine(std::string line, unsigned int nbElements)
{
    std::vector<Real> values;
    values.clear();

    std::istringstream count(line);

    //bool onlyCoordinates = false;
    if ( !count.eof())
    {
        Real currentNumber;
        count >> currentNumber;
        values.push_back(currentNumber);
        while (!count.eof())
        {
            count >> currentNumber;
            values.push_back(currentNumber);
        }

        //the current algorithm needs either a triangle index and a coordinates for each element, or only coordinates
        //if only the coordinates are specified, the algorithm tries to find the corresponding index (less robust and precise and computationally more expensive)
        if( nbElements*4 != values.size())
        {
            if ( nbElements*3 == values.size())
            {
            }
            else
            {
                if (nbElements*4 < values.size())
                {
                    msg_warning() << "Incorrect input in '" << m_filename.getValue() <<"'. Too much values (" << values.size()<< ") in input in " << std::string(line) ;
                }
                else if (nbElements*3 > values.size())
                {
                    msg_error() << "Incorrect input in '" << m_filename.getValue() <<"'. Not enough values in input in " << std::string(line) << msgendl
                                << "Topological changes aborted" ;
                    values.clear();
                    return values;
                }
                else
                {
                    msg_warning() << "Incorrect input in '" << m_filename.getValue() << "' in line " << std::string(line) << msgendl
                                  << "If only coordinates are wanted, there are too much values. If coordinates with the index are wanted, there are not enough values."
                                  << "Will consider values as coordinates only." ;
                }
            }
        }
    }
    else
    {
        msg_error() << "No input values in '" << m_filename.getValue() << "'." ;
        values.clear();
        return values;
    }

    return values;
}

/**
 * Find the triangle index where the point with coordinates coord can be
 * NOTE : the need of oldTriangleIndex comes to avoid some cases when the old triangle is overlapping another. It keeps the same
 * index instead of taking the new one.
 */
void  TopologicalChangeProcessor::findElementIndex(Vector3 coord, int& triangleIndex, int oldTriangleIndex)
{
    if (!m_topology)
        return;

    //get the number of triangle in the topology
    unsigned int nbTriangle = m_topology->getNbTriangles();

    sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
    m_topology->getContext()->get(triangleAlg);
    if (!triangleAlg)
    {
        msg_error() <<"TopologicalChangeProcessor needs a TriangleSetTopologyAlgorithms component." ;
    }

    sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);
    if (!triangleGeo)
    {
        msg_error() << "TopologicalChangeProcessor needs a TriangleSetGeometryAlgorithms component." ;
    }

    std::vector<unsigned int> triIndices;
    triIndices.clear();

    /***********
     * fast scan
     * check if the point is in the AABB of each triangle
     ***********/
    for ( unsigned int i = 0 ; i < nbTriangle ; i++)
    {
        Vector3 minAABB, maxAABB;
        triangleGeo->computeTriangleAABB(i, minAABB, maxAABB);
        bool isPointInAABB = true;
        for (int j = 0 ; j < 3 ; j++)
        {
            if (coord[j] < minAABB[j] || coord[j] > maxAABB[j])
            {
                isPointInAABB = false;
                break;
            }
        }

        if (isPointInAABB)
        {
            triIndices.push_back(i);
        }
    }

    std::vector<unsigned int> finalTriIndices;
    finalTriIndices.clear();

    for (unsigned int i = 0 ; i < triIndices.size() ; i++)
    {
        const bool is_tested = false;
        unsigned int indTest = 0;
        const bool isPointInTriangle = triangleGeo->isPointInsideTriangle(triIndices[i], is_tested, coord, indTest);

        if (isPointInTriangle)
        {
            finalTriIndices.push_back(triIndices[i]);

            if ((int)triIndices[i] == oldTriangleIndex)
            {
                triangleIndex = oldTriangleIndex;
                return;
            }
        }
    }

    if ( finalTriIndices.size() == 1)
    {
        triangleIndex = finalTriIndices[0];
        return;
    }

    finalTriIndices.clear();

    /***
     * Projection of the point followed by a including test
     */
    Real x = coord[0], y = coord[1], z = coord[2];
    //project point along the normal
    for (unsigned int i = 0 ; i < nbTriangle ; i++)
    {
        //get the normal of the current triangle
        Vector3 normal = triangleGeo->computeTriangleNormal(i);
        Real normalNorm = normal.norm();
        if (!normalNorm)
            break;
        //normalize the normal (avoids to divide by the norm)
        normal /= normal.norm();
        Real a = normal[0], b = normal[1], c = normal[2];

        //get the coordinates points of the triangle
        Vector3 points[3];
        triangleGeo->getTriangleVertexCoordinates(i, points);

        //get d in the equation of the plane of the triangle ax+by+cz + d = 0
        Real d = - (points[0][0] * c + points[0][1] * b + points[0][2] * c );
        Vector3 projectedPoint;

        projectedPoint[0] = ((b * b + c * c) * x - a * b * y - a * c * z - d * a) /*/normalNorm*/;
        projectedPoint[1] = (- a * b * x + (a * a + c * c) * y - b * c * z - d * b) /*/normalNorm*/;
        projectedPoint[2] = (- a * c * x - b * c * y + (a * a + b * b) * z - d * c) /*/normalNorm*/;

        const bool is_tested = false;
        unsigned int indTest = 0;
        //test if the projected point is inside the current triangle
        const bool isPointInTriangle = triangleGeo->isPointInsideTriangle(i, is_tested, projectedPoint, indTest);

        if (isPointInTriangle)
        {
            if ( (int)i == oldTriangleIndex)
            {
                triangleIndex = i;
                return;
            }
            finalTriIndices.push_back(i);
        }
    }


    if (finalTriIndices.size() == 1)
    {
        triangleIndex = finalTriIndices[0];
        return;
    }
    else if (finalTriIndices.size() > 1)
    {
        // TODO: choose the best triangle between the ones found
        triangleIndex = finalTriIndices[0];
        return;
    }

    triangleIndex = -1;
    return;

    //TODO(dmarchal 2017-05-03) So what ? Can we remove this ?
    //beastly way
//                for( unsigned int i = 0 ; i < nbTriangle ; i++)
//                {
//                    bool isPointInTriangle = false;
//
//                    bool is_tested = false;
//                    unsigned int indTest;
//                    isPointInTriangle = triangleGeo->isPointInsideTriangle(i, is_tested, coord, indTest);
//
//                    if (isPointInTriangle)
//                    {
//                        triangleIndex = i;
//                        return;
//                    }
//                }

}

void TopologicalChangeProcessor::inciseWithSavedIndices()
{
    sofa::component::topology::TriangleSetTopologyModifier* triangleMod;
    m_topology->getContext()->get(triangleMod);

    sofa::component::topology::TriangleSetTopologyAlgorithms<Vec3Types>* triangleAlg;
    m_topology->getContext()->get(triangleAlg);

    sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    int indexOfTime = findIndexInListOfTime(getContext()->getTime());

    if (indexOfTime == -1)
    {
        std::stringstream tmp ;
        tmp <<" Unable to find a time index with time " <<  getContext()->getTime() << ". The possible values are ";
        for (unsigned int i = 0 ; i < triangleIncisionInformation.size() ; i++)
            tmp << triangleIncisionInformation[i].timeToIncise << " | ";
        tmp << ". Aborting." ;
        msg_error() << tmp.str() ;
        return;
    }

    Vec3Types::Coord aCoord[3];
    Vec3Types::Coord bCoord[3];
    Vector3 a;
    Vector3 b;

    unsigned int a_last = core::topology::BaseMeshTopology::InvalidID;
    unsigned int b_last = core::topology::BaseMeshTopology::InvalidID;
    bool firstCut= true;

    std::vector<Vector3> coordinates;
    for (unsigned int i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        if ( (int) i == indexOfTime)
            coordinates = triangleIncisionInformation[indexOfTime].computeCoordinates(m_topology);
        else
            triangleIncisionInformation[i].computeCoordinates(m_topology);
    }

    unsigned int ind_ta = 0;
    if (indexOfTime < (int)triangleIncisionInformation.size())
    {
        if (triangleIncisionInformation[indexOfTime].triangleIndices.empty())
        {
            msg_error() << "List of triangles indices cannot be empty. Aborting. " ;
            return;
        }
        ind_ta = triangleIncisionInformation[indexOfTime].triangleIndices[0];
    }
    else
    {
        msg_error() <<  "found index '" << indexOfTime << "' which is larger than the vector size '" << triangleIncisionInformation.size() <<"'" ;
        return;
    }

    triangleGeo->getTriangleVertexCoordinates(ind_ta, aCoord);

    a.clear();
    a = coordinates[0];

    unsigned int ind_tb = 0;
    for (unsigned int i =1; i < triangleIncisionInformation[indexOfTime].triangleIndices.size(); ++i)
    {
        ind_tb = triangleIncisionInformation[indexOfTime].triangleIndices[i];

        triangleGeo->getTriangleVertexCoordinates(ind_tb, bCoord);
        b.clear();
        b = coordinates[i];

        // Output declarations
        sofa::helper::vector< sofa::core::topology::TopologyObjectType> topoPath_list;
        sofa::helper::vector<unsigned int> indices_list;
        sofa::helper::vector< Vec<3, double> > coords2_list;

        if(firstCut)
            a_last = core::topology::BaseMeshTopology::InvalidID;
        else
        {
            core::behavior::MechanicalState<Vec3Types>* mstate = m_topology->getContext()->get<core::behavior::MechanicalState<Vec3Types> >();
            //get the coordinates of the mechanical state
            const helper::vector<Vector3> &v_coords =  mstate->read(core::ConstVecCoordId::position())->getValue();
            a = v_coords[a_last];
        }

        errorTrianglesIndices.push_back(ind_ta);
        errorTrianglesIndices.push_back(ind_tb);

        //Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
        bool isPathOk = triangleGeo->computeIntersectedObjectsList(a_last, a, b, ind_ta, ind_tb, topoPath_list, indices_list, coords2_list);

        if (!isPathOk)
        {
            dmsg_error() << "While computing computeIntersectedPointsList between triangles '"
                    << errorTrianglesIndices[errorTrianglesIndices.size() - 1] << "' and '" << errorTrianglesIndices[errorTrianglesIndices.size() - 2]  << "' at time = '" << getContext()->getTime()  << "'" ;

            if(DEBUG_MSG)
            {
                dmsg_error() << " a = " << a << " b = " << b << msgendl
                             << "ind_ta = " << ind_ta << " ind_tb = " << ind_tb ;
            }

            break;
        }
        else
        {
            errorTrianglesIndices.pop_back();
            errorTrianglesIndices.pop_back();
        }

        sofa::helper::vector< unsigned int > new_edges;

        //Split triangles to create edges along a path given as a the list of existing edges and triangles crossed by it.
        triangleAlg->SplitAlongPath(a_last, a, b_last, b, topoPath_list, indices_list, coords2_list, new_edges, m_epsilonSnapPath.getValue(), m_epsilonSnapBorder.getValue());

        sofa::helper::vector<unsigned int> new_points;
        sofa::helper::vector<unsigned int> end_points;
        bool reachBorder = false;

        //Duplicates the given edges
        triangleAlg->InciseAlongEdgeList(new_edges, new_points, end_points, reachBorder);

        msg_info_when(reachBorder) << "Incision has reached a border.";

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

        //update the triangle incision information
        updateTriangleIncisionInformation();
    }
}

/**
 * If a topological change happened, the triangleIncisionInformation are wrong, so the need to update them
 * Note : only after a computeCoordinates
 */
void TopologicalChangeProcessor::updateTriangleIncisionInformation()
{
    unsigned int nbTriangleInfo = triangleIncisionInformation.size();
    sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    for ( unsigned int i = 0 ; i < nbTriangleInfo ; i++)
    {
        for (unsigned int j = 0 ; j < triangleIncisionInformation[i].triangleIndices.size() ; j++ )
        {
            //update the triangle index corresponding to the current coordinates
            int newTriangleIndexb;
            unsigned int currentTriangleIndex = triangleIncisionInformation[i].triangleIndices[j];

            if ( j >= triangleIncisionInformation[i].coordinates.size() || triangleIncisionInformation[i].coordinates.empty())
            {
                msg_warning() << "(updateTriangleIncisionInformation): error accessing coordinates" ;
                break;
            }

            findElementIndex(triangleIncisionInformation[i].coordinates[j], newTriangleIndexb, currentTriangleIndex);

            if ( newTriangleIndexb == -1)
            {
                msg_warning() << "(updateTriangleIncisionInformation): error while finding the point " << triangleIncisionInformation[i].coordinates[j] << " in a new triangle. Current triangle index = " << currentTriangleIndex ;
                break;
            }

            msg_info_when((int)currentTriangleIndex != newTriangleIndexb)
                          << "(updateTriangleIncisionInformation): incision point which was in triangle " << currentTriangleIndex
                          << " has been updated to " << newTriangleIndexb  ;

            triangleIncisionInformation[i].triangleIndices[j] = newTriangleIndexb;

            //update the triangle barycentric coordinates corresponding to the current coordinates
            const Vector3 constCoord = triangleIncisionInformation[i].coordinates[j];
            sofa::helper::vector< double > baryCoef = triangleGeo->computeTriangleBarycoefs( newTriangleIndexb, constCoord);
            triangleIncisionInformation[i].barycentricCoordinates[j] = Vector3(baryCoef[0], baryCoef[1], baryCoef[2]);
        }
    }
}


void TopologicalChangeProcessor::draw(const core::visual::VisualParams* vparams)
{
    if (!m_topology)
        return;

    if(!m_draw.getValue())
        return;

    if (!vparams->displayFlags().getShowBehaviorModels())
        return;

    sofa::component::topology::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    if (!triangleGeo)
        return;

    unsigned int nbTriangles = m_topology->getNbTriangles();

    std::vector< Vector3 > trianglesToDraw;
    std::vector< Vector3 > pointsToDraw;

    for (unsigned int i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        for (unsigned int j = 0 ; j < triangleIncisionInformation[i].triangleIndices.size() ; j++)
        {
            unsigned int triIndex = triangleIncisionInformation[i].triangleIndices[j];

            if ( triIndex > nbTriangles -1)
                break;

            Vec3Types::Coord coord[3];
            triangleGeo->getTriangleVertexCoordinates(triIndex, coord);

            for(unsigned int k = 0 ; k < 3 ; k++)
                trianglesToDraw.push_back(coord[k]);

            Vector3 a;
            a.clear();
            for (unsigned k = 0 ; k < 3 ; k++)
                a += coord[k] * triangleIncisionInformation[i].barycentricCoordinates[j][k];

            pointsToDraw.push_back(a);
        }
    }

    vparams->drawTool()->drawTriangles(trianglesToDraw, Vec<4,float>(0.0,0.0,1.0,1.0));
    vparams->drawTool()->drawPoints(pointsToDraw, 15.0,  Vec<4,float>(1.0,0.0,1.0,1.0));

    if (!errorTrianglesIndices.empty())
    {
        trianglesToDraw.clear();
        /* initialize random seed: */
        srand ( (unsigned int)time(NULL) );

        for (unsigned int i = 0 ; i < errorTrianglesIndices.size() ; i++)
        {
            Vec3Types::Coord coord[3];
            triangleGeo->getTriangleVertexCoordinates(errorTrianglesIndices[i], coord);

            for(unsigned int k = 0 ; k < 3 ; k++)
                trianglesToDraw.push_back(coord[k]);
        }

        vparams->drawTool()->drawTriangles(trianglesToDraw,
                Vec<4,float>(1.0f,(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, 1.0f));
    }
}

} // namespace misc

} // namespace component

} // namespace sofa
