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
#include <sofa/component/topology/utility/TopologicalChangeProcessor.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyModifier.h>

#include <ctime>

#ifndef NDEBUG
    #define DEBUG_MSG true
#else
    #define DEBUG_MSG false
#endif

namespace sofa::component::topology::utility
{

using namespace sofa::type;
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
    , m_pointsToRemove(initData (&m_pointsToRemove, "pointsToRemove", "List of point IDs to be removed."))
    , m_edgesToRemove (initData (&m_edgesToRemove, "edgesToRemove", "List of edge IDs to be removed."))
    , m_trianglesToRemove (initData (&m_trianglesToRemove, "trianglesToRemove", "List of triangle IDs to be removed."))
    , m_quadsToRemove (initData (&m_quadsToRemove, "quadsToRemove", "List of quad IDs to be removed."))
    , m_tetrahedraToRemove (initData (&m_tetrahedraToRemove, "tetrahedraToRemove", "List of tetrahedron IDs to be removed."))
    , m_hexahedraToRemove (initData (&m_hexahedraToRemove, "hexahedraToRemove", "List of hexahedron IDs to be removed."))
    , m_saveIndicesAtInit( initData(&m_saveIndicesAtInit, false, "saveIndicesAtInit", "set to 'true' to save the incision to do in the init to incise even after a movement"))
    , m_epsilonSnapPath( initData(&m_epsilonSnapPath, (SReal)0.1, "epsilonSnapPath", "epsilon snap path"))
    , m_epsilonSnapBorder( initData(&m_epsilonSnapBorder, (SReal)0.25, "epsilonSnapBorder", "epsilon snap path"))
    , m_draw( initData(&m_draw, false, "draw", "draw information"))
    , l_topology(initLink("topology", "link to the topology container"))
    , m_topology(nullptr)
    , infile(nullptr)
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
    , gzfile(nullptr)
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
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}


void TopologicalChangeProcessor::init()
{
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

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
        infile = nullptr;
    }
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
    if (gzfile)
    {
        gzclose(gzfile);
        gzfile = nullptr;
    }
#endif

    const std::string& filename = m_filename.getFullPath();
    if (filename.empty())
    {
        msg_error() << "empty filename";
    }
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            msg_error() << "TopologicalChangeProcessor: Error opening compressed file " << filename;
        }
    }
#endif
    else
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            msg_error() << "TopologicalChangeProcessor: Error opening file " << filename;
            delete infile;
            infile = nullptr;
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
        const helper::ReadAccessor< Data<type::vector<Index> > > points = m_pointsToRemove;
        const helper::ReadAccessor< Data<type::vector<Index> > > edges = m_edgesToRemove;
        const helper::ReadAccessor< Data<type::vector<Index> > > triangles = m_trianglesToRemove;
        const helper::ReadAccessor< Data<type::vector<Index> > > quads = m_quadsToRemove;
        const helper::ReadAccessor< Data<type::vector<Index> > > tetrahedra = m_tetrahedraToRemove;
        const helper::ReadAccessor< Data<type::vector<Index> > > hexahedra = m_hexahedraToRemove;

        if (!hexahedra.empty())
        {
            sofa::component::topology::container::dynamic::HexahedronSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            type::vector<Index> vitems;
            vitems.assign(hexahedra.begin(), hexahedra.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No HexahedraTopology available";
        }

        if (!tetrahedra.empty())
        {
            sofa::component::topology::container::dynamic::TetrahedronSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            type::vector<Index> vitems;
            vitems.assign(tetrahedra.begin(), tetrahedra.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No TetrahedraTopology available";
        }

        if (!quads.empty())
        {
            sofa::component::topology::container::dynamic::QuadSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            type::vector<Index> vitems;
            vitems.assign(quads.begin(), quads.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No QuadTopology available";
        }

        if (!triangles.empty())
        {
            sofa::component::topology::container::dynamic::TriangleSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            sofa::type::vector<Index> vitems;
            vitems.assign(triangles.begin(), triangles.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No TriangleTopology available";
        }

        if (!edges.empty())
        {
            sofa::component::topology::container::dynamic::EdgeSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            type::vector<Index> vitems;
            vitems.assign(edges.begin(), edges.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No EdgeTopology available";
        }

        if (!points.empty())
        {
            sofa::component::topology::container::dynamic::PointSetTopologyModifier* topoMod;
            m_topology->getContext()->get(topoMod);
            type::vector<Index> vitems;
            vitems.assign(points.begin(), points.end());

            if (topoMod)
                topoMod->removeItems(vitems);
            else
                msg_error() << "No PointTopology Modifier available";
        }

        // iterate, time set to infini if no interval.
        double& newTime = *m_timeToRemove.beginEdit();
        if (m_interval.getValue() != 0.0)
            newTime += m_interval.getValue();
        else
            newTime = (Index)-1;
        m_timeToRemove.endEdit();
    }
}


bool TopologicalChangeProcessor::readNext(double time, std::vector<std::string>& validLines)
{
    if (!m_topology) return false;
    if (!infile
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
        && !gzfile
#endif
       )
        return false;
    lastTime = time;
    validLines.clear();
    std::string line, cmd;
    const SReal epsilon = std::numeric_limits<SReal>::epsilon();
    while (nextTime < time || fabs(nextTime - time) < epsilon )
    {
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
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

    size_t nbElements = 0;
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
                sofa::component::topology::container::dynamic::TriangleSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if(!topoMod)
                {
                    msg_error() << "No PointSetTopologyModifier available";
                    continue;
                }

                type::vector< Vec2 > baryCoords;
                baryCoords.resize(nbElements);
                type::vector< Index > triangles;
                triangles.resize(nbElements);

                for(size_t i=0;i<nbElements;++i)
                {
                    Sin >> triangles[i];
                    Vec2& baryCoord = baryCoords[i];
                    Sin >> baryCoord[0] >> baryCoord[1];
                }


                type::vector< type::vector< Index > > p_ancestors(nbElements);
                sofa::type::vector< type::vector< SReal > > p_baryCoefs(nbElements);
                for(size_t i=0; i<nbElements; ++i)
                {
                    auto& ancestor = p_ancestors[i];
                    ancestor.resize(3);
                    const core::topology::BaseMeshTopology::Triangle& t = m_topology->getTriangle( triangles[i] );
                    ancestor[0] = t[0];
                    ancestor[1] = t[1];
                    ancestor[2] = t[2];
                    type::vector<SReal>& baryCoef = p_baryCoefs[i];
                    baryCoef.resize(3);
                    baryCoef[0] = baryCoords[i][0];
                    baryCoef[1] = baryCoords[i][1];
                    baryCoef[2] = 1 - baryCoef[0] - baryCoef[1];
                }
                topoMod->addPoints(nbElements, p_ancestors, p_baryCoefs, true);

            }
            else if ( EleType == "Edge" || EleType == "Edges")
            {
                sofa::component::topology::container::dynamic::EdgeSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    msg_error() << "No QuadTopology available";
                    continue;
                }

                type::vector<core::topology::Topology::Edge > vitems;
                vitems.resize (nbElements);

                for (size_t i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1];

                topoMod->addEdges(vitems);

            }
            else if ( EleType == "Triangle" || EleType == "Triangles")
            {

                type::vector<type::vector<Index> >  p_ancestors(nbElements);
                type::vector<type::vector<SReal> >        p_baryCoefs(nbElements);

                if(!str.eof() )
                {
                    std::string token;
                    str >> token;
                    if(token == "Ancestors" || token == "Ancestor")
                    {
                        for(size_t i = 0; i<nbElements; ++i)
                        {
                            type::vector<Index>& ancestor = p_ancestors[i];
                            ancestor.resize(1);
                            str >> ancestor[0];
                        }
                    }
                }

                sofa::component::topology::container::dynamic::TriangleSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    msg_error() << "No TriangleTopology available";
                    continue;
                }

                type::vector<core::topology::Topology::Triangle > vitems;
                vitems.resize (nbElements);

                for (size_t i = 0; i<nbElements; ++i)
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
                sofa::component::topology::container::dynamic::QuadSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    msg_error() << "No QuadTopology available";
                    continue;
                }

                type::vector<core::topology::Topology::Quad > vitems;
                vitems.resize (nbElements);

                for (size_t i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3];

                topoMod->addQuads(vitems);
            }
            else if ( EleType == "Tetrahedron" || EleType == "Tetrahedra")
            {
                sofa::component::topology::container::dynamic::TetrahedronSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    msg_error() << "No TetrahedraTopology available";
                    continue;
                }

                type::vector<core::topology::Topology::Tetrahedron > vitems;
                vitems.resize (nbElements);

                for (size_t i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3];

                topoMod->addTetrahedra(vitems);
            }
            else if ( EleType == "Hexahedron" || EleType == "Hexahedra")
            {
                sofa::component::topology::container::dynamic::HexahedronSetTopologyModifier* topoMod;
                m_topology->getContext()->get(topoMod);

                if (!topoMod)
                {
                    msg_error() << "No HexahedraTopology available";
                    continue;
                }

                type::vector<core::topology::Topology::Hexahedron > vitems;
                vitems.resize (nbElements);

                for (size_t i = 0; i<nbElements; ++i)
                    Sin >> vitems[i][0] >> vitems[i][1] >> vitems[i][2] >> vitems[i][3]
                        >> vitems[i][4] >> vitems[i][5] >> vitems[i][6] >> vitems[i][7];

                topoMod->addHexahedra(vitems);
            }
            else
            {
                msg_error() << "keyword: '" << EleType << "' not expected.";
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

            type::vector<Index> vitems;
            vitems.resize (nbElements);

            for (size_t i = 0; i<nbElements; ++i)
                Sin >> vitems[i];

            topoMod->removeItems(vitems);

            //TODO: check  Cas des mappings volume to surface. Il ny a pas suppression des element surfaceique isole.

            ++it;
            continue;
        }
        else if ( buff == "INCISE=" )
        {
            msg_info() << "processTopologicalChanges: about to make a incision with time = " << time;

            if (m_saveIndicesAtInit.getValue())
            {
                inciseWithSavedIndices();
                ++it; ++it; continue;
            }

            sofa::component::topology::container::dynamic::TriangleSetTopologyModifier* triangleMod;
            m_topology->getContext()->get(triangleMod);

            sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
            m_topology->getContext()->get(triangleGeo);

            sofa::type::Vec3 a;
            sofa::type::Vec3 b;
            Index ind_ta;
            Index ind_tb;
            Index a_last = sofa::InvalidID;
            Index b_last = sofa::InvalidID;
            bool firstCut = true;

            //get the number of element
            str >> nbElements;

            ++it;//go to the next line

            //get the values in the current line and put them in a vector
            std::vector<SReal> values = getValuesInLine(*it, nbElements);
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

            for (Index j = 0; j < 3; ++j)
                Sin >> a[j];

            if (onlyCoordinates)
            {
                //find the triangle corresponding to the current coordinates
                findElementIndex(a, ind_ta, -1);
            }

            for (size_t i = 1; i < nbElements; ++i)
            {
                if (!onlyCoordinates)
                {
                    Sin >> ind_tb;//get the second index of triangle
                }
                for (Index j = 0; j < 3; ++j)
                    Sin >> b[j];

                if (onlyCoordinates)
                {
                    findElementIndex(b, ind_tb, -1);
                }

                // Output declarations
                sofa::type::vector<sofa::geometry::ElementType>       topoPath_list;
                sofa::type::vector<Index> indices_list;
                sofa::type::vector<Vec3 > coords2_list;

                if (firstCut)
                    a_last
                        = sofa::InvalidID;
                else
                {
                    core::behavior::MechanicalState<Vec3Types> * mstate =
                        m_topology->getContext()->get< core::behavior::MechanicalState<Vec3Types> > ();
                    //get the coordinates of the mechanical state
                    const auto &v_coords = mstate->read(core::ConstVecCoordId::position())->getValue();
                    a = v_coords[a_last];
                }

                //Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
                Index uInd_ta = (Index) ind_ta;
                Index uInd_tb = (Index) ind_tb;

                bool isPathOk =
                    triangleGeo->computeIntersectedObjectsList(
                            a_last, a, b, uInd_ta, uInd_tb,
                            topoPath_list, indices_list,
                            coords2_list);

                if (!isPathOk)
                {
                    msg_error() << "Invalid path in computeIntersectedPointsList";
                    break;
                }

                sofa::type::vector<Index> new_edges;

                //Split triangles to create edges along a path given as a the list of existing edges and triangles crossed by it.
                triangleGeo->SplitAlongPath(a_last, a, b_last, b,
                        topoPath_list, indices_list, coords2_list,
                        new_edges, 0.1, 0.25);

                sofa::type::vector<Index> new_points;
                sofa::type::vector<Index> end_points;
                bool reachBorder = false;

                //Duplicates the given edges
                triangleGeo->InciseAlongEdgeList(new_edges,
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
    for (size_t i = 0 ; i < listInTheFile.size() ; i++)
    {
        std::string currentString(listInTheFile[i]);

        if (currentString.empty())
            break;

        size_t found=currentString.find("INCISE");
        if (found!=std::string::npos)
        {
            size_t foundT = listInTheFile[i-1].find("T=");
            if (foundT != std::string::npos)
            {
                linesAboutIncision.push_back(listInTheFile[i - 1]);
                linesAboutIncision.push_back(listInTheFile[i]);
                linesAboutIncision.push_back(listInTheFile[i + 1]);
            }
            else
                msg_error() << " Error in line " << i << " : " << listInTheFile[i - 1];
        }
    }

    if (linesAboutIncision.size() % 3)
    {
        msg_error() << " Problem (Bug) while saving the lines about incision.";
    }

    for (std::vector<std::string>::iterator it=linesAboutIncision.begin(); it!=linesAboutIncision.end();)
    {
        TriangleIncisionInformation incisionInfo;

        std::string buff;
        std::istringstream str(*it);

        SReal timeToIncise;
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

        Index nbElements = 0;
        str2 >> buff;
        if (buff == "INCISE=")
        {
            str2 >> nbElements;
        }

        //go to the next line
        ++it;

        std::vector<SReal> values = getValuesInLine(*it, nbElements);

        msg_error_when(values.empty()) << "Error while saving the indices. Cannot get the values of line " << *it;

        bool onlyCoordinates = false;

        if (values.size() == nbElements * 3)
        {
            onlyCoordinates = true;
            msg_info() << "Use only coordinates. Triangles indices will be computed. ";
        }

        unsigned int increment = ( onlyCoordinates ) ? 3 : 4; // 3 if only the coordinates, 4 if there is also a triangle index

        for (size_t i = 0 ; i < values.size() ; i+=increment)
        {
            Vec3 coord;
            Index triangleIndex;
            if (onlyCoordinates)
            {
                coord = Vec3(values[i], values[i+1], values[i+2]);
                findElementIndex(coord, triangleIndex, -1);
            }
            else
            {
                coord = Vec3(values[i+1], values[i+2], values[i+3]);
                triangleIndex = (Index)values[i];
            }

            incisionInfo.triangleIndices.push_back(triangleIndex);

            const Vec3 constCoord = coord;
            const unsigned int triInd = triangleIndex;
            sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
            m_topology->getContext()->get(triangleGeo);

            const auto baryCoef = triangleGeo->computeTriangleBarycoefs( triInd, constCoord);

            Vec3 barycentricCoordinates(baryCoef[0], baryCoef[1], baryCoef[2]);
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
    for ( size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        triangleIncisionInformation[i].computeCoordinates(m_topology);
        if ( i )
        {
            SReal epsilon = 1e-5;

            bool equal = true;

            for (unsigned int j = 0 ; j < 3 ; j++)
            {
                equal &= ( fabs( triangleIncisionInformation[i].coordinates.front()[j] - triangleIncisionInformation[i-1].coordinates.back()[j]) < epsilon );
            }

            if (equal &&  triangleIncisionInformation[i].coordinates.size() > 1)
            {
                msg_warning() << "Two consecutives values are equal" ;

                Vec3 direction =  triangleIncisionInformation[i].coordinates[1] - triangleIncisionInformation[i].coordinates[0];
                direction *= epsilon;

                const Vec3 newPosition = triangleIncisionInformation[i].coordinates[0] + direction;

                sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
                m_topology->getContext()->get(triangleGeo);

                Index triIndex;
                findElementIndex(Vec3(newPosition), triIndex, -1);

                msg_error_when( (triIndex==sofa::InvalidID) ) << "Error while searching triangle index." ;

                triangleIncisionInformation[i].triangleIndices[0] = triIndex;

                const auto newBaryCoef = triangleGeo->computeTriangleBarycoefs( triangleIncisionInformation[i].triangleIndices[0], newPosition);

                for (unsigned int j = 0 ; j < 3 ; j++)
                    triangleIncisionInformation[i].barycentricCoordinates.front()[j] = newBaryCoef[j];

                triangleIncisionInformation[i].computeCoordinates(m_topology);
            }

        }
    }
}

TopologicalChangeProcessor::Index TopologicalChangeProcessor::findIndexInListOfTime(SReal time)
{
    const double epsilon = 1e-10;
    for (size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        if ( fabs(time - triangleIncisionInformation[i].timeToIncise) < epsilon )
        {
            return i;
        }
    }
    return sofa::InvalidID;
}

std::vector<SReal> TopologicalChangeProcessor::getValuesInLine(std::string line, size_t nbElements)
{
    std::vector<SReal> values;
    values.clear();

    std::istringstream count(line);

    //bool onlyCoordinates = false;
    if ( !count.eof())
    {
        SReal currentNumber;
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
                    msg_warning() << "Incorrect input in '" << m_filename.getValue() << "'. Too much values (" << values.size() << ") in input in " << std::string(line);
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
void  TopologicalChangeProcessor::findElementIndex(Vec3 coord, Index& triangleIndex, Index oldTriangleIndex)
{
    if (!m_topology)
        return;

    //get the number of triangle in the topology
    const size_t nbTriangle = m_topology->getNbTriangles();

    sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
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
        sofa::type::Vec3 minAABB, maxAABB;
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

    std::vector<Index> finalTriIndices;
    finalTriIndices.clear();

    for (size_t i = 0 ; i < triIndices.size() ; i++)
    {
        const bool is_tested = false;
        Index indTest = 0;
        const bool isPointInTriangle = triangleGeo->isPointInsideTriangle(triIndices[i], is_tested, coord, indTest);

        if (isPointInTriangle)
        {
            finalTriIndices.push_back(triIndices[i]);

            if (triIndices[i] == oldTriangleIndex)
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
    const SReal x = coord[0], y = coord[1], z = coord[2];
    //project point along the normal
    for (unsigned int i = 0 ; i < nbTriangle ; i++)
    {
        //get the normal of the current triangle
        auto normal = triangleGeo->computeTriangleNormal(i);
        const SReal normalNorm = normal.norm();
        if (!normalNorm)
            break;
        //normalize the normal (avoids to divide by the norm)
        normal /= normal.norm();
        const SReal a = normal[0], b = normal[1], c = normal[2];

        //get the coordinates points of the triangle
        sofa::type::Vec3 points[3];
        triangleGeo->getTriangleVertexCoordinates(i, points);

        //get d in the equation of the plane of the triangle ax+by+cz + d = 0
        const SReal d = - (points[0][0] * a + points[0][1] * b + points[0][2] * c );
        sofa::type::Vec3 projectedPoint;

        projectedPoint[0] = ((b * b + c * c) * x - a * b * y - a * c * z - d * a) /*/normalNorm*/;
        projectedPoint[1] = (- a * b * x + (a * a + c * c) * y - b * c * z - d * b) /*/normalNorm*/;
        projectedPoint[2] = (- a * c * x - b * c * y + (a * a + b * b) * z - d * c) /*/normalNorm*/;

        const bool is_tested = false;
        Index indTest = 0;
        //test if the projected point is inside the current triangle
        const bool isPointInTriangle = triangleGeo->isPointInsideTriangle(i, is_tested, projectedPoint, indTest);

        if (isPointInTriangle)
        {
            if ( i == oldTriangleIndex)
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
//                for( size_t i = 0 ; i < nbTriangle ; i++)
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
    sofa::component::topology::container::dynamic::TriangleSetTopologyModifier* triangleMod;
    m_topology->getContext()->get(triangleMod);

    sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    const int indexOfTime = findIndexInListOfTime(getContext()->getTime());

    if (indexOfTime == -1)
    {
        std::stringstream tmp ;
        tmp <<" Unable to find a time index with time " <<  getContext()->getTime() << ". The possible values are ";
        for (size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
            tmp << triangleIncisionInformation[i].timeToIncise << " | ";
        tmp << ". Aborting." ;
        msg_error() << tmp.str() ;
        return;
    }

    Vec3Types::Coord aCoord[3];
    Vec3Types::Coord bCoord[3];
    sofa::type::Vec3 a;
    sofa::type::Vec3 b;

    sofa::Index a_last = sofa::InvalidID;
    const sofa::Index b_last = sofa::InvalidID;
    bool firstCut= true;

    std::vector<Vec3> coordinates;
    for (size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        if ( (int) i == indexOfTime)
            coordinates = triangleIncisionInformation[indexOfTime].computeCoordinates(m_topology);
        else
            triangleIncisionInformation[i].computeCoordinates(m_topology);
    }

    Index ind_ta = 0;
    if (indexOfTime < (int)triangleIncisionInformation.size())
    {
        if (triangleIncisionInformation[indexOfTime].triangleIndices.empty())
        {
            msg_error() << "List of triangles indices cannot be empty. Aborting. ";
            return;
        }
        ind_ta = triangleIncisionInformation[indexOfTime].triangleIndices[0];
    }
    else
    {
        msg_error() << "found index '" << indexOfTime << "' which is larger than the vector size '" << triangleIncisionInformation.size() << "'";
        return;
    }

    triangleGeo->getTriangleVertexCoordinates(ind_ta, aCoord);

    a.clear();
    a = coordinates[0];

    Index ind_tb = 0;
    for (size_t i =1; i < triangleIncisionInformation[indexOfTime].triangleIndices.size(); ++i)
    {
        ind_tb = triangleIncisionInformation[indexOfTime].triangleIndices[i];

        triangleGeo->getTriangleVertexCoordinates(ind_tb, bCoord);
        b.clear();
        b = coordinates[i];

        // Output declarations
        sofa::type::vector< sofa::geometry::ElementType> topoPath_list;
        sofa::type::vector<Index> indices_list;
        sofa::type::vector< Vec3 > coords2_list;

        if(firstCut)
            a_last = sofa::InvalidID;
        else
        {
            const core::behavior::MechanicalState<Vec3Types>* mstate = m_topology->getContext()->get<core::behavior::MechanicalState<Vec3Types> >();
            //get the coordinates of the mechanical state
            const auto &v_coords =  mstate->read(core::ConstVecCoordId::position())->getValue();
            a = v_coords[a_last];
        }

        errorTrianglesIndices.push_back(ind_ta);
        errorTrianglesIndices.push_back(ind_tb);

        //Computes the list of objects (points, edges, triangles) intersected by the segment from point a to point b and the triangular mesh.
        const bool isPathOk = triangleGeo->computeIntersectedObjectsList(a_last, a, b, ind_ta, ind_tb, topoPath_list, indices_list, coords2_list);

        if (!isPathOk)
        {
            msg_error() << "While computing computeIntersectedPointsList between triangles '"
                    << errorTrianglesIndices[errorTrianglesIndices.size() - 1] << "' and '" << errorTrianglesIndices[errorTrianglesIndices.size() - 2]  << "' at time = '" << getContext()->getTime()  << "'" ;

            msg_error() << " a = " << a << " b = " << b << msgendl
                             << "ind_ta = " << ind_ta << " ind_tb = " << ind_tb ;

            break;
        }
        else
        {
            errorTrianglesIndices.pop_back();
            errorTrianglesIndices.pop_back();
        }

        sofa::type::vector< Index > new_edges;

        //Split triangles to create edges along a path given as a the list of existing edges and triangles crossed by it.
        triangleGeo->SplitAlongPath(a_last, a, b_last, b, topoPath_list, indices_list, coords2_list, new_edges, m_epsilonSnapPath.getValue(), m_epsilonSnapBorder.getValue());

        sofa::type::vector<Index> new_points;
        sofa::type::vector<Index> end_points;
        bool reachBorder = false;

        //Duplicates the given edges
        triangleGeo->InciseAlongEdgeList(new_edges, new_points, end_points, reachBorder);

        msg_info_when(reachBorder) << "Incision has reached a border.";

        if (!end_points.empty())
        {
            a_last = end_points.back();
        }
        ind_ta = ind_tb;
        firstCut=false;

        // notify the end for the current sequence of topological change events
        triangleMod->notifyEndingEvent();

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
    const size_t nbTriangleInfo = triangleIncisionInformation.size();
    sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    for ( size_t i = 0 ; i < nbTriangleInfo ; i++)
    {
        for (unsigned int j = 0 ; j < triangleIncisionInformation[i].triangleIndices.size() ; j++ )
        {
            //update the triangle index corresponding to the current coordinates
            Index newTriangleIndexb;
            Index currentTriangleIndex = triangleIncisionInformation[i].triangleIndices[j];

            if ( j >= triangleIncisionInformation[i].coordinates.size() || triangleIncisionInformation[i].coordinates.empty())
            {
                msg_warning() << "(updateTriangleIncisionInformation): error accessing coordinates" ;
                break;
            }

            findElementIndex(triangleIncisionInformation[i].coordinates[j], newTriangleIndexb, currentTriangleIndex);

            if ( newTriangleIndexb == sofa::InvalidID)
            {
                msg_warning() << "(updateTriangleIncisionInformation): error while finding the point " << triangleIncisionInformation[i].coordinates[j] << " in a new triangle. Current triangle index = " << currentTriangleIndex ;
                break;
            }

            msg_info_when(currentTriangleIndex != newTriangleIndexb)
                          << "(updateTriangleIncisionInformation): incision point which was in triangle " << currentTriangleIndex
                          << " has been updated to " << newTriangleIndexb  ;

            triangleIncisionInformation[i].triangleIndices[j] = newTriangleIndexb;

            //update the triangle barycentric coordinates corresponding to the current coordinates
            const Vec3 constCoord = triangleIncisionInformation[i].coordinates[j];
            const auto baryCoef = triangleGeo->computeTriangleBarycoefs( newTriangleIndexb, constCoord);
            triangleIncisionInformation[i].barycentricCoordinates[j] = Vec3(baryCoef[0], baryCoef[1], baryCoef[2]);
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

    sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<Vec3Types>* triangleGeo;
    m_topology->getContext()->get(triangleGeo);

    if (!triangleGeo)
        return;

    const size_t nbTriangles = m_topology->getNbTriangles();

    std::vector< Vec3 > trianglesToDraw;
    std::vector< Vec3 > pointsToDraw;

    for (size_t i = 0 ; i < triangleIncisionInformation.size() ; i++)
    {
        for (size_t j = 0 ; j < triangleIncisionInformation[i].triangleIndices.size() ; j++)
        {
            const unsigned int triIndex = triangleIncisionInformation[i].triangleIndices[j];

            if ( triIndex > nbTriangles -1)
                break;

            Vec3Types::Coord coord[3];
            triangleGeo->getTriangleVertexCoordinates(triIndex, coord);

            for(unsigned int k = 0 ; k < 3 ; k++)
                trianglesToDraw.push_back(coord[k]);

            Vec3 a;
            a.clear();
            for (unsigned k = 0 ; k < 3 ; k++)
                a += coord[k] * triangleIncisionInformation[i].barycentricCoordinates[j][k];

            pointsToDraw.push_back(a);
        }
    }

    vparams->drawTool()->drawTriangles(trianglesToDraw, sofa::type::RGBAColor::blue());
    vparams->drawTool()->drawPoints(pointsToDraw, 15.0,  sofa::type::RGBAColor::magenta());

    if (!errorTrianglesIndices.empty())
    {
        trianglesToDraw.clear();
        /* initialize random seed: */
        srand ( (unsigned int)time(nullptr) );

        for (const unsigned int errorTrianglesIndex : errorTrianglesIndices)
        {
            Vec3Types::Coord coord[3];
            triangleGeo->getTriangleVertexCoordinates(errorTrianglesIndex, coord);

            for(auto & k : coord)
                trianglesToDraw.push_back(k);
        }

        vparams->drawTool()->drawTriangles(trianglesToDraw,
                sofa::type::RGBAColor(1.0f,(float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX, 1.0f));
    }
}

} // namespace sofa::component::topology::utility
