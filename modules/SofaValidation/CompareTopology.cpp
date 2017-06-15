/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#ifndef SOFA_COMPONENT_MISC_COMPARETOPOLOGY_INL
#define SOFA_COMPONENT_MISC_COMPARETOPOLOGY_INL

#include <SofaValidation/CompareTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <sstream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{



int CompareTopologyClass = core::RegisterObject("Compare Topology containers from a reference frame to the associated Topology")
        .add< CompareTopology >();

CompareTopology::CompareTopology(): ReadTopology()
{
    EdgesError = 0;
    TrianglesError = 0;
    QuadsError = 0;
    TetrahedraError = 0;
    HexahedraError = 0;
    TotalError = 0;
}


//-------------------------------- handleEvent-------------------------------------------
void CompareTopology::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        processCompareTopology();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {
    }
}


//-------------------------------- processCompareTopology------------------------------------
void CompareTopology::processCompareTopology()
{

    sofa::core::topology::BaseMeshTopology* topo = m_topology = this->getContext()->getMeshTopology();
    if (!topo)
    {
        serr << "Error, compareTopology can't acces to the Topology." << sendl;
        return;
    }

    double time = getContext()->getTime() + f_shift.getValue();
    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;

    unsigned int nbr = 0;
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end();)
    {
        //For one Timestep store all topology data available.

        std::string buff;
        int tmp;
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
            //Looking fo the number of edges, if not null, then compare it to the actual topology.
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                const helper::vector <core::topology::Topology::Edge>& SeqEdges = topo->getEdges();

                if ((unsigned int)topo->getNbEdges() != nbr)
                {
                    tmp = nbr - topo->getNbEdges();
                    EdgesError += (unsigned int)abs (tmp);
                }
                else
                {
                    std::istringstream Sedges(*it);
                    for (unsigned int i = 0; i<nbr; ++i)
                    {
                        helper::fixed_array <unsigned int,2> nodes;
                        Sedges >> nodes[0] >> nodes[1];

                        if ( nodes[0] != SeqEdges[i][0] || nodes[1] != SeqEdges[i][1] )
                            EdgesError++;
                    }
                }
            }

            ++it;
            continue;
        }
        else if ( buff == "Triangles=")
        {
            //Looking fo the number of Triangles, if not null, then compare it to the actual topology.
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                const core::topology::BaseMeshTopology::SeqTriangles& SeqTriangles = topo->getTriangles();

                if ((unsigned int)topo->getNbTriangles() != nbr)
                {
                    tmp = nbr - topo->getNbTriangles();
                    TrianglesError += (unsigned int)abs (tmp);
                }
                else
                {
                    std::istringstream Stri(*it);
                    for (unsigned int i = 0; i<nbr; ++i)
                    {
                        helper::fixed_array <unsigned int,3> nodes;
                        Stri >> nodes[0] >> nodes[1] >> nodes[2];

                        for (unsigned int j = 0; j<3; ++j)
                            if (nodes[j] != SeqTriangles[i][j])
                            {
                                TrianglesError++;
                                break;
                            }
                    }
                }
            }

            ++it;
            continue;
        }
        else if ( buff == "Quads=")
        {
            //Looking fo the number of Quads, if not null, then compare it to the actual topology.
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                const core::topology::BaseMeshTopology::SeqQuads& SeqQuads = topo->getQuads();

                if ((unsigned int)topo->getNbQuads() != nbr)
                {
                    tmp = nbr - topo->getNbQuads();
                    QuadsError += (unsigned int)abs (tmp);
                }
                else
                {
                    std::istringstream Squads(*it);
                    for (unsigned int i = 0; i<nbr; ++i)
                    {
                        helper::fixed_array <unsigned int,4> nodes;
                        Squads >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3];

                        for (unsigned int j = 0; j<4; ++j)
                            if (nodes[j] != SeqQuads[i][j])
                            {
                                QuadsError++;
                                break;
                            }
                    }
                }
            }

            ++it;
            continue;
        }
        else if ( buff == "Tetrahedra=")
        {
            //Looking fo the number of Tetrahedra, if not null, then compare it to the actual topology.
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                const core::topology::BaseMeshTopology::SeqTetrahedra& SeqTetrahedra = topo->getTetrahedra();

                if ((unsigned int)topo->getNbTetrahedra() != nbr)
                {
                    tmp = nbr - topo->getNbTetrahedra();
                    TetrahedraError += (unsigned int)abs (tmp);
                }
                else
                {
                    std::istringstream Stetra(*it);
                    for (unsigned int i = 0; i<nbr; ++i)
                    {
                        helper::fixed_array <unsigned int,4> nodes;
                        Stetra >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3];

                        for (unsigned int j = 0; j<4; ++j)
                            if (nodes[j] != SeqTetrahedra[i][j])
                            {
                                TetrahedraError++;
                                break;
                            }
                    }
                }
            }

            ++it;
            continue;
        }
        else if ( buff == "Hexahedra=")
        {
            //Looking fo the number of Hexahedra, if not null, then compare it to the actual topology.
            str >> nbr;
            ++it;

            if (nbr != 0)
            {
                const core::topology::BaseMeshTopology::SeqHexahedra& SeqHexahedra = topo->getHexahedra();

                if ((unsigned int)topo->getNbHexahedra() != nbr)
                {
                    tmp = nbr - topo->getNbHexahedra();
                    HexahedraError += (unsigned int)abs (tmp);
                }
                else
                {
                    std::istringstream Shexa(*it);
                    for (unsigned int i = 0; i<nbr; ++i)
                    {
                        helper::fixed_array <unsigned int,8> nodes;
                        Shexa >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> nodes[4] >> nodes[5] >> nodes[6] >> nodes[7];

                        for (unsigned int j = 0; j<8; ++j)
                            if (nodes[j] != SeqHexahedra[i][j])
                            {
                                HexahedraError++;
                                break;
                            }
                    }
                }
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

    // Sum all errors from different containers and storing infos in a vector
    listError.resize (5);
    listError[0] = EdgesError;
    listError[1] = TrianglesError;
    listError[2] = QuadsError;
    listError[3] = TetrahedraError;
    listError[4] = HexahedraError;

    TotalError = EdgesError + TrianglesError + QuadsError + TetrahedraError + HexahedraError;


}





CompareTopologyCreator::CompareTopologyCreator(const core::ExecParams* params)
    :Visitor(params)
    , sceneName("")
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(true)
    , counterCompareTopology(0)
{
}

CompareTopologyCreator::CompareTopologyCreator(const std::string &n, const core::ExecParams* params, bool i, int c)
    :Visitor(params)
    , sceneName(n)
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(i)
    , counterCompareTopology(c)
{
}

//Create a Compare Topology component each time a mechanical state is found
simulation::Visitor::Result CompareTopologyCreator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;
    sofa::core::topology::BaseMeshTopology* topo = dynamic_cast<sofa::core::topology::BaseMeshTopology *>( gnode->getMeshTopology());
    if (!topo)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a meshTopology
    addCompareTopology(topo, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}



void CompareTopologyCreator::addCompareTopology(sofa::core::topology::BaseMeshTopology* topology, simulation::Node* gnode)
{

    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping;
    context->get(mapping);
    if (createInMapping || mapping== NULL)
    {
        sofa::component::misc::CompareTopology::SPtr ct; context->get(ct, core::objectmodel::BaseContext::Local);
        if (  ct == NULL )
        {
            ct = sofa::core::objectmodel::New<sofa::component::misc::CompareTopology>(); gnode->addObject(ct);
        }

        std::ostringstream ofilename;
        ofilename << sceneName << "_" << counterCompareTopology << "_" << topology->getName()  << "_topology" << extension ;

        ct->f_filename.setValue(ofilename.str());  ct->f_listening.setValue(false); //Deactivated only called by extern functions
        if (init) ct->init();

        ++counterCompareTopology;
    }
}


CompareTopologyResult::CompareTopologyResult(const core::ExecParams* params)
    :Visitor(params)
{
    TotalError = 0;
    numCompareTopology = 0;
    listError.resize (5);
    for (unsigned int i = 0; i<5; i++)
        listError[i] = 0;
}



//Create a Compare Topology result component each time a topology is found
simulation::Visitor::Result CompareTopologyResult::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::CompareTopology *ct;
    gnode->get(ct);
    if (!ct)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a topology
    TotalError +=ct->getTotalError();

    std::vector <unsigned int> tmpError = ct->getErrors();

    for (unsigned int i = 0 ; i<5; i++)
        listError[i] += tmpError[i];

    numCompareTopology++;
    return simulation::Visitor::RESULT_CONTINUE;
}



} // namespace misc

} // namespace component

} // namespace sofa

#endif
