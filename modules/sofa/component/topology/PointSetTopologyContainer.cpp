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
#include <sofa/component/topology/PointSetTopologyContainer.h>

#include <sofa/simulation/common/Node.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PointSetTopologyContainer)
int PointSetTopologyContainerClass = core::RegisterObject("Point set topology container")
        .add< PointSetTopologyContainer >()
        ;

PointSetTopologyContainer::PointSetTopologyContainer(int npoints)
    : nbPoints (initData(&nbPoints, (unsigned int )npoints, "nbPoints", "Number of points"))
    , d_initPoints (initData(&d_initPoints, "position", "Initial position of points"))
{
    addAlias(&d_initPoints,"points");
}

void PointSetTopologyContainer::setNbPoints(int n)
{
    nbPoints.setValue(n);
}

unsigned int PointSetTopologyContainer::getNumberOfElements() const
{
    return nbPoints.getValue();
}

bool PointSetTopologyContainer::checkTopology() const
{
    return true;
}

void PointSetTopologyContainer::clear()
{
    nbPoints.setValue(0);
    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.clear();
}

void PointSetTopologyContainer::addPoint(double px, double py, double pz)
{
    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.push_back(InitTypes::Coord((SReal)px, (SReal)py, (SReal)pz));
    if (initPoints.size() > nbPoints.getValue())
        nbPoints.setValue(initPoints.size());
}

bool PointSetTopologyContainer::hasPos() const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    return !initPoints.empty();
}

double PointSetTopologyContainer::getPX(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][0];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPY(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][1];
    else
        return 0.0;
}

double PointSetTopologyContainer::getPZ(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][2];
    else
        return 0.0;
}

void PointSetTopologyContainer::init()
{
    core::topology::TopologyContainer::init();

    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if (nbPoints.getValue() == 0 && !initPoints.empty())
        nbPoints.setValue(initPoints.size());
}

void PointSetTopologyContainer::addPoints(const unsigned int nPoints)
{
    nbPoints.setValue( nbPoints.getValue() + nPoints);
}

void PointSetTopologyContainer::removePoints(const unsigned int nPoints)
{
    nbPoints.setValue(nbPoints.getValue() - nPoints);
}

void PointSetTopologyContainer::addPoint()
{
    nbPoints.setValue(nbPoints.getValue()+1);
}

void PointSetTopologyContainer::removePoint()
{
    nbPoints.setValue(nbPoints.getValue()-1);
}

#ifdef SOFA_HAVE_NEW_TOPOLOGYCHANGES
void PointSetTopologyContainer::updateTopologyEngineGraph()
{
    std::cout << "PointSetTopologyContainer::updateTopologyEngineGraph()" << std::endl;
    this->updateDataEngineGraph(this->d_initPoints, this->m_enginesList);

    //std::cout << "PointSetTopologyContainer::updateTopologyEngineGraph() end" << std::endl;
}



void PointSetTopologyContainer::updateDataEngineGraph(sofa::core::objectmodel::BaseData &my_Data, sofa::helper::list<sofa::core::topology::TopologyEngine *> &my_enginesList)
{
    std::cout << "updateDataEngineGraph()" << std::endl;

    // clear data stored by previous call of this function
    my_enginesList.clear();
    this->m_enginesGraph.clear();
    this->m_dataGraph.clear();


    sofa::helper::list <sofa::core::objectmodel::DDGNode* > _outs = my_Data.getOutputs();
    sofa::helper::list <sofa::core::objectmodel::DDGNode* >::iterator it;

    //std::cout << "PointSetTopologyContainer - Number of outputs for points array: " << _outs.size() << std::endl;

    bool allDone = false;

    unsigned int cpt_security = 0;
    sofa::helper::list <sofa::core::topology::TopologyEngine *> _engines;
    sofa::helper::list <sofa::core::topology::TopologyEngine *>::iterator it_engines;

    while (!allDone && cpt_security < 1000)
    {
        sofa::helper::list <sofa::core::objectmodel::DDGNode* > next_GraphLevel;
        sofa::helper::list <sofa::core::topology::TopologyEngine *> next_enginesLevel;

        // for drawing graph
        sofa::helper::vector <std::string> enginesNames;
        sofa::helper::vector <std::string> dataNames;

        // doing one level of data outputs, looking for engines
        for ( it = _outs.begin(); it!=_outs.end(); ++it)
        {
            sofa::core::topology::TopologyEngine* topoEngine = dynamic_cast <sofa::core::topology::TopologyEngine*> ( (*it));

            if (topoEngine)
            {
                //std::cout << "topoEngine here: "<< topoEngine->getName() << std::endl;
                next_enginesLevel.push_back(topoEngine);
                enginesNames.push_back(topoEngine->getName());
            }

            sofa::core::objectmodel::BaseData* data = dynamic_cast<sofa::core::objectmodel::BaseData*>( (*it) );
            if (data)
            {
                std::cout << "Data alone linked: " << data->getName() << std::endl;
            }
        }

        _outs.clear();

        // looking for data linked to engines
        for ( it_engines = next_enginesLevel.begin(); it_engines!=next_enginesLevel.end(); ++it_engines)
        {
            // for each output engine, looking for data outputs

            // There is a conflict with Base::getOutputs()
            sofa::core::objectmodel::DDGNode* my_topoEngine = (*it_engines);
            sofa::helper::list <sofa::core::objectmodel::DDGNode* > _outsTmp = my_topoEngine->getOutputs();
            sofa::helper::list <sofa::core::objectmodel::DDGNode* >::iterator itTmp;

            for ( itTmp = _outsTmp.begin(); itTmp!=_outsTmp.end(); ++itTmp)
            {
                sofa::core::objectmodel::BaseData* data = dynamic_cast<sofa::core::objectmodel::BaseData*>( (*itTmp) );
                if (data)
                {
                    //std::cout << "PointSetTopologyModifier - Data linked to engine: " << data->getName() << std::endl;
                    next_GraphLevel.push_back((*itTmp));
                    dataNames.push_back(data->getName());

                    sofa::helper::list <sofa::core::objectmodel::DDGNode* > _outsTmp2 = data->getOutputs();
                    _outs.insert(_outs.end(), _outsTmp2.begin(), _outsTmp2.end());
                }
            }

            this->m_dataGraph.push_back(dataNames);
            dataNames.clear();
        }


        // Iterate:
        _engines.insert(_engines.end(), next_enginesLevel.begin(), next_enginesLevel.end());
        this->m_enginesGraph.push_back(enginesNames);

        if (next_GraphLevel.empty()) // end
            allDone = true;

        next_GraphLevel.clear();
        next_enginesLevel.clear();
        enginesNames.clear();

        cpt_security++;
    }


    // check good loop escape
    if (cpt_security >= 1000)
        std::cerr << "Error: PointSetTopologyContainer::updateTopologyEngineGraph reach end loop security." << std::endl;


    // Reorder engine graph by inverting order and avoiding duplicate engines
    sofa::helper::list <sofa::core::topology::TopologyEngine *>::reverse_iterator it_engines_rev;

    //std::cout << "DEBUG: _engines size: " << _engines.size() << std::endl;


    for ( it_engines_rev = _engines.rbegin(); it_engines_rev != _engines.rend(); ++it_engines_rev)
    {
        std::string name = (*it_engines_rev)->getName();
        //std::cout << "engine name: " << name << std::endl;
        bool find = false;

        for ( it_engines = my_enginesList.begin(); it_engines!=my_enginesList.end(); ++it_engines)
        {
            std::string nameStored = (*it_engines)->getName();
            //std::cout << "engine name stored: " << nameStored << std::endl;
            if (nameStored == name)
            {
                std::cout << "found!" << std::endl;
                find = true;
                break;
            }
        }

        if (!find)
            my_enginesList.push_back((*it_engines_rev));
        //this->addEngineToList((*it_engines_rev));
    }

    for ( it_engines = my_enginesList.begin(); it_engines!=my_enginesList.end(); ++it_engines)
        std::cout << (*it_engines)->getName() << "   -------- ";
    std::cout << std::endl;

    std::cout << "updateDataEngineGraph() end" << std::endl;

    this->displayDataGraph(my_Data);
    return;
}


void PointSetTopologyContainer::displayDataGraph(sofa::core::objectmodel::BaseData& my_Data)
{
    std::cout << "displayDataGraph()" << std::endl;
    // A cout very lite version
    std::string name;

    name = my_Data.getName();
    std::cout << name << std::endl;
    std::cout << std::endl;

    unsigned int cpt_engine = 0;




    for (unsigned int i=0; i<this->m_enginesGraph.size(); ++i ) // per engine level
    {
        sofa::helper::vector <std::string> enginesNames = this->m_enginesGraph[i];

        unsigned int cpt_engine_tmp = cpt_engine;
        for (unsigned int j=0; j<enginesNames.size(); ++j) // per engine on the same level
        {
            std::cout << enginesNames[j];

            for (unsigned int k=0; k<this->m_enginesGraph[cpt_engine].size(); ++k) // create espace between engines name
                std::cout << "     ";

            cpt_engine++;
        }
        std::cout << std::endl;
        cpt_engine = cpt_engine_tmp;


        for (unsigned int j=0; j<enginesNames.size(); ++j) // per engine on the same level
        {
            sofa::helper::vector <std::string> dataNames = this->m_dataGraph[cpt_engine];
            for (unsigned int k=0; k<dataNames.size(); ++k)
                std::cout << dataNames[k] << "     " ;
            std::cout << "            ";

            cpt_engine++;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }


    std::cout << "displayDataGraph() end" << std::endl;
}

#endif

} // namespace topology

} // namespace component

} // namespace sofa

