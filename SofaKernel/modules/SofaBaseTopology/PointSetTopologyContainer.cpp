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
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/Node.h>
#include <sofa/core/ObjectFactory.h>

#include <algorithm>

namespace sofa
{

namespace component
{

namespace topology
{

namespace
{

struct GeneratePointID
{
    typedef sofa::core::topology::BaseMeshTopology::PointID PointID;

    GeneratePointID( PointID startId = PointID(0) )
    :current(startId)
    {
    }

    PointID operator() () { return current++; }

    PointID current;
};

}

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PointSetTopologyContainer)
int PointSetTopologyContainerClass = core::RegisterObject("Point set topology container")
        .add< PointSetTopologyContainer >()
        ;

PointSetTopologyContainer::PointSetTopologyContainer(int npoints)
    : d_initPoints (initData(&d_initPoints, "position", "Initial position of points",true,true))
    , m_pointTopologyDirty(false)
    , nbPoints (initData(&nbPoints, (unsigned int )npoints, "nbPoints", "Number of points"))
    , points(initData(&points, "points","List of point indices"))
{
    addAlias(&d_initPoints,"points");
}

void PointSetTopologyContainer::setNbPoints(int n)
{

    int diffSize = n - (int)nbPoints.getValue();
    sofa::helper::WriteAccessor< sofa::Data< sofa::helper::vector<PointID> > > points = this->points;
    points.resize(n);

    if( diffSize > 0 )
    {
        GeneratePointID generator( PointID( nbPoints.getValue() ) );
        std::generate( points.begin()+nbPoints.getValue(), points.end(), generator );
    }

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
    sofa::helper::WriteAccessor< sofa::Data< sofa::helper::vector<PointID> > > points = this->points;
    points.clear();
}

void PointSetTopologyContainer::addPoint(double px, double py, double pz)
{
    // NB: This implementation of addPoint was and is still very dangerous to use since it compromises any prior 
    // modifications that were done on the container. The new size is imposed by the size of the initPoints array,
    // which is not maintained whatsoever by the other add / remove point methods.

    helper::WriteAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    initPoints.push_back(InitTypes::Coord((SReal)px, (SReal)py, (SReal)pz));
    if (initPoints.size() > nbPoints.getValue())
    {
        setNbPoints(initPoints.size());
    }
}

bool PointSetTopologyContainer::hasPos() const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    return !initPoints.empty();
}

SReal PointSetTopologyContainer::getPX(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][0];
    else
        return 0.0;
}

SReal PointSetTopologyContainer::getPY(int i) const
{
    helper::ReadAccessor< Data<InitTypes::VecCoord> > initPoints = d_initPoints;
    if ((unsigned)i < initPoints.size())
        return initPoints[i][1];
    else
        return 0.0;
}

SReal PointSetTopologyContainer::getPZ(int i) const
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
    int pointsDiff = (int)initPoints.size() - (int)getNbPoints(); 
    if( pointsDiff > 0 )
    {
        addPoints( pointsDiff );
    }

}

void PointSetTopologyContainer::addPoints(const unsigned int nPoints)
{
    //nbPoints.setValue( nbPoints.getValue() + nPoints);
    setNbPoints( nbPoints.getValue() + nPoints );
}

void PointSetTopologyContainer::removePoints(const unsigned int nPoints)
{
    //nbPoints.setValue(nbPoints.getValue() - nPoints);
    setNbPoints( nbPoints.getValue() - nPoints );
}

void PointSetTopologyContainer::addPoint()
{
    //nbPoints.setValue(nbPoints.getValue()+1);
    setNbPoints( nbPoints.getValue() + 1 );
}

void PointSetTopologyContainer::removePoint()
{
    //nbPoints.setValue(nbPoints.getValue()-1);
    setNbPoints( nbPoints.getValue() -1 );
}


void PointSetTopologyContainer::updateTopologyEngineGraph()
{
    this->updateDataEngineGraph(this->d_initPoints, this->m_enginesList);
}


void PointSetTopologyContainer::addEngineToList(sofa::core::topology::TopologyEngine *_engine)
{
    this->m_enginesList.push_back(_engine);
}

const sofa::helper::vector< PointSetTopologyContainer::PointID >& PointSetTopologyContainer::getPoints() const
{
    return points.getValue();
}


void PointSetTopologyContainer::updateDataEngineGraph(sofa::core::objectmodel::BaseData &my_Data, std::list<sofa::core::topology::TopologyEngine *> &my_enginesList)
{
    // clear data stored by previous call of this function
    my_enginesList.clear();
    this->m_enginesGraph.clear();
    this->m_dataGraph.clear();


    sofa::core::objectmodel::DDGNode::DDGLinkContainer _outs = my_Data.getOutputs();
    sofa::core::objectmodel::DDGNode::DDGLinkIterator it;

    bool allDone = false;

    unsigned int cpt_security = 0;
    std::list<sofa::core::topology::TopologyEngine *> _engines;
    std::list<sofa::core::topology::TopologyEngine *>::iterator it_engines;

    while (!allDone && cpt_security < 1000)
    {
        std::list<sofa::core::objectmodel::DDGNode* > next_GraphLevel;
        std::list<sofa::core::topology::TopologyEngine *> next_enginesLevel;

        // for drawing graph
        sofa::helper::vector <std::string> enginesNames;
        sofa::helper::vector <std::string> dataNames;

        // doing one level of data outputs, looking for engines
        for ( it = _outs.begin(); it!=_outs.end(); ++it)
        {
            sofa::core::topology::TopologyEngine* topoEngine = dynamic_cast <sofa::core::topology::TopologyEngine*> ( (*it));

            if (topoEngine)
            {
                next_enginesLevel.push_back(topoEngine);
                enginesNames.push_back(topoEngine->getName());
            }

            sofa::core::objectmodel::BaseData* data = dynamic_cast<sofa::core::objectmodel::BaseData*>( (*it) );
            if (data)
            {
                sout << "Warning: Data alone linked: " << data->getName() << sendl;
            }
        }

        _outs.clear();

        // looking for data linked to engines
        for ( it_engines = next_enginesLevel.begin(); it_engines!=next_enginesLevel.end(); ++it_engines)
        {
            // for each output engine, looking for data outputs

            // There is a conflict with Base::getOutputs()
            sofa::core::objectmodel::DDGNode* my_topoEngine = (*it_engines);
            const sofa::core::objectmodel::DDGNode::DDGLinkContainer& _outsTmp = my_topoEngine->getOutputs();
            sofa::core::objectmodel::DDGNode::DDGLinkIterator itTmp;

            for ( itTmp = _outsTmp.begin(); itTmp!=_outsTmp.end(); ++itTmp)
            {
                sofa::core::objectmodel::BaseData* data = dynamic_cast<sofa::core::objectmodel::BaseData*>( (*itTmp) );
                if (data)
                {
                    next_GraphLevel.push_back((*itTmp));
                    dataNames.push_back(data->getName());

                    const sofa::core::objectmodel::DDGNode::DDGLinkContainer& _outsTmp2 = data->getOutputs();
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
        serr << "Error: PointSetTopologyContainer::updateTopologyEngineGraph reach end loop security." << sendl;


    // Reorder engine graph by inverting order and avoiding duplicate engines
    std::list<sofa::core::topology::TopologyEngine *>::reverse_iterator it_engines_rev;

    for ( it_engines_rev = _engines.rbegin(); it_engines_rev != _engines.rend(); ++it_engines_rev)
    {
        bool find = false;

        for ( it_engines = my_enginesList.begin(); it_engines!=my_enginesList.end(); ++it_engines)
        {
            if ((*it_engines_rev) == (*it_engines))
            {
                find = true;
                break;
            }
        }

        if (!find)
            my_enginesList.push_back((*it_engines_rev));
    }

    return;
}


void PointSetTopologyContainer::displayDataGraph(sofa::core::objectmodel::BaseData& my_Data)
{
    // A cout very lite version
    std::string name;
    std::stringstream tmpmsg;
    name = my_Data.getName();
    tmpmsg << name << msgendl << msgendl;

    unsigned int cpt_engine = 0;


    for (unsigned int i=0; i<this->m_enginesGraph.size(); ++i ) // per engine level
    {
        sofa::helper::vector <std::string> enginesNames = this->m_enginesGraph[i];

        unsigned int cpt_engine_tmp = cpt_engine;
        for (unsigned int j=0; j<enginesNames.size(); ++j) // per engine on the same level
        {
            tmpmsg << enginesNames[j];

            for (unsigned int k=0; k<this->m_enginesGraph[cpt_engine].size(); ++k) // create espace between engines name
                tmpmsg << "     ";

            cpt_engine++;
        }
        tmpmsg << msgendl;
        cpt_engine = cpt_engine_tmp;


        for (unsigned int j=0; j<enginesNames.size(); ++j) // per engine on the same level
        {
            sofa::helper::vector <std::string> dataNames = this->m_dataGraph[cpt_engine];
            for (unsigned int k=0; k<dataNames.size(); ++k)
                tmpmsg << dataNames[k] << "     " ;
            tmpmsg << "            ";

            cpt_engine++;
        }
        tmpmsg << msgendl ;
        msg_info() << tmpmsg.str() ;
    }
}


} // namespace topology

} // namespace component

} // namespace sofa

