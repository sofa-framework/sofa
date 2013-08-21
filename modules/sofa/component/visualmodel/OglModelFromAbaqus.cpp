/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/visualmodel/OglModelFromAbaqus.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/core/CollisionModel.h>
#include <sstream>
#include <cctype>
#include <string.h>

namespace sofa
{

namespace component
{

namespace visualmodel
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(OglModelFromAbaqus)

int OglModelFromAbaqusClass = core::RegisterObject("Specific visual model for OpenGL display")
        .add< OglModelFromAbaqus >()
        ;

OglModelFromAbaqus::OglModelFromAbaqus()
    : filePath(initData(&filePath, "filePath"," Path to the model"))
    , m_radius(initData(&m_radius,(float)(1.0), "radius","Radius of all spheres"))
    , m_currentTime( initData(&m_currentTime, "currentTime", "current time"))
    , m_difference( initData(&m_difference, "difference", "distance for each point between the Sofa and Abaqus models"))
    , m_updateDifference(initData(&m_updateDifference,(bool)true, "updateDifference","update distance between models coordinate ?"))
    , m_updateStressVector(initData(&m_updateStressVector,(bool)false, "updateStressVector","update the stress vector ?"))
{
    this->f_listening.setValue(true);
}

OglModelFromAbaqus::~OglModelFromAbaqus()
{
}

void OglModelFromAbaqus::init()
{
    context = this->getContext();
    context->get(mstate); // MechanicalState
    
    if(!mstate)
    {
        serr << "OglModelFromAbaqus : error, no mechanical state." << sendl;
        return;
    }
    
    sofa::core::objectmodel::BaseData* nam = NULL;
    sofa::core::objectmodel::BaseData* timeT = NULL;
    sofa::core::objectmodel::BaseData* pos = NULL;
    
    if(context)
    {
        nam = context->findField("name");
        timeT = context->findField("time");
        m_currentTime.setParent(timeT);
    }
    
    pos = mstate->findField("position");
    if(!pos)
    {
        serr << "INPExporter : error, missing positions in mechanical state" << sendl;
        return;
    }
    m_name.setParent(nam);
    m_position.setParent(pos);
    
}

void OglModelFromAbaqus::bwdInit()
{
    std::string filename = filePath.getFullPath();
    if (!sofa::helper::system::DataRepository.findFile(filename))
    {
        std::cerr << "File " << filename << " not found " << std::endl;
        return;
    }
    std::ifstream file(filename.c_str());
    if (file.good())
    {
        readRPTFile(filename);
    }
    else
        std::cerr << "Error: Cannot read file '" << filename << "'." << std::endl;
    
    file.close();
    
    m_frameCount = 0;
    m_timeNextFrame = m_vecFrame[0].second;
    
}

void OglModelFromAbaqus::updateDifference()
{
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > posIndices = m_position;
    helper::WriteAccessor<Data< vector<double> > > difIndices = m_difference;
    
    unsigned int vecSize = m_vecFrame[m_frameCount-1].first.size();
    difIndices.resize(vecSize);
    for(unsigned int i=0;i<vecSize;i++)
    {
        if(vecSize!=posIndices.size())
        {
            serr << "Error: mismatch in number of vertex" << sendl;
            return;
        }
        difIndices[i] = sqrt(pow(posIndices[i][0] - m_vecFrame[m_frameCount].first[i][0],2.0)
                           + pow(posIndices[i][1] - m_vecFrame[m_frameCount].first[i][1],2.0)
                           + pow(posIndices[i][2] - m_vecFrame[m_frameCount].first[i][2],2.0));
    }
}

void OglModelFromAbaqus::updateStressVector()
{
    helper::WriteAccessor<Data< vector< defaulttype::Vec3f > > > stressIndices = m_stress;
    
    unsigned int vecSize = m_vecStress[m_frameCount-1].first.size();
    stressIndices.resize(vecSize);
    for(unsigned int i=0;i<vecSize;i++)
    {
        for(int j=0;j<3;j++)
        {
            stressIndices[i][j] = m_vecStress[m_frameCount].first[i][j];
        }
    }
    
}

void OglModelFromAbaqus::readRPTFile(const std::string filename)
{
    /* The purpose of this function is to read data from the .rpt file generated from Abaqus and
     * store the time, and points coordinates for each frame */
    std::ifstream file(filename.c_str());
    std::string line;
    
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > startPos = m_position;
    
    int frame = -1;
    double time = -1.0;
    unsigned int vertexCount = 0;
    int stressDisplacement= -1;
    bool dataIsGood = 0;
    
    std::pair< vector< defaulttype::Vec3f >, double> coordTime;
    std::pair< vector< defaulttype::Vec3f >, double> stressTime;
    vector< defaulttype::Vec3f > vecCo;
    vector< defaulttype::Vec3f > vecSt;
    
    while( std::getline(file,line) )
    {
        if (line.empty()) continue;
        std::istringstream values(line);
        std::string token;
        values >> token;
        if (token == "Frame:")
        {
            if(!vecCo.empty())
            {
                // We store the last displacement vector and time in a structure
                coordTime = std::make_pair(vecCo,time);
                m_vecFrame.push_back(coordTime);
                vecCo.clear();
                vertexCount = 0;
            }
            if(!vecSt.empty())
            {
                // We store the last stress vector and time in a structure
                stressTime = std::make_pair(vecSt,time);
                m_vecStress.push_back(stressTime);
                vecSt.clear();
                vertexCount = 0;
            }
            
            std::string temp1, temp2;
            std::stringstream cvrt;
            values >> temp1 >> temp2;
            temp2.resize(temp2.size()-1);
            frame = atoi(temp2.c_str());
            
            while(!isdigit(temp1.c_str()[0]))
            {
                values >> temp1;
            }
            
            time = atof(temp1.c_str());
            //std::cout << "Frame : " << frame << "    time : " << time << std::endl;
        }
        else if(token == "Node")
        {
            std::string temp;
            values >> temp;
            if(temp=="U.U1")
            {
                stressDisplacement = 0;
            }
            else if(temp=="S.S11")
            {
                stressDisplacement = 1;
            }
            else
            {
                serr << "Error: token '" << temp << "' unexpected" << sendl;
                return;
            }
        }
        else if(token == "Field")
        {
            std::string temp1, temp2;
            values >> temp1 >> temp2;
            if(temp1=="Output" && temp2=="reported")
            {
                while(temp1!="part:")
                {
                    values >> temp1;
                }
                values >> temp1;
                std::string partName = temp1.substr(0,temp1.find('_'));
                std::string nodeName = m_name.getValue();
                std::transform(nodeName.begin(), nodeName.end(), nodeName.begin(), ::toupper);
                if(partName == nodeName)
                {
                    dataIsGood = 1;
                }
                else
                {
                    dataIsGood = 0;
                }
            }
        }
        else if(isdigit(token.c_str()[0])) // if token is a number
        {
            if(dataIsGood)
            {
                vertexCount++;
                if(stressDisplacement==0)
                {
                    if(vertexCount>startPos.size())
                    {
                        serr << "Error: mismatch in number of vertex (too many vertices in .rpt)" << sendl;
                        return;
                    }
                    if(frame>-1 && time>-1.0)
                    {
                        defaulttype::Vec3f coord;
                        for(int i=0;i<3;i++)
                        {
                            values >> token;
                            coord[i]=startPos[vertexCount-1][i] + atof( token.c_str() );
                        }
                        vecCo.push_back(coord);
                    }
                    else
                    {
                        serr << "Error: bad file format" << sendl;
                        return;
                    }
                }
                else if(stressDisplacement==1)
                {
                    if(vertexCount>startPos.size())
                    {
                        serr << "Error: mismatch in number of vertex (too many vertices in .rpt)" << sendl;
                        return;
                    }
                    if(frame>-1 && time>-1.0)
                    {
                        defaulttype::Vec3f stress;
                        for(int i=0;i<3;i++)
                        {
                            values >> token;
                            stress[i]=atof( token.c_str() );
                        }
                        vecSt.push_back(stress);
                    }
                    else
                    {
                        serr << "Error: bad file format" << sendl;
                        return;
                    }
                }
                else
                {
                    serr << "Error: bad file format" << sendl;
                    return;
                }
            }
        }
    }
    // Push the last frame
    if(!vecCo.empty())
        m_vecFrame.push_back(coordTime);
    
    // Push the last stess vector
    if(!vecSt.empty())
        m_vecStress.push_back(stressTime);
}

void OglModelFromAbaqus::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowVisualModels())
    {
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());
        
        std::vector< Vector3 > points;
        Vector3 point;
        glColor4f (1.0f,0.35f,0.35f,1.0f);
        for (unsigned i=0; i<m_vecFrame[m_frameCount].first.size(); i++)
        {
            point = m_vecFrame[m_frameCount].first[i];
            points.push_back(point);
        }
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning
        vparams->drawTool()->drawSpheres(points, (float)m_radius.getValue(), Vec<4,float>(0.25f, 0.25f, 0.25f, 1));
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }
    vparams->drawTool()->setPolygonMode(0,false);
}

void OglModelFromAbaqus::handleEvent(sofa::core::objectmodel::Event *event)
{
    if ( /*simulation::AnimateEndEvent* ev =*/  dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
        double currTime = m_currentTime.getValue();
        if( (m_timeNextFrame<=currTime) && (m_frameCount < m_vecFrame.size()-1) )
        {
            m_frameCount++;
            m_timeNextFrame = m_vecFrame[m_frameCount].second;
        }
        if(m_updateDifference.getValue())
            updateDifference();
        
        if(m_updateStressVector.getValue())
            updateStressVector();
    }
}

void OglModelFromAbaqus::cleanup()
{
    m_frameCount = 0;
    m_timeNextFrame = m_vecFrame[0].second;
}

} // namespace visualmodel

} // namespace component

} // namespace sofa

