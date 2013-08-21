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
#ifndef SOFA_COMPONENT_VISUALMODEL_OGLMODELFROMABAQUS_H
#define SOFA_COMPONENT_VISUALMODEL_OGLMODELFROMABAQUS_H

#include <vector>
#include <string>
#include <fstream>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/visual/VisualModel.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/helper.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

class SOFA_OPENGL_VISUAL_API OglModelFromAbaqus : public core::visual::VisualModel
{
public:
    SOFA_CLASS(OglModelFromAbaqus, VisualModel);

protected:
    sofa::core::objectmodel::BaseContext* context;
    sofa::core::behavior::BaseMechanicalState* mstate;
    
    void draw(const core::visual::VisualParams* vparams);

    virtual void updateDifference();
    virtual void updateStressVector();
    virtual void readRPTFile(const std::string filename);
    
    vector< std::pair< vector<defaulttype::Vec3f> /*Positions*/, double /*Time*/ > > m_vecFrame;
    vector< std::pair< vector<defaulttype::Vec3f> /*Stress in x,y,z*/, double /*Time*/ > > m_vecStress;
    unsigned int m_frameCount;
    double m_timeNextFrame;
    
    OglModelFromAbaqus();

    ~OglModelFromAbaqus();
    
public:
    void init();
    void cleanup();
    void bwdInit();
    
    void handleEvent(sofa::core::objectmodel::Event *);
    
    sofa::core::objectmodel::DataFileName filePath;
    Data< std::string > m_name;
    Data<float> m_radius;
    Data<double> m_currentTime;
    Data< defaulttype::Vec3Types::VecCoord > m_position;
    Data< vector<double> > m_difference;
    Data< vector< defaulttype::Vec3f > > m_stress; // Stress vectors (x,y,z) at each vertex for the current simulation time
    Data<bool> m_updateDifference;
    Data<bool> m_updateStressVector;
    
};

} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
