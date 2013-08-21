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

#include "INPExporterMaster.h"

#include <sstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/ExportINPVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>


namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(INPExporterMaster)

int INPExporterMasterClass = core::RegisterObject("Read State vectors from file")
        .add< INPExporterMaster >();

INPExporterMaster::INPExporterMaster()
    : stepCounter(0)
    , inpFilename( initData(&inpFilename, "filename", "output INP file name"))
    , dt( initData(&dt, "dt", "time step"))
    , time( initData(&time, "time", "current time"))
    , gravity( initData(&gravity, "gravity", "gravity"))
    , exportAtEnd( initData(&exportAtEnd, (bool)false, "exportAtEnd", "export file when the simulation is finished"))
{
    this->f_listening.setValue(true);
}

INPExporterMaster::~INPExporterMaster()
{
    if (outfile)
        delete outfile;
}

void INPExporterMaster::init()
{
    context = this->getContext();
    context->get(solver, sofa::core::objectmodel::BaseContext::SearchDown); // Ode Solver
    
    sofa::core::objectmodel::BaseData* deltaT = NULL;
    sofa::core::objectmodel::BaseData* timeT = NULL;
    sofa::core::objectmodel::BaseData* graviT = NULL;
    sofa::core::objectmodel::BaseData* alphaT = NULL;
    sofa::core::objectmodel::BaseData* betaT = NULL;
    
    if(context)
    {
        deltaT = context->findField("dt");
        timeT = context->findField("time");
        graviT = context->findField("gravity");
        
        dt.setParent(deltaT);
        time.setParent(timeT);
        gravity.setParent(graviT);
    }
    
    if(solver)
    {
        alphaT = solver->findField("rayleighMass");
        betaT = solver->findField("rayleighStiffness");
        
        if(!alphaT || !betaT)
            serr << "Error: unsuitable solver" << sendl;
        
        m_alpha.setParent(alphaT);
        m_beta.setParent(betaT);
    }
    else
    {
        serr << "Error: no solver" << sendl;
        return;
    }
    
    nbFiles = 0;

}


void INPExporterMaster::writeINPMaster()
{
    std::string filename = inpFilename.getFullPath();
    std::string fname = filename;
    if (maxStep)
    {
        std::ostringstream oss;
        oss.width(5);
        oss.fill('0');
        oss << nbFiles;
        filename += oss.str();
    }
    filename += ".inp";

    outfile = new std::ofstream(filename.c_str());
    if( !outfile->is_open() )
    {
        serr << "Error creating file " << filename << sendl;
        delete outfile;
        outfile = NULL;
        return;
    }
    
    helper::ReadAccessor<Data< double > > deltaT = dt;
    helper::ReadAccessor<Data< double > > timeT = time;
    helper::ReadAccessor<Data< defaulttype::Vec3d > > graviT = gravity;
    helper::ReadAccessor<Data< double > > alphaFact = m_alpha;
    helper::ReadAccessor<Data< double > > betaFact = m_beta;
    vector< std::string > nameT;
    vector< defaulttype::Vec3Types::VecCoord > positionT;
    vector< double > densiT;
    vector< vector< sofa::component::topology::Tetrahedron > > tetrahedraT;
    vector< vector< sofa::component::topology::Hexahedron > > hexahedraT;
    vector< vector< unsigned int > > fixedPointT;
    vector< double > youngModulusT;
    vector< double > poissonRatioT;
    
    /* Header */
    *outfile << "*Heading" << std::endl;
    *outfile << "Analysis for " << fname << " model" << std::endl;
    *outfile << "** Job name: " << fname << " Model name: Model-1" << std::endl;
    *outfile << "** Generated by: Sofa version 1.0 RC 1" << std::endl;
    *outfile << "**Preprint, " << "echo=NO, "
                              << "model=NO, "
                              << "history=NO, "
                              << "contact=NO" 
                              << std::endl;
                              
    sofa::simulation::ExportINPVisitor exportINP(core::ExecParams::defaultInstance(), &nameT, &positionT, &densiT, &tetrahedraT, &hexahedraT, &fixedPointT, &youngModulusT, &poissonRatioT);
    context->executeVisitor(&exportINP);
    
    unsigned int nbrNode = positionT.size(); // Number of nodes containing an INPExporter component in the scene
    vector<unsigned int> nbrElem_array; // Array containing the number of elements for each shape
    
    outfile->precision(7);
    outfile->flags ( std::ios::right | std::ios::showpoint );
    
    /* Parts */
    *outfile << "**" << std::endl;
    *outfile << "** PARTS" << std::endl;
    *outfile << "**" << std::endl;
    
    for(unsigned int i=0;i<nbrNode;i++)
    {
        std::string name = nameT[i];
        std::stringstream ss;
        ss << "_" << i;
        name += ss.str();
        
        *outfile << "*part, name=" << name << std::endl;
        
        /* nodes */
        *outfile << "*Node, Nset=" << name << std::endl;
        for(unsigned int j=0;j<positionT[i].size();j++)
        {
            outfile->width (5);
            // i+1 because facet indices start at 1 in the file
            *outfile << j+1 << ',';
            outfile->width (15);
            *outfile << positionT[i][j][0] << ',';
            outfile->width (15);
            *outfile << positionT[i][j][1] << ',';
            outfile->width (15);
            *outfile << positionT[i][j][2] << std::endl;
        }
        
        /* elements */
        unsigned int nbrElem = 0;
        *outfile << "*Element, Elset=" << name << ", type=";
        if( !hexahedraT[i].empty() )
        {
            *outfile << "C3D8" << std::endl; // Continuum -- 3Dimension -- 8nodes
            for(unsigned int j=0;j<hexahedraT[i].size();j++)
            {
                outfile->width (4);
                *outfile << j+1 << ',';
                for(unsigned int k=0;k<hexahedraT[i][0].size()-1;k++)
                {
                    outfile->width (5);
                    *outfile << hexahedraT[i][j][k]+1 << ',';
                }
                outfile->width (5);
                *outfile << hexahedraT[i][j][ hexahedraT[i][0].size()-1 ]+1 << std::endl;
                nbrElem++;
            }
            nbrElem_array.push_back(nbrElem);
        }
        else
        {
            *outfile << "C3D4" << std::endl; // Continuum -- 3Dimension -- 4nodes
            for(unsigned int j=0;j<tetrahedraT[i].size();j++)
            {
                outfile->width (4);
                *outfile << j+1 << ',';
                for(unsigned int k=0;k<tetrahedraT[i][0].size()-1;k++)
                {
                    outfile->width (5);
                    *outfile << tetrahedraT[i][j][k]+1 << ',';
                }
                outfile->width (5);
                *outfile << tetrahedraT[i][j][ tetrahedraT[i][0].size()-1 ]+1 << std::endl;
                nbrElem++;
            }
            nbrElem_array.push_back(nbrElem);
        }
        /* Solid section */
        *outfile << "** Section: Section-" << name << std::endl;
        *outfile << "*Solid Section, Elset=" << name << ", material=Material-" << name << std::endl;
        *outfile << "*End part" << std::endl;
    
        /* Material */
        *outfile << "**" << std::endl;
        *outfile << "** MATERIALS" << std::endl;
        *outfile << "**" << std::endl;
        *outfile << "*Material, name=Material-" << name << std::endl;
            // Density
        *outfile << "*Density" << std::endl;
        outfile->flags ( std::ios::right | std::ios::showpoint );
        outfile->width (7);
        *outfile << densiT[i] << std::endl;
            // Elastic
        if( youngModulusT[i]>0 && poissonRatioT[i]>0 )
        {
            *outfile << "*Elastic" << std::endl;
            outfile->width (7);
            *outfile << youngModulusT[i] << ','; // Young's modulus
            outfile->width (7);
            *outfile << poissonRatioT[i] << std::endl; // Poisson's ratio
        }
        else
        {
            *outfile << "*Elastic" << std::endl;
            *outfile << "200000, 0.0001" << std::endl;
        }
        // Damping
        *outfile << "*Damping, Alpha=" << alphaFact << ", Beta=" << betaFact << std::endl;
        
    }
    
    /* Assembly */
    *outfile << "**" << std::endl;
    *outfile << "** ASSEMBLY" << std::endl;
    *outfile << "**" << std::endl;
    *outfile << "*Assembly, name=Assembly" << std::endl;
    
    for(unsigned int i=0;i<nbrNode;i++)
    {
        std::string name = nameT[i];
        std::stringstream ss;
        ss << "_" << i;
        name += ss.str();
        
        /* Instanciation */
        *outfile << "**" << std::endl;
        *outfile << "*Instance, name=" << name << ", part=" << name << std::endl;
        *outfile << "*End Instance" << std::endl;
        *outfile << "**" << std::endl;
        
        *outfile << "*Nset, nset=load-" << name << ", internal, instance=" << name  << ", generate" << std::endl;
        *outfile << "   1,";
        outfile->width (5);
        *outfile << positionT[i].size();
        *outfile << ",    1" << std::endl;
        *outfile << "*Elset, elset=load-" << name << ", internal, instance=" << name  << ", generate" << std::endl;
        *outfile << "   1,";
        outfile->width (5);
        *outfile << nbrElem_array[i];
        *outfile << ",    1" << std::endl;
        
        if(youngModulusT[i]<=0 || poissonRatioT[i]<=0)
        {
            *outfile << "*Nset, nset=bound-" << name << ", internal, instance=" << name  << ", generate" << std::endl;
            *outfile << "   1,";
            outfile->width (5);
            *outfile << positionT[i].size();
            *outfile << ",    1" << std::endl;
        }
        else if(!fixedPointT[i].empty())
        {
            // Set of bounded node/element
            *outfile << "*Nset, nset=bound-" << name << ", internal, instance=" << name  << std::endl;
            for(unsigned int j = 0;j<fixedPointT[i].size()-1;j++)
            {
                outfile->width (4);
                *outfile << fixedPointT[i][j]+1 << ',';
            }
            outfile->width (4);
            *outfile << fixedPointT[i][fixedPointT[i].size()-1]+1 << std::endl;
        }
    }
    
    *outfile << "*End Assembly" << std::endl;
    
    /* Boundary conditions */
    *outfile << "**" << std::endl;
    *outfile << "** BOUNDARY CONDITIONS" << std::endl;
    *outfile << "**" << std::endl;
    for(unsigned int i=0;i<nbrNode;i++)
    {
        std::string name = nameT[i];
        std::stringstream ss;
        ss << "_" << i;
        name += ss.str();
        
        if(!fixedPointT[i].empty() || youngModulusT[i]<=0 || poissonRatioT[i]<=0)
        {
            *outfile << "*Boundary" << std::endl;;
            *outfile << "bound-" << name << ", PINNED" << std::endl;
        }
    }
    
    /* Step */
    *outfile << "**" << std::endl;
    *outfile << "** STEP" << std::endl;
    *outfile << "**" << std::endl;
    *outfile << "*Step, nlgeom , inc=" << (int)(timeT/deltaT) << std::endl;
        // Response Type
    *outfile << "*Dynamic" << std::endl; // stabilize with damping factor and accuracy tolerance in Abaqus
    *outfile << deltaT << ", " << timeT << std::endl;
    *outfile << "**" << std::endl;
        // Loading
    *outfile << "*Dload" << std::endl; // Distributed load
    std::stringstream strGrav;
    defaulttype::Vec3d gt;
    strGrav << graviT;
    strGrav >> gt[0] >> gt[1] >> gt[2];
    for(unsigned int i=0;i<nbrNode;i++)
    {
        std::string name = nameT[i];
        std::stringstream ss;
        ss << "_" << i;
        name += ss.str();
        
        *outfile << "load-" << name << ", GRAV, ";
        if(gt[0]!=0)
        {
            *outfile << std::abs(gt[0]) << ", ";
            if(gt[0]>0)
            {
                *outfile << "-1, 0, 0" << std::endl;
            }
            else
            {
                *outfile << "1, 0, 0" << std::endl;
            }
        }
        else if(gt[1]!=0)
        {
            *outfile << std::abs(gt[1]) << ", ";
            if(gt[1]>0)
            {
                *outfile << "0, 1, 0" << std::endl;
            }
            else
            {
                *outfile << "0, -1, 0" << std::endl;
            }
        }
        else if(gt[2]!=0)
        {
            *outfile << std::abs(gt[2]) << ", ";
            if(gt[2]>0)
            {
                *outfile << "0, 0, -1" << std::endl;
            }
            else
            {
                *outfile << "0, 0, 1" << std::endl;
            }
        }
    }
    *outfile << "**" << std::endl;
        // Boundary Conditions
    //*outfile << "" << std::endl;
        // Output Control
    *outfile << "*Restart, write, frequency=0" << std::endl;
    *outfile << "**" << std::endl;
    *outfile << "*Output, field, variable=PRESELECT" << std::endl;
    *outfile << "**" << std::endl;
    *outfile << "*Output, history, variable=PRESELECT" << std::endl;
    *outfile << "**" << std::endl;
        // Contact
    //*outfile << "" << std::endl;
        // Auxiliary Controls
    //*outfile << "" << std::endl;
        // Element and Surface Removal/Reactivation
    //*outfile << "" << std::endl;
        // Co-simulation
    //*outfile << "" << std::endl;
    
    *outfile << "*End Step" << std::endl;
    
    outfile->close();
    sout << filename << " written" << sendl;
    nbFiles++;
}

void INPExporterMaster::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent* ev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent*>(event))
    {
        std::cout << "key pressed " << std::endl;
        switch(ev->getKey())
        {

        case 'E':
        case 'e':
            writeINPMaster();
            break;

        case 'F':
        case 'f':
            break;
        }
    }
}

void INPExporterMaster::cleanup()
{
    if (exportAtEnd.getValue())
        writeINPMaster();
}


} // namespace misc

} // namespace component

} // namespace sofa

