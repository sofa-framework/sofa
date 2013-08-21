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

#ifndef INPEXPORTERMASTER_H_
#define INPEXPORTERMASTER_H_

#include <sofa/helper/helper.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/component/forcefield/HexahedronFEMForceField.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API INPExporterMaster : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(INPExporterMaster,core::objectmodel::BaseObject);

private:
    sofa::core::objectmodel::BaseContext* context;
    sofa::core::behavior::OdeSolver* solver;
    
    unsigned int stepCounter;
    unsigned int maxStep;

    std::ofstream* outfile;

    void writeINPMaster();
    
    int nbFiles;

public:
    sofa::core::objectmodel::DataFileName inpFilename;
    Data<double> dt;
    Data<double> time;
    Data< defaulttype::Vec3d > gravity;
    Data<double> m_alpha;
    Data<double> m_beta;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

protected:
    INPExporterMaster();
    virtual ~INPExporterMaster();
public:
    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* INPEXPORTERMASTER_H_ */
