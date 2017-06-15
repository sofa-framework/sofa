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
/*
 * OBJExporter.h
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#ifndef OBJEXPORTER_H_
#define OBJEXPORTER_H_
#include "config.h"

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API OBJExporter : public core::objectmodel::BaseObject
{
public:
	SOFA_CLASS(OBJExporter, core::objectmodel::BaseObject);

private:
    unsigned int stepCounter;
    sofa::core::objectmodel::BaseContext* context;
    unsigned int maxStep;

public:
    sofa::core::objectmodel::DataFileName objFilename;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;
    bool  activateExport;
protected:
    OBJExporter();
    virtual ~OBJExporter();
public:
    void init();
    void cleanup();
    void bwdInit();
	void handleEvent(sofa::core::objectmodel::Event *);
	void writeOBJ(); 
};

}

}

}

#endif /* OBJEXPORTER_H_ */
