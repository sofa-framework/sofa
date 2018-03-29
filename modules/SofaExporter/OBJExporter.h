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
/* OBJExporter.h
 *
 *  Created on: 9 sept. 2009
 *
 *  Contributors:
 *    - froy
 *    - damien.marchal@univ-lille1.fr
 *
 ************************************************************************************/

#ifndef OBJEXPORTER_H_
#define OBJEXPORTER_H_
#include "config.h"

#include <sofa/simulation/BaseSimulationExporter.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace _objexporter_
{

using sofa::simulation::BaseSimulationExporter ;
using sofa::core::objectmodel::Event ;
using sofa::core::objectmodel::Base ;

class SOFA_EXPORTER_API OBJExporter : public BaseSimulationExporter
{
public:
    SOFA_CLASS(OBJExporter, BaseSimulationExporter);

    virtual bool write() override ;
    bool writeOBJ();

    virtual void handleEvent(Event *event) override ;

protected:
    virtual ~OBJExporter();
};

}

using _objexporter_::OBJExporter ;

/// This is for compatibility with old code base in which OBJExporter where in sofa::component::misc.
namespace misc  { using _objexporter_::OBJExporter ; }

}

}

#endif /* OBJEXPORTER_H_ */
