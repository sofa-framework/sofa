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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_DATAENGINE_H
#define SOFA_CORE_DATAENGINE_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <list>

namespace sofa
{

namespace core
{

/**
 *  \brief from a set of Data inputs computes a set of Data outputs
 *
 */
class SOFA_CORE_API DataEngine : public core::objectmodel::DDGNode, public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(DataEngine, core::objectmodel::BaseObject);
protected:
    /// Constructor
    DataEngine();

    /// Destructor. Do nothing
    virtual ~DataEngine();
public:
    /// Add a new input to this engine
    void addInput(objectmodel::BaseData* n);

    /// Remove an input from this engine
    void delInput(objectmodel::BaseData* n);

    /// Add a new output to this engine
    void addOutput(objectmodel::BaseData* n);

    /// Remove an output from this engine
    void delOutput(objectmodel::BaseData* n);

};

} // namespace core

} // namespace sofa

#endif
