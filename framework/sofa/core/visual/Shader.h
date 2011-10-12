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
//
// C++ Interface: Shader
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_CORE_VISUAL_SHADER_H
#define SOFA_CORE_VISUAL_SHADER_H


namespace sofa
{

namespace core
{

namespace visual
{

/**
 *  \brief A basic interface to define a Shader for different system (OpenGL, DirectX, ...).
 *
 *
 *
 */
class Shader : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(Shader, objectmodel::BaseObject);
protected:
    /// Destructor
    virtual ~Shader() { };
public:
    /// Start the shader
    virtual void start() = 0;
    /// Stop the shader
    virtual void stop() = 0;
    ///Tells if it must be activated automatically(value false : the visitor will switch the shader)
    ///or manually (value true : useful when another component wants to use it for itself only)
    virtual bool isActive() = 0;
};

/**
 *  \brief A basic interface to define an element to be used with a Shader.
 *
 *
 *
 */
class ShaderElement: public virtual objectmodel::BaseObject
{
public:
    /// Destructor
    virtual ~ShaderElement() { };
};

} // namespace visual

} // namespace core

} // namespace sofa

#endif //SOFA_CORE_VISUAL_SHADER_H
