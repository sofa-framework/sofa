/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_VISUALACTION_H
#define SOFA_SIMULATION_TREE_VISUALACTION_H

#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/Visitor.h>
#include <sofa/core/VisualModel.h>
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{

namespace tree
{

class VisualVisitor : public Visitor
{
public:
    virtual void processVisualModel(GNode* node, core::VisualModel* vm) = 0;

    virtual Result processNodeTopDown(GNode* node)
    {
        for_each(this, node, node->visualModel, &VisualVisitor::processVisualModel);
        return RESULT_CONTINUE;
    }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "visual"; }
};

class VisualDrawVisitor : public VisualVisitor
{
public:
    enum Pass { Std, Transparent, Shadow };
    Pass pass;
    VisualDrawVisitor(Pass pass = Std)
        : pass(pass)
    {
    }
    virtual Result processNodeTopDown(GNode* node);
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

class VisualUpdateVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

class VisualInitTexturesVisitor : public VisualVisitor
{
public:
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

class VisualComputeBBoxVisitor : public VisualVisitor
{
public:
    double minBBox[3];
    double maxBBox[3];
    VisualComputeBBoxVisitor();
    virtual void processVisualModel(GNode*, core::VisualModel* vm);
};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
