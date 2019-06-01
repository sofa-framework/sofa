/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_MERGEVISUALMODELS_H
#define SOFA_COMPONENT_ENGINE_MERGEVISUALMODELS_H

#include <SofaOpenglVisual/config.h>
#include <SofaOpenglVisual/OglModel.h>

#include <sofa/helper/vectorLinks.h>


namespace sofa
{

namespace component
{

namespace visualmodel
{

/**
 * This class merges several visual models.
 */
class SOFA_OPENGL_VISUAL_API MergeVisualModels : public OglModel
{
public:
    SOFA_CLASS(MergeVisualModels,OglModel);


    Data<unsigned int> d_nbInput; ///< number of input visual models to merge

    typedef core::objectmodel::SingleLink< MergeVisualModels, VisualModelImpl, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkVisualModel;
    helper::VectorLinks< LinkVisualModel, MergeVisualModels > vl_input;

    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;
    void parseFields ( const std::map<std::string,std::string*>& str ) override;
    void init() override;
    void reinit() override;

protected:
    MergeVisualModels();
    ~MergeVisualModels() override;

    void update();
};


} // namespace visualmodel

} // namespace component

} // namespace sofa

#endif
