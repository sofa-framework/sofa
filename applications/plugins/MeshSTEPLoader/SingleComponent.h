/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#ifndef SOFA_COMPONENT_ENGINE_SINGLECOMPONENT_H
#define SOFA_COMPONENT_ENGINE_SINGLECOMPONENT_H

#include <vector>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace sofa::defaulttype;

template <class DataTypes>
class SingleComponent : public sofa::core::DataEngine
{
public:
    SOFA_CLASS(SingleComponent,sofa::core::DataEngine);

    SingleComponent();

    virtual void init();
    virtual void reinit();
    virtual void update();

    template <class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::DataEngine::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const SingleComponent<DataTypes>*)
    {
        return "int";
    }

protected:
    void loadMesh();

public:
    Data<helper::vector<sofa::defaulttype::Vector3> > _positionsI;
    Data<helper::vector<sofa::defaulttype::Vector3> > _positionsO;
    Data<helper::vector<helper::fixed_array <unsigned int,2> > > _edgesI;
    Data<helper::vector<helper::fixed_array <unsigned int,2> > > _edgesO;
    Data<helper::vector<helper::fixed_array <unsigned int,3> > > _trianglesI;
    Data<helper::vector<helper::fixed_array <unsigned int,3> > > _trianglesO;
    Data<helper::vector<sofa::defaulttype::Vector3> > _normalsI;
    Data<helper::vector<sofa::defaulttype::Vector3> > _normalsO;
    Data<helper::vector<sofa::defaulttype::Vector2> > _uvI;
    Data<helper::vector<sofa::defaulttype::Vector2> > _uvO;
    Data<helper::vector<helper::fixed_array <unsigned int,3> > > _indicesComponents;
    Data<int> _numberShape;
};

}

}

}

#endif
