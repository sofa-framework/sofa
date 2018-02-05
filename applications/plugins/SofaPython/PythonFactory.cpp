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
#include "PythonFactory.h"


namespace sofa
{

PyObject* PythonFactory::s_sofaPythonModule = NULL;

PythonFactory::PythonBoundTypes PythonFactory::s_boundComponents[PythonFactory::NB_LISTS];
PythonFactory::PythonBoundTypes PythonFactory::s_boundData;

/// check for the corresponding type in the given list
PyObject* PythonFactory::toPython( const PythonBoundTypes& list, sofa::core::objectmodel::Base* obj, PyTypeObject* pyTypeObject  )
{
    for(PythonBoundTypes::const_reverse_iterator it=list.rbegin(),itend=list.rend();it!=itend;++it)
        if( (*it)->canCast( obj ) ) return BuildPySPtr<sofa::core::objectmodel::Base>(obj,(*it)->pyTypeObject);

    return BuildPySPtr<sofa::core::objectmodel::Base>(obj,pyTypeObject);
}


/// to convert a sofa::core::objectmodel::Base-inherited object to its corresponding pyObject
/// could be improve for known types, but since it is only setup...
PyObject* PythonFactory::toPython(sofa::core::objectmodel::Base* obj)
{
    if( obj->toBaseObject() )
    {
        if( obj->toBaseLoader() ) return toPython( obj->toBaseLoader() );

        if( obj->toTopology() )
        {
            if( obj->toBaseMeshTopology() ) return toPython( obj->toBaseMeshTopology() );
            return toPython( obj->toTopology() );
        }

        if( obj->toBaseTopologyObject() ) return toPython( obj->toBaseTopologyObject() );

        if( obj->toVisualModel()) return toPython( obj->toVisualModel() );

        if( obj->toBaseState() )
        {
            if (obj->toBaseMechanicalState()) return toPython( obj->toBaseMechanicalState() );

            return toPython( obj->toBaseState() );
        }

        if (obj->toBaseMapping()) return toPython( obj->toBaseMapping() );

        if (obj->toDataEngine()) return toPython( obj->toDataEngine() );

        return toPython( obj->toBaseObject() );
    }
    else if( obj->toBaseContext() ) return toPython( obj->toBaseContext() );


    return toPython( s_boundComponents[Base], obj, &SP_SOFAPYTYPEOBJECT(Base) );
}

PyObject* PythonFactory::toPython(sofa::core::objectmodel::BaseObject* obj)
{
    if( obj->toBaseLoader() ) return toPython( obj->toBaseLoader() );

    if( obj->toTopology() )
    {
        if( obj->toBaseMeshTopology() ) return toPython( obj->toBaseMeshTopology() );
        return toPython( obj->toTopology() );
    }

    if( obj->toBaseTopologyObject() ) return toPython( obj->toBaseTopologyObject() );

    if( obj->toVisualModel()) return toPython( obj->toVisualModel() );

    if( obj->toBaseState() )
    {
        if (obj->toBaseMechanicalState()) return toPython( obj->toBaseMechanicalState() );

        return toPython( obj->toBaseState() );
    }

    if (obj->toBaseMapping()) return toPython( obj->toBaseMapping() );

    if (obj->toDataEngine()) return toPython( obj->toDataEngine() );

    return toPython( s_boundComponents[BaseObject], obj, &SP_SOFAPYTYPEOBJECT(BaseObject) );
}

/// to convert a BaseContext-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::objectmodel::BaseContext* obj)
{
    return toPython( s_boundComponents[BaseContext], obj, &SP_SOFAPYTYPEOBJECT(BaseContext) );
}

/// to convert a BaseLoader-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::loader::BaseLoader* obj)
{
    return toPython( s_boundComponents[BaseLoader], obj, &SP_SOFAPYTYPEOBJECT(BaseLoader) );
}

/// to convert a Topology-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::topology::Topology* obj)
{
    if( obj->toBaseMeshTopology() ) return toPython( obj->toBaseMeshTopology() );
    return toPython( s_boundComponents[Topology], obj, &SP_SOFAPYTYPEOBJECT(Topology) );
}

/// to convert a BaseMeshTopology-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::topology::BaseMeshTopology* obj)
{
    return toPython( s_boundComponents[BaseMeshTopology], obj, &SP_SOFAPYTYPEOBJECT(BaseMeshTopology) );
}

/// to convert a BaseTopologyObject-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::topology::BaseTopologyObject* obj)
{
    return toPython( s_boundComponents[BaseTopologyObject], obj, &SP_SOFAPYTYPEOBJECT(BaseTopologyObject) );
}

/// to convert a VisualModel-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::visual::VisualModel* obj)
{
    return toPython( s_boundComponents[VisualModel], obj, &SP_SOFAPYTYPEOBJECT(VisualModel) );
}

/// to convert a BaseState-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::BaseState* obj)
{
    if (obj->toBaseMechanicalState()) return toPython( obj->toBaseMechanicalState() );

    return toPython( s_boundComponents[BaseState], obj, &SP_SOFAPYTYPEOBJECT(BaseState) );
}

/// to convert a BaseMechanicalState-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::behavior::BaseMechanicalState* obj)
{
    return toPython( s_boundComponents[BaseMechanicalState], obj, &SP_SOFAPYTYPEOBJECT(BaseMechanicalState) );
}

/// to convert a BaseMapping-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::BaseMapping* obj)
{
    return toPython( s_boundComponents[BaseMapping], obj, &SP_SOFAPYTYPEOBJECT(BaseMapping) );
}

/// to convert a DataEngine-inherited object to its corresponding pyObject
 PyObject* PythonFactory::toPython(sofa::core::DataEngine* obj)
{
    return toPython( s_boundComponents[DataEngine], obj, &SP_SOFAPYTYPEOBJECT(DataEngine) );
}

/// to convert a sofa::core::objectmodel::BaseData to its corresponding pyObject
/// returns NULL if the data does not correpond to any special type
 PyObject* PythonFactory::toPython(sofa::core::objectmodel::BaseData* data)
{
    for(PythonBoundTypes::const_reverse_iterator it=s_boundData.rbegin(),itend=s_boundData.rend();it!=itend;++it)
    {
        if( (*it)->canCast( data ) )
            return BuildPyPtr<sofa::core::objectmodel::BaseData>(data,(*it)->pyTypeObject,false);
    }
    return NULL;
}

} // namespace sofa
