/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef __PYTHON_FACTORY_H
#define __PYTHON_FACTORY_H

#include "PythonMacros.h"
#include <SofaPython/config.h>


#include <SofaPython/Binding_Base.h>
#include <SofaPython/Binding_BaseObject.h>
#include <SofaPython/Binding_BaseLoader.h>
#include <SofaPython/Binding_Topology.h>
#include <SofaPython/Binding_BaseMeshTopology.h>
#include <SofaPython/Binding_VisualModel.h>
#include <SofaPython/Binding_BaseState.h>
#include <SofaPython/Binding_BaseMechanicalState.h>
#include <SofaPython/Binding_BaseMapping.h>
#include <SofaPython/Binding_DataEngine.h>
#include <SofaPython/Binding_BaseContext.h>


namespace sofa
{


#define SP_ADD_CLASS_IN_FACTORY( PYTHONNAME, CPPCLASSNAME ) {\
    SP_ADD_CLASS( PythonFactory::s_sofaPythonModule, PYTHONNAME )  \
    PythonFactory::add<CPPCLASSNAME>( &SP_SOFAPYTYPEOBJECT(PYTHONNAME) ); \
    }



// TODO
// - doc
// - several typed lists (+plrs fonctions toPython specializées)
//Base
//BaseObject
//    BaseLoader
//    Topology
//        BaseMeshTopology
//    VisualModel
//    BaseState
//        BaseMechanicalState
//    BaseMapping
//    DataEngine
//BaseContext
class SOFA_SOFAPYTHON_API PythonFactory
{

public:
    static PyObject *s_sofaPythonModule;

protected:

    struct BasePythonBoundType
    {
        BasePythonBoundType(PyTypeObject*pyTypeObject):pyTypeObject(pyTypeObject){}
        virtual bool canCast(sofa::core::objectmodel::Base*) const { return false; }
        PyTypeObject* pyTypeObject;
    };

    template <class T>
    struct PythonBoundType : public BasePythonBoundType
    {
        PythonBoundType(PyTypeObject*pyTypeObject):BasePythonBoundType(pyTypeObject){}
        virtual bool canCast(sofa::core::objectmodel::Base* obj) const { return dynamic_cast<T*>(obj); }
    };

    enum{Base=0,BaseObject,Topology,BaseMeshTopology,VisualModel,BaseState,BaseMechanicalState,BaseMapping,DataEngine,BaseContext,NB_LISTS};
    typedef std::list< BasePythonBoundType* > PythonBoundTypes;
    static PythonBoundTypes s_boundTypes[NB_LISTS];


    static PyObject* toPython( const PythonBoundTypes& list, sofa::core::objectmodel::Base* obj, PyTypeObject* pyTypeObject  )
    {
        for(auto it=list.rbegin(),itend=list.rend();it!=itend;++it)
        {
            if( (*it)->canCast( obj ) ) return BuildPySPtr<sofa::core::objectmodel::Base>(obj,(*it)->pyTypeObject);
        }

        return BuildPySPtr<sofa::core::objectmodel::Base>(obj,pyTypeObject);
    }


public:

    template<class T>
    static void add( PyTypeObject* pyTypeObject )
    {

//           if( std::is_base_of<sofa::core::objectmodel::BaseObject,T>() )
//        {
//            if( obj->toBaseLoader() )
//            {
//                return toPython( obj->toBaseLoader() );
//            }

//            if( obj->toTopology() )
//            {
//                if( obj->toBaseMeshTopology() )
//                {
//                    return toPython( obj->toBaseMeshTopology() );
//                }
//                return toPython( obj->toTopology() );
//            }

//            if( obj->toVisualModel())
//            {
//                return toPython( obj->toVisualModel() );
//            }

//            if( obj->toBaseState() )
//            {
//                if (obj->toBaseMechanicalState())
//                {
//                    return toPython( obj->toBaseMechanicalState() );
//                }

//                return toPython( obj->toBaseState() );
//            }


//            if (obj->toBaseMapping())
//            {
//                return toPython( obj->toBaseMapping() );
//            }

//            if (obj->toDataEngine())
//                return toPython( obj->toDataEngine() );


//        PythonBoundType<sofa::core::objectmodel::BaseObject>* t = new PythonBoundType<sofa::core::objectmodel::BaseObject>(pyTypeObject);
//        s_boundTypes[BaseObject].push_back( t );
//        return;

//        }
//        else if( obj->toBaseContext() )
//        {
//            return toPython( obj->toBaseContext() );
//        }


        PythonBoundType<sofa::core::objectmodel::Base>* t = new PythonBoundType<sofa::core::objectmodel::Base>(pyTypeObject);
        s_boundTypes[Base].push_back( t );

    }



    // to convert
    static PyObject* toPython(sofa::core::objectmodel::Base* obj)
    {
//        if( obj->toBaseObject() )
//        {
//            if( obj->toBaseLoader() )
//            {
//                return toPython( obj->toBaseLoader() );
//            }

//            if( obj->toTopology() )
//            {
//                if( obj->toBaseMeshTopology() )
//                {
//                    return toPython( obj->toBaseMeshTopology() );
//                }
//                return toPython( obj->toTopology() );
//            }

//            if( obj->toVisualModel())
//            {
//                return toPython( obj->toVisualModel() );
//            }

//            if( obj->toBaseState() )
//            {
//                if (obj->toBaseMechanicalState())
//                {
//                    return toPython( obj->toBaseMechanicalState() );
//                }

//                return toPython( obj->toBaseState() );
//            }


//            if (obj->toBaseMapping())
//            {
//                return toPython( obj->toBaseMapping() );
//            }

//            if (obj->toDataEngine())
//                return toPython( obj->toDataEngine() );

//            return toPython( s_boundTypes[BaseObject], obj->toBaseObject(), &SP_SOFAPYTYPEOBJECT(BaseObject) );
//        }
//        else if( obj->toBaseContext() )
//        {
//            return toPython( obj->toBaseContext() );
//        }

        return toPython( s_boundTypes[Base], obj, &SP_SOFAPYTYPEOBJECT(Base) );
    }



    // get singleton
    PythonFactory& getInstance()
    {
        static PythonFactory singleton;
        return singleton;
    }

private:

    // singleton
    PythonFactory();
    PythonFactory(const PythonFactory&);


};



} // namespace sofa

#endif
