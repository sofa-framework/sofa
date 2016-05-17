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

#include <boost/type_traits/is_base_of.hpp>


namespace sofa
{



/// This is the macro to call to bind a new type inherited from sofa::core::objectmodel::Base
#define SP_ADD_CLASS_IN_FACTORY( PYTHONNAME, CPPCLASSNAME ) {\
    SP_ADD_CLASS( PythonFactory::s_sofaPythonModule, PYTHONNAME )  \
    PythonFactory::add<CPPCLASSNAME>( &SP_SOFAPYTYPEOBJECT(PYTHONNAME) ); \
    }


/// @internal To convert a sofa::core::objectmodel::Base in its corresponding pyObject
// retourne automatiquement le type Python de plus haut niveau possible,
// en fonction du type de l'objet Cpp (spécifiquement bindé)
// afin de permettre l'utilisation de fonctions des sous-classes de Base
class SOFA_SOFAPYTHON_API PythonFactory
{

public:
    static PyObject *s_sofaPythonModule;

protected:

    /// a undefined bound type
    /// nothing can be casted in it
    struct BasePythonBoundType
    {
        BasePythonBoundType(PyTypeObject*pyTypeObject):pyTypeObject(pyTypeObject){}
        virtual bool canCast(sofa::core::objectmodel::Base*) const { return false; }
        PyTypeObject* pyTypeObject;
    };

    /// a know bound type (built when calling 'add')
    /// cast is checked by dynamic_cast (this is similar to the previous implementation, but could be improved)
    template <class T>
    struct PythonBoundType : public BasePythonBoundType
    {
        PythonBoundType(PyTypeObject*pyTypeObject):BasePythonBoundType(pyTypeObject){}
        virtual bool canCast(sofa::core::objectmodel::Base* obj) const { return dynamic_cast<T*>(obj); }
    };

    /// a list of Abstract classes that can be cheaply deduced from Base* (by static_cast)
    /// this limits checking the right cast on a limited number of types
    /// Note this list is built from actual needs, but can be easily extended to any types that Base* can be statically casted from.
    enum{Base=0,BaseObject,BaseLoader,Topology,BaseMeshTopology,VisualModel,BaseState,BaseMechanicalState,BaseMapping,DataEngine,BaseContext,NB_LISTS};
    typedef std::list< BasePythonBoundType* > PythonBoundTypes;
    /// a list of types for each sub-classes (prefiltering types not to have to check casting with any of them)
    static PythonBoundTypes s_boundTypes[NB_LISTS];


    /// check for the corresponding type in the given list
    static PyObject* toPython( const PythonBoundTypes& list, sofa::core::objectmodel::Base* obj, PyTypeObject* pyTypeObject  )
    {
//        std::cerr<<"toPython "<<obj->getClassName()<<std::endl;

        for(PythonBoundTypes::const_reverse_iterator it=list.rbegin(),itend=list.rend();it!=itend;++it)
            if( (*it)->canCast( obj ) ) return BuildPySPtr<sofa::core::objectmodel::Base>(obj,(*it)->pyTypeObject);

        return BuildPySPtr<sofa::core::objectmodel::Base>(obj,pyTypeObject);
    }


public:

    /// add a new sofa::core::objectmodel::Base-inherited type with it corresponding pyTypeObject to the Factory
    template<class T>
    static void add( PyTypeObject* pyTypeObject )
    {
//        std::cerr<<"ADD "<<T::template className<T>()<<std::endl;

        PythonBoundType<T>* t = new PythonBoundType<T>(pyTypeObject);

        if( boost::is_base_of<sofa::core::objectmodel::BaseObject,T>::value )
        {
            if( boost::is_base_of<sofa::core::loader::BaseLoader, T>::value )
                return s_boundTypes[BaseLoader].push_back( t );

            if( boost::is_base_of<sofa::core::topology::Topology, T>::value )
            {
                if( boost::is_base_of<sofa::core::topology::BaseMeshTopology, T>::value )
                    return s_boundTypes[BaseMeshTopology].push_back( t );
                return s_boundTypes[Topology].push_back( t );
            }

            if( boost::is_base_of<sofa::core::visual::VisualModel, T>::value )
                return s_boundTypes[VisualModel].push_back( t );

            if( boost::is_base_of<sofa::core::BaseState, T>::value )
            {
                if( boost::is_base_of<sofa::core::behavior::BaseMechanicalState, T>::value )
                    return s_boundTypes[BaseMechanicalState].push_back( t );

                return s_boundTypes[BaseState].push_back( t );
            }


            if( boost::is_base_of<sofa::core::BaseMapping, T>::value )
                return s_boundTypes[BaseMapping].push_back( t );

            if( boost::is_base_of<sofa::core::DataEngine, T>::value )
                return s_boundTypes[DataEngine].push_back( t );

            return s_boundTypes[BaseObject].push_back( t );
        }
        else if( boost::is_base_of<sofa::core::objectmodel::BaseContext, T>::value )
            return s_boundTypes[BaseContext].push_back( t );

        return s_boundTypes[Base].push_back( t );
    }



    /// to convert a sofa::core::objectmodel::Base-inherited object to its corresponding pyObject
    static PyObject* toPython(sofa::core::objectmodel::Base* obj)
    {
//        std::cerr<<"toPython0 "<<obj->getClassName()<<std::endl;

        if( obj->toBaseObject() )
        {
            if( obj->toBaseLoader() )
                return toPython( s_boundTypes[BaseLoader], obj, &SP_SOFAPYTYPEOBJECT(BaseLoader) );

            if( obj->toTopology() )
            {
                if( obj->toBaseMeshTopology() )
                    return toPython( s_boundTypes[BaseMeshTopology], obj, &SP_SOFAPYTYPEOBJECT(BaseMeshTopology) );
                return toPython( s_boundTypes[Topology], obj, &SP_SOFAPYTYPEOBJECT(Topology) );
            }

            if( obj->toVisualModel())
                return toPython( s_boundTypes[VisualModel], obj, &SP_SOFAPYTYPEOBJECT(VisualModel) );

            if( obj->toBaseState() )
            {
                if (obj->toBaseMechanicalState())
                    return toPython( s_boundTypes[BaseMechanicalState], obj, &SP_SOFAPYTYPEOBJECT(BaseMechanicalState) );

                return toPython( s_boundTypes[BaseState], obj, &SP_SOFAPYTYPEOBJECT(BaseState) );
            }

            if (obj->toBaseMapping())
                return toPython( s_boundTypes[BaseMapping], obj, &SP_SOFAPYTYPEOBJECT(BaseMapping) );

            if (obj->toDataEngine())
                return toPython( s_boundTypes[DataEngine], obj, &SP_SOFAPYTYPEOBJECT(DataEngine) );

            return toPython( s_boundTypes[BaseObject], obj, &SP_SOFAPYTYPEOBJECT(BaseObject) );
        }
        else if( obj->toBaseContext() )
            return toPython( s_boundTypes[BaseContext], obj, &SP_SOFAPYTYPEOBJECT(BaseContext) );

        return toPython( s_boundTypes[Base], obj, &SP_SOFAPYTYPEOBJECT(Base) );
    }

    /// get singleton
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
