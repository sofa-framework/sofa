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
#ifndef SOFA_CORE_DATAENGINE_H
#define SOFA_CORE_DATAENGINE_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/core.h>
#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <list>
#include "DataTracker.h"

namespace sofa
{

namespace core
{

/**
 *  \brief from a set of Data inputs computes a set of Data outputs
 *
 * Implementation good rules:
 *
 * void init()
 * {
 *    addInput // indicate all inputs
 *    addOutput // indicate all outputs
 *    setDirtyValue(); // the engine must start dirty (of course, no output are up-to-date)
 * }
 *
 * // optional (called each time a data is modified in the gui)
 * // it is not always desired
 * void reinit()
 * {
 *    update();
 * }
 *
 * void update()
 * {
 *      // FIRST all inputs must be updated
 *      // can be done by Data::getValue, ReadAccessor, Data::updateIfDirty, DataEngine::updateAllInputsIfDirty
 *
 *      // must be called AFTER updating all inputs, otherwise a modified input will set the engine to dirty again.
 *      // must be called BEFORE read access to an output, otherwise read-accessing the output will call update
 *      cleanDirty();
 *
 *      // FINALLY access and set outputs
 *      // Note that a write-only access has better performance and is enough in 99% engines   Data::beginWriteOnly, WriteOnlyAccessor
 *      // A read access is possible, in that case, be careful the cleanDirty is called before the read-access, otherwise it can call an DataEngine::update itself.  Data::beginEdit, WriteAccessor
 * }
 *
 */
class SOFA_CORE_API DataEngine : public core::DataTrackerDDGNode, public virtual core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(DataEngine, core::objectmodel::BaseObject);
    SOFA_BASE_CAST_IMPLEMENTATION(DataEngine)
protected:
    /// Constructor
    DataEngine();

    /// Destructor. Do nothing
    virtual ~DataEngine();

private:
	DataEngine(const DataEngine& n) ;
	DataEngine& operator=(const DataEngine& n) ;
	
public:
    /// Add a new input to this engine
    void addInput(objectmodel::BaseData* n);

    /// Add a new output to this engine
    void addOutput(objectmodel::BaseData* n);

    // The methods below must be redefined because of the
    // double inheritance from Base and DDGNode

    /// @name Class reflection system
    /// @{

    template<class T>
    static std::string typeName(const T* ptr= NULL)
    {
        return core::objectmodel::BaseObject::typeName(ptr);
    }

    /// Helper method to get the class name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::className(ptr); \endcode
    /// This way derived classes can redefine the className method
    template<class T>
    static std::string className(const T* ptr= NULL)
    {
        return core::objectmodel::BaseObject::className(ptr);
    }

    /// Helper method to get the namespace name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::namespaceName(ptr); \endcode
    /// This way derived classes can redefine the namespaceName method
    template<class T>
    static std::string namespaceName(const T* ptr= NULL)
    {
        return core::objectmodel::BaseObject::namespaceName(ptr);
    }

    /// Helper method to get the template name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::templateName(ptr); \endcode
    /// This way derived classes can redefine the templateName method
    template<class T>
    static std::string templateName(const T* ptr= NULL)
    {
        return core::objectmodel::BaseObject::templateName(ptr);
    }

    /// Helper method to get the shortname of a type derived from this class.
    /// The default implementation return the class name.
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::shortName(ptr); \endcode
    /// This way derived classes can redefine the shortName method
    template< class T>
    static std::string shortName( const T* ptr = NULL, core::objectmodel::BaseObjectDescription* desc = NULL )
    {
        return core::objectmodel::BaseObject::shortName(ptr,desc);
    }

    template<class T>
    static void dynamicCast(T*& ptr, Base* b)
    {
        core::objectmodel::BaseObject::dynamicCast(ptr, b);
    }

    /// @}

    /// This method is needed by DDGNode
    const std::string& getName() const override
    {
        return objectmodel::BaseObject::getName();
    }

    /// This method is needed by DDGNode
    objectmodel::Base* getOwner() const override
    {
        return const_cast<DataEngine*>(this);
    }

    /// This method is needed by DDGNode
    objectmodel::BaseData* getData() const override
    {
        return NULL;
    }

    /// Add a link.
    void addLink(objectmodel::BaseLink* l)
    {
        objectmodel::BaseObject::addLink(l);
    }

};

} // namespace core

} // namespace sofa

#endif
