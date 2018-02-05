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
#ifndef SOFA_CORE_OBJECTMODEL_DDGNODE_H
#define SOFA_CORE_OBJECTMODEL_DDGNODE_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/helper/fixed_array.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/core.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/core/objectmodel/BaseClass.h>
#include <list>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;
class BaseData;
class DDGNode;
class BaseObjectDescription;

template<>
class LinkTraitsPtrCasts<DDGNode>
{
public:
    static sofa::core::objectmodel::Base* getBase(sofa::core::objectmodel::DDGNode* n);
    static sofa::core::objectmodel::BaseData* getData(sofa::core::objectmodel::DDGNode* n);
};

/**
 *  \brief Abstract base to manage data dependencies. BaseData and DataEngine inherites from this class
 *
 */
class SOFA_CORE_API DDGNode
{
public:

    typedef MultiLink<DDGNode, DDGNode, BaseLink::FLAG_DOUBLELINK|BaseLink::FLAG_DATALINK> DDGLink;
    typedef DDGLink::Container DDGLinkContainer;
    typedef DDGLink::const_iterator DDGLinkIterator;

    /// Constructor
    DDGNode();

    /// Destructor. Automatically remove remaining links
    virtual ~DDGNode();

    /// @name Class reflection system
    /// @{
    typedef TClass<DDGNode> MyClass;
    static const MyClass* GetClass() { return MyClass::get(); }
    virtual const BaseClass* getClass() const
    { return GetClass(); }

    template<class T>
    static void dynamicCast(T*& ptr, Base* /*b*/)
    {
        ptr = NULL; // DDGNode does not derive from Base
    }

    /// Helper method to get the type name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::typeName(ptr); \endcode
    /// This way derived classes can redefine the typeName method
    template<class T>
    static std::string typeName(const T* ptr= NULL)
    {
        return BaseClass::defaultTypeName(ptr);
    }

    /// Helper method to get the class name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::className(ptr); \endcode
    /// This way derived classes can redefine the className method
    template<class T>
    static std::string className(const T* ptr= NULL)
    {
        return BaseClass::defaultClassName(ptr);
    }

    /// Helper method to get the namespace name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::namespaceName(ptr); \endcode
    /// This way derived classes can redefine the namespaceName method
    template<class T>
    static std::string namespaceName(const T* ptr= NULL)
    {
        return BaseClass::defaultNamespaceName(ptr);
    }

    /// Helper method to get the template name of a type derived from this class
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::templateName(ptr); \endcode
    /// This way derived classes can redefine the templateName method
    template<class T>
    static std::string templateName(const T* ptr= NULL)
    {
        return BaseClass::defaultTemplateName(ptr);
    }

    /// Helper method to get the shortname of a type derived from this class.
    /// The default implementation return the class name.
    ///
    /// This method should be used as follow :
    /// \code  T* ptr = NULL; std::string type = T::shortName(ptr); \endcode
    /// This way derived classes can redefine the shortName method
    template< class T>
    static std::string shortName( const T* ptr = NULL, BaseObjectDescription* = NULL )
    {
        std::string shortname = T::className(ptr);
        if( !shortname.empty() )
        {
            *shortname.begin() = ::tolower(*shortname.begin());
        }
        return shortname;
    }
    /// @}

    /// Add a new input to this node
    void addInput(DDGNode* n);

    /// Remove an input from this node
    void delInput(DDGNode* n);

    /// Add a new output to this node
    void addOutput(DDGNode* n);

    /// Remove an output from this node
    void delOutput(DDGNode* n);

    /// Get the list of inputs for this DDGNode
    const DDGLinkContainer& getInputs();

    /// Get the list of outputs for this DDGNode
    const DDGLinkContainer& getOutputs();

    /// Update this value
    virtual void update() = 0;

    /// Returns true if the DDGNode needs to be updated
    bool isDirty(const core::ExecParams* params = 0) const
    {
        return dirtyFlags[currentAspect(params)].dirtyValue;
    }

    /// Indicate the value needs to be updated
    virtual void setDirtyValue(const core::ExecParams* params = 0);

    /// Indicate the outputs needs to be updated. This method must be called after changing the value of this node.
    virtual void setDirtyOutputs(const core::ExecParams* params = 0);

    /// Set dirty flag to false
    void cleanDirty(const core::ExecParams* params = 0);

    /// Utility method to call update if necessary. This method should be called before reading of writing the value of this node.
    void updateIfDirty(const core::ExecParams* params = 0) const
    {
        if (isDirty(params))
        {
            const_cast <DDGNode*> (this)->update();
        }
    }

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect);

    static int currentAspect()
    {
        return core::ExecParams::currentAspect();
    }
    static int currentAspect(const core::ExecParams* params)
    {
        return core::ExecParams::currentAspect(params);
    }

    virtual const std::string& getName() const = 0;

    virtual Base* getOwner() const = 0;

    virtual BaseData* getData() const = 0;

    virtual bool findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link);

    void addLink(BaseLink* l);

protected:

    BaseLink::InitLink<DDGNode>
    initLink(const char* name, const char* help)
    {
        return BaseLink::InitLink<DDGNode>(this, name, help);
    }

    //std::list<DDGNode*> inputs;
    //std::list<DDGNode*> outputs;
    DDGLink inputs;
    DDGLink outputs;

    virtual void doAddInput(DDGNode* n)
    {
        inputs.add(n);
    }

    virtual void doDelInput(DDGNode* n)
    {
        inputs.remove(n);
    }

    virtual void doAddOutput(DDGNode* n)
    {
        outputs.add(n);
    }

    virtual void doDelOutput(DDGNode* n)
    {
        outputs.remove(n);
    }

    /// the dirtyOutputs flags of all the inputs will be set to false
    void cleanDirtyOutputsOfInputs(const core::ExecParams* params);

private:

    struct DirtyFlags
    {
        DirtyFlags() : dirtyValue(false), dirtyOutputs(false) {}

        bool dirtyValue;
        bool dirtyOutputs;
    };
    helper::fixed_array<DirtyFlags, SOFA_DATA_MAX_ASPECTS> dirtyFlags;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
