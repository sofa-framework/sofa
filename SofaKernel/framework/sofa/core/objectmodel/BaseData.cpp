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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

//#define SOFA_DDG_TRACE

BaseData::BaseData(const char* h, DataFlags dataflags)
    : help(h), ownerClass(""), group(""), widget("")
    , m_counters(), m_isSets(), m_dataFlags(dataflags)
    , m_owner(NULL), m_name("")
    , parentBaseData(initLink("parent", "Linked Data, from which values are automatically copied"))
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);
    m_isSets.assign(false);
    //setAutoLink(true);
    //if (owner) owner->addData(this);
}

BaseData::BaseData( const char* h, bool isDisplayed, bool isReadOnly)
    : help(h), ownerClass(""), group(""), widget("")
    , m_counters(), m_isSets(), m_dataFlags(FLAG_DEFAULT), m_owner(NULL), m_name("")
    , parentBaseData(initLink("parent", "Linked Data, from which values are automatically copied"))
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);
    m_isSets.assign(false);
    setFlag(FLAG_DISPLAYED,isDisplayed);
    setFlag(FLAG_READONLY,isReadOnly);
    //setAutoLink(true);
    //if (owner) owner->addData(this);
}

BaseData::BaseData( const BaseInitData& init)
    : help(init.helpMsg), ownerClass(init.ownerClass), group(init.group), widget(init.widget)
    , m_counters(), m_isSets(), m_dataFlags(init.dataFlags)
    , m_owner(init.owner), m_name(init.name)
    , parentBaseData(initLink("parent", "Linked Data, from which values are automatically copied"))
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);
    m_isSets.assign(false);
    if (init.data && init.data != this)
    {
        {
        helper::logging::MessageDispatcher::LoggerStream msgerror = msg_error("BaseData");
        msgerror << "initData POINTER MISMATCH: field name \"" << init.name << "\"";
        if (init.owner)
            msgerror << " created by class " << init.owner->getClassName();
        }
        sofa::helper::BackTrace::dump();
        exit( EXIT_FAILURE );
    }
    //setAutoLink(true);
    if (m_owner) m_owner->addData(this, m_name);
}

BaseData::~BaseData()
{
}

bool BaseData::validParent(BaseData* parent)
{
    // Check if automatic conversion is possible
    if (this->getValueTypeInfo()->ValidInfo() && parent->getValueTypeInfo()->ValidInfo())
        return true;
    // Check if one of the data is a simple string
    if (this->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name() || parent->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name())
        return true;
    // No conversion found
    return false;
}

bool BaseData::setParent(BaseData* parent, const std::string& path)
{
    // First remove previous parents
    while (!this->inputs.empty())
        this->delInput(*this->inputs.begin());
    if (parent && !validParent(parent))
    {
        if (m_owner)
        {
            m_owner->serr << "Invalid Data link from " << (parent->m_owner ? parent->m_owner->getName() : std::string("?")) << "." << parent->getName() << " to " << m_owner->getName() << "." << getName() << m_owner->sendl;
            if (!this->getValueTypeInfo()->ValidInfo())
                m_owner->serr << "  Possible reason: destination Data " << getName() << " has an unknown type" << m_owner->sendl;
            if (!parent->getValueTypeInfo()->ValidInfo())
                m_owner->serr << "  Possible reason: source Data " << parent->getName() << " has an unknown type" << m_owner->sendl;
        }
        return false;
    }
    doSetParent(parent);
    if (!path.empty())
        parentBaseData.set(parent, path);
    if (parent)
    {
        addInput(parent);
        BaseData::setDirtyValue();
        if (!isCounterValid())
            update();

        m_counters[currentAspect()]++;
        m_isSets[currentAspect()] = true;
    }
    return true;
}

bool BaseData::setParent(const std::string& path)
{
    BaseData* parent = NULL;
    if (this->findDataLinkDest(parent, path, &parentBaseData))
        return setParent(parent, path);
    else // simply set the path
    {
        if (parentBaseData.get())
            this->delInput(parentBaseData.get());
        parentBaseData.set(parent, path);
        return false;
    }
}

void BaseData::doSetParent(BaseData* parent)
{
    parentBaseData.set(parent);
}

void BaseData::doDelInput(DDGNode* n)
{
    if (parentBaseData == n)
        doSetParent(NULL);
    DDGNode::doDelInput(n);
}

void BaseData::update()
{
    cleanDirty();
    for(DDGLinkIterator it=inputs.begin(); it!=inputs.end(); ++it)
    {
        if ((*it)->isDirty())
        {
            (*it)->update();
        }
    }
    if (parentBaseData)
    {
#ifdef SOFA_DDG_TRACE
        if (m_owner)
            m_owner->sout << "Data " << m_name << ": update from parent " << parentBaseData->m_name<< m_owner->sendl;
#endif
        updateFromParentValue(parentBaseData);
        // If the value is dirty clean it
        if(this->isDirty())
        {
            cleanDirty();
        }
    }
}

/// Update this Data from the value of its parent
bool BaseData::updateFromParentValue(const BaseData* parent)
{
    const defaulttype::AbstractTypeInfo* dataInfo = this->getValueTypeInfo();
    const defaulttype::AbstractTypeInfo* parentInfo = parent->getValueTypeInfo();

    // Check if one of the data is a simple string
    if (this->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name() || parent->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name())
    {
        std::string text = parent->getValueString();
        return this->read(text);
    }

    // Check if automatic conversion is possible
    if (!dataInfo->ValidInfo() || !parentInfo->ValidInfo())
        return false; // No conversion found
    std::ostringstream msgs;
    const void* parentValue = parent->getValueVoidPtr();
    void* dataValue = this->beginEditVoidPtr();

    // First decide how many values will be copied
    std::size_t inSize = 1;
    std::size_t outSize = 1;
    std::size_t copySize = 1;
    std::size_t nbl = 1;
    if (dataInfo->FixedSize())
    {
        outSize = dataInfo->size();
        inSize = parentInfo->size(parentValue);
        if (outSize > inSize)
        {
            msgs << "parent Data type " << parentInfo->name() << " contains " << inSize << " values while Data type " << dataInfo->name() << " requires " << outSize << " values.";
            copySize = inSize;
        }
        else if (outSize < inSize)
        {
            msgs << "parent Data type " << parentInfo->name() << " contains " << inSize << " values while Data type " << dataInfo->name() << " only requires " << outSize << " values.";
            copySize = outSize;
        }
        else
            copySize = outSize;
    }
    else
    {
        std::size_t dataBSize = dataInfo->size();
        std::size_t parentBSize = parentInfo->size();
        if (dataBSize > parentBSize)
            msgs << "parent Data type " << parentInfo->name() << " contains " << parentBSize << " values per element while Data type " << dataInfo->name() << " requires " << dataBSize << " values.";
        else if (dataBSize < parentBSize)
            msgs << "parent Data type " << parentInfo->name() << " contains " << parentBSize << " values per element while Data type " << dataInfo->name() << " only requires " << dataBSize << " values.";
        std::size_t parentSize = parentInfo->size(parentValue);
        inSize = parentBSize;
        outSize = dataBSize;
        nbl = parentSize / parentBSize;
        copySize = (dataBSize < parentBSize) ? dataBSize : parentBSize;
        dataInfo->setSize(dataValue, outSize * nbl);
    }

    // Then select the besttype for values transfer

    if (dataInfo->Integer() && parentInfo->Integer())
    {
        // integer conversion
        for (size_t l=0; l<nbl; ++l)
            for (size_t c=0; c<copySize; ++c)
                dataInfo->setIntegerValue(dataValue, l*outSize+c, parentInfo->getIntegerValue(parentValue, l*inSize+c));
    }
    else if ((dataInfo->Integer() || dataInfo->Scalar()) && (parentInfo->Integer() || parentInfo->Scalar()))
    {
        // scalar conversion
        for (size_t l=0; l<nbl; ++l)
            for (size_t c=0; c<copySize; ++c)
                dataInfo->setScalarValue(dataValue, l*outSize+c, parentInfo->getScalarValue(parentValue, l*inSize+c));
    }
    else
    {
        // text conversion
        for (size_t l=0; l<nbl; ++l)
            for (size_t c=0; c<copySize; ++c)
                dataInfo->setTextValue(dataValue, l*outSize+c, parentInfo->getTextValue(parentValue, l*inSize+c));
    }

    std::string m = msgs.str();
    if (m_owner
#ifdef NDEBUG
        && (!m.empty() || m_owner->notMuted())
#endif
    )
    {
        m_owner->sout << "Data link from " << (parent->m_owner ? parent->m_owner->getName() : std::string("?")) << "." << parent->getName() << " to " << m_owner->getName() << "." << getName() << " : ";
        if (!m.empty()) m_owner->sout << m;
        else            m_owner->sout << "OK, " << nbl << "*"<<copySize<<" values copied.";
        m_owner->sout << m_owner->sendl;
    }

    return true;
}

/// Copy the value of another Data.
/// Note that this is a one-time copy and not a permanent link (otherwise see setParent)
/// @return true if copy was successfull
bool BaseData::copyValue(const BaseData* parent)
{
    if (updateFromParentValue(parent))
        return true;
    return false;
}

bool BaseData::findDataLinkDest(DDGNode*& ptr, const std::string& path, const BaseLink* link)
{
    return DDGNode::findDataLinkDest(ptr, path, link);
}

bool BaseData::findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link)
{
    if (m_owner)
        return m_owner->findDataLinkDest(ptr, path, link);
    else
    {
        msg_error("BaseData") << "findDataLinkDest: no owner defined for Data " << getName() << ", cannot lookup Data link " << path;
        return false;
    }
}

/// Add a link.
/// Note that this method should only be called if the link was not initialized with the initLink method
void BaseData::addLink(BaseLink* l)
{
    m_vecLink.push_back(l);
}

void BaseData::copyAspect(int destAspect, int srcAspect)
{
    m_counters[destAspect] = m_counters[srcAspect];
    m_isSets[destAspect] = m_isSets[srcAspect];
    DDGNode::copyAspect(destAspect, srcAspect);
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        (*iLink)->copyAspect(destAspect, srcAspect);
    }
}

void BaseData::releaseAspect(int aspect)
{
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        (*iLink)->releaseAspect(aspect);
    }
}

std::string BaseData::decodeTypeName(const std::type_info& t)
{
    return BaseClass::decodeTypeName(t);
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

