/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::core::objectmodel
{

BaseData::BaseData(const char* h, DataFlags dataflags) : BaseData(sofa::helper::safeCharToString(h), dataflags)
{
}

BaseData::BaseData(const std::string& h, DataFlags dataflags)
    : help(h), ownerClass(""), group(""), widget("")
    , m_counter(), m_isSet(), m_dataFlags(dataflags)
    , m_owner(nullptr), m_name("")
    , parentData(*this)
{
    m_counter = 0;
    m_isSet = false;
    setFlag(FLAG_PERSISTENT, false);
}

BaseData::BaseData( const char* helpMsg, bool isDisplayed, bool isReadOnly) : BaseData(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly)
{
}


BaseData::BaseData( const std::string& h, bool isDisplayed, bool isReadOnly)
    : help(h), group(""), widget("")
    , m_counter(), m_isSet(), m_dataFlags(FLAG_DEFAULT), m_owner(nullptr), m_name("")
    , parentData(*this)
{
    m_counter = 0;
    m_isSet = false;
    setFlag(FLAG_DISPLAYED,isDisplayed);
    setFlag(FLAG_READONLY,isReadOnly);
    setFlag(FLAG_PERSISTENT, false);
}

BaseData::BaseData( const BaseInitData& init)
    : help(init.helpMsg), group(init.group), widget(init.widget)
    , m_counter(), m_isSet(), m_dataFlags(init.dataFlags)
    , m_owner(init.owner), m_name(init.name)
    , parentData(*this)
{
    m_counter = 0;
    m_isSet = false;

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
    if (m_owner) m_owner->addData(this, m_name);
    setFlag(FLAG_PERSISTENT, false);
}

BaseData::~BaseData()
{
}

bool BaseData::validParent(const BaseData* parent)
{
    /// Query the implementation side to see if they two are similar.
    if(doIsExactSameDataType(parent))
        return true;

    /// If not, check if automatic conversion is possible
    if (this->getValueTypeInfo()->ValidInfo() && parent->getValueTypeInfo()->ValidInfo())
        return true;

    /// If not, check if one of the data is a simple string
    if (this->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name() || parent->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name())
        return true;

    return false;
}

bool BaseData::setParent(BaseData* parent, const std::string& path)
{
    /// First remove previous parents
    while (!this->inputs.empty())
        this->delInput(*this->inputs.begin());

    if (parent && !validParent(parent))
    {
        if (m_owner)
        {
            msg_error(m_owner) << "Invalid Data link from " << (parent->m_owner ? parent->m_owner->getName() : std::string("?")) << "." << parent->getName() << " to " << m_owner->getName() << "." << getName();
            msg_error_when(!this->getValueTypeInfo()->ValidInfo(), m_owner) << "Possible reason: destination Data " << getName() << " has an unknown type";
            msg_error_when(!parent->getValueTypeInfo()->ValidInfo(), m_owner) << "Possible reason: source Data " << parent->getName() << " has an unknown type";
        }
        return false;
    }

    parentData.setTarget(parent);
    if (parent)
    {
        addInput(parent);
        BaseData::setDirtyValue();
        m_counter++;
    }
    // the referenced parent data has not been created yet but a path has been given
    else if (!path.empty())
    {
        parentData.setPath(path);
    }

    m_isSet = true;

    return true;
}

std::string BaseData::getPathName() const
{
    if(m_owner)
        return m_owner->getPathName()+"."+getName();
    return getName();
}

std::string BaseData::getLinkPath() const
{
    if(m_owner)
        return "@"+getPathName();
    return "";
}

bool BaseData::setParent(const std::string& path)
{
    parentData.setPath(path);
    return setParent(parentData.getTarget(), parentData.getPath());
}

void BaseData::doDelInput(DDGNode* n)
{
    if (parentData.getTarget() == n)
        parentData.setTarget(nullptr);
    DDGNode::doDelInput(n);
}

void BaseData::update()
{
    cleanDirty();
    for (DDGNode* input : inputs)
    {
        input->updateIfDirty();
    }

    /// Check if there is a parent (so a predecessor in the DDG), if so
    /// update the internal value.
    const auto parent = parentData.resolvePathAndGetTarget();
    if (parent)
    {
        updateValueFromLink(parent);
        // If the value is dirty clean it
        if(this->isDirty())
        {
            cleanDirty();
        }
    }
    doOnUpdate();
}

bool BaseData::updateValueFromLink(const BaseData* parent)
{
    /// Try if the fast path succeeded, in general this means that the two Data are
    /// of exactly the same internal types.
    if(doSetValueFromLink(parent))
    {
        return true;
    }
    return copyValueFrom(parent);
}

/// Update this Data from the value of its parent
bool BaseData::copyValueFrom(const BaseData* parent)
{
    /// Try if the fast path succeeded, in general this means that the two Data are
    /// of exactly the same internal types.
    if(doCopyValueFrom(parent))
    {
        return true;
    }
    /// If the fast path didn't succeeded we try to do copy using a much less efficient approach.
    return genericCopyValueFrom(parent);
}


/// Update this Data from the value of its parent
bool BaseData::genericCopyValueFrom(const BaseData* parent)
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
    sofa::Size inSize = 1;
    sofa::Size outSize = 1;
    sofa::Size copySize = 1;
    sofa::Size nbl = 1;
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
        const auto dataBSize = dataInfo->size();
        const auto parentBSize = parentInfo->size();
        if (dataBSize > parentBSize)
            msgs << "parent Data type " << parentInfo->name() << " contains " << parentBSize << " values per element while Data type " << dataInfo->name() << " requires " << dataBSize << " values.";
        else if (dataBSize < parentBSize)
            msgs << "parent Data type " << parentInfo->name() << " contains " << parentBSize << " values per element while Data type " << dataInfo->name() << " only requires " << dataBSize << " values.";
        const auto parentSize = parentInfo->size(parentValue);
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
        for (decltype(nbl) l=0; l<nbl; ++l)
            for (decltype(copySize) c=0; c<copySize; ++c)
                dataInfo->setIntegerValue(dataValue, l*outSize+c, parentInfo->getIntegerValue(parentValue, l*inSize+c));
    }
    else if ((dataInfo->Integer() || dataInfo->Scalar()) && (parentInfo->Integer() || parentInfo->Scalar()))
    {
        // scalar conversion
        for (decltype(nbl) l=0; l<nbl; ++l)
            for (decltype(copySize) c=0; c<copySize; ++c)
                dataInfo->setScalarValue(dataValue, l*outSize+c, parentInfo->getScalarValue(parentValue, l*inSize+c));
    }
    else
    {
        // text conversion
        for (decltype(nbl) l=0; l<nbl; ++l)
            for (decltype(copySize) c=0; c<copySize; ++c)
                dataInfo->setTextValue(dataValue, l*outSize+c, parentInfo->getTextValue(parentValue, l*inSize+c));
    }

    std::string m = msgs.str();
    if (m_owner
        #ifdef NDEBUG
            && (!m.empty() || m_owner->notMuted())
        #endif
            )
    {
        std::stringstream tmp;
        tmp << "Data link from " << (parent->m_owner ? parent->m_owner->getName() : std::string("?")) << "." << parent->getName() << " to " << m_owner->getName() << "." << getName() << " : ";
        if (!m.empty()) tmp << m;
        else            tmp << "OK, " << nbl << "*"<<copySize<<" values copied.";
        msg_info(m_owner) << tmp.str();
    }

    return true;
}

/// Get current value as a void pointer (use getValueTypeInfo to find how to access it)
const void* BaseData::getValueVoidPtr() const
{
    return doGetValueVoidPtr();
}

/// Begin edit current value as a void pointer (use getValueTypeInfo to find how to access it)
void* BaseData::beginEditVoidPtr()
{
    return doBeginEditVoidPtr();
}

/// End edit current value as a void pointer (use getValueTypeInfo to find how to access it)
void BaseData::endEditVoidPtr()
{
    doEndEditVoidPtr();
}

std::string BaseData::decodeTypeName(const std::type_info& t)
{
    return sofa::helper::NameDecoder::decodeTypeName(t);
}

std::ostream& operator<<(std::ostream &out, const sofa::core::objectmodel::BaseData& df)
{
    out<<df.getValueString();
    return out;
}

} // namespace sofa::core::objectmodel
