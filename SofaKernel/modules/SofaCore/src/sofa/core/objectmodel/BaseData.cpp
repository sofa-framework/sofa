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
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

void BaseData::update()
{
    std::cout << "DDGNodeBaseData::update(BEGIN)" << std::endl;

    updateDirtyInputs();

    if(hasParent())
        copyValue(getParent());

    cleanDirty();
    std::cout << "DDGNodeBaseData::update(END)" << std::endl;
}


BaseData::BaseData(const char* h, DataFlags dataflags) :
    BaseData(sofa::helper::safeCharToString(h), dataflags)
{
}

BaseData::BaseData() :
    help(""), group(""), widget("")
  , m_counter(0), m_isSet(false), m_dataFlags(FLAG_DISPLAYED | FLAG_AUTOLINK)
  , m_owner(nullptr), m_name("")
  , m_parentData(*this)
{
}

BaseData::BaseData(const std::string& h, DataFlags dataflags)
    : BaseData()
{
    setHelp(h);
    m_dataFlags = dataflags;
}

BaseData::BaseData( const char* helpMsg, bool isDisplayed, bool isReadOnly) : BaseData(sofa::helper::safeCharToString(helpMsg), isDisplayed, isReadOnly)
{
}

BaseData::BaseData( const std::string& h, bool isDisplayed, bool isReadOnly)
    : BaseData()
{
    setHelp(h);
    setDisplayed(isDisplayed);
    setReadOnly(isReadOnly);
}

BaseData::BaseData( const BaseInitData& init)
    : BaseData()
{
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

    setOwner(init.owner);
    setHelp(init.helpMsg);
    setGroup(init.group);
    setWidget(init.widget);
    m_dataFlags = init.dataFlags;
    setName(init.name);

    if (m_owner)
        m_owner->addData(this, m_name);
}

BaseData::~BaseData()
{
}

bool BaseData::canBeParent(BaseData* parent)
{
    if (parent==nullptr)
        return false;

    // Check if automatic conversion is possible
    if (getValueTypeInfo()->ValidInfo() && parent->getValueTypeInfo()->ValidInfo())
        return true;

    // Check if one of the data is a simple string
    //// TODO(dmarchal: 2020-03-27...why is this needed ? Why getValueTypeInfo() not enough)
    if (getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name()
            || parent->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name())
        return true;

    // No conversion found
    return false;
}

bool BaseData::setParent(BaseData* parent)
{
    assert(parent != nullptr && "it is not allowed to use nullptr as parent.");

    if(parent==nullptr)
        return false;

    if(!canBeParent(parent))
        return false;

    m_parentData.set(parent);
    return true;

}

/// Update this Data from the value of its parent
bool BaseData::updateFromParentValue(const BaseData* parent)
{
    return copyValue(parent);
}

const std::string& BaseData::getOwnerClass() const
{
    return m_owner->getClass()->className;
}

/// Copy the value of another Data.
/// Note that this is a one-time copy and not a permanent link (otherwise see setParent)
/// @return true if copy was successfull
bool BaseData::copyValue(const BaseData* parent)
{
    std::cout << "BaseData::copyValue "<< this->getName() << "," << parent->getName() << std::endl;
    const defaulttype::AbstractTypeInfo* dataInfo = this->getValueTypeInfo();
    const defaulttype::AbstractTypeInfo* parentInfo = parent->getValueTypeInfo();

    // Check if one of the data is a simple string
    //TODO(dmarchal 2020-03-20: Deprecate this and replace with a fast mecanisme.
    if (this->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name()
            || parent->getValueTypeInfo()->name() == defaulttype::DataTypeInfo<std::string>::name())
    {
        std::string text = parent->getValueString();
        return this->read(text);
    }

    std::cout << "UPDAte FROM PARENT VALUE 2 "<< this << "," << parent << std::endl;

    // Check if automatic conversion is possible
    if (!dataInfo->ValidInfo() || !parentInfo->ValidInfo())
        return false; // No conversion found
    std::ostringstream msgs;

    std::cout << "UPDAte FROM PARENT VALUE 2.1 "<< this->getName() << "," << parent->getName() << std::endl;

    const void* parentValue = parent->getValueVoidPtr();
    std::cout << "UPDAte FROM PARENT VALUE 2.2 "<< this->getName() << "," << parent->getName() << std::endl;

    void* dataValue = this->beginWriteOnlyVoidPtr();

    std::cout << "UPDAte FROM PARENT VALUE 3"<< this->getName() << "," << parent->getName() << std::endl;


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
    if (m_owner)
    {
        m_owner->sout << "Data link from " << (parent->m_owner ? parent->m_owner->getName() : std::string("?")) << "." << parent->getName() << " to " << m_owner->getName() << "." << getName() << " : ";
        if (!m.empty()) m_owner->sout << m;
        else            m_owner->sout << "OK, " << nbl << "*"<<copySize<<" values copied.";
        m_owner->sout << m_owner->sendl;
    }
    return true;
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

