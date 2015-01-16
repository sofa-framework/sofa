/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/helper/BackTrace.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

//#define SOFA_DDG_TRACE

const std::string DDGDataNode::NO_NAME;
const DDGNode::DDGLinkContainer DDGDataNode::NO_LINKS;
DDGDataNode::CreationArray DDGDataNode::m_creationEnabled(true);

DDGDataNode::DDGDataNode(BaseData* data, const char* h)
    : help(h), ownerClass(""), group(""), widget("")
    , m_counters(), m_isSets()
    , m_name("")
    , m_parent(initLink("parent", "Linked Data, from which values are automatically copied"))
	, m_data(data)
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);
    m_isSets.assign(false);
}

DDGDataNode::DDGDataNode(const BaseInitData& init)
    : help(init.helpMsg), ownerClass(init.ownerClass), group(init.group), widget(init.widget)
    , m_counters(), m_isSets()
    , m_name(init.name)
    , m_parent(initLink("parent", "Linked Data, from which values are automatically copied"))
	, m_data(init.data)
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);
    m_isSets.assign(false);
}

Base* DDGDataNode::getOwner() const
{
	return m_data->getOwner();
}

BaseData* DDGDataNode::getData() const
{
	return const_cast<BaseData*>(m_data);
}

void DDGDataNode::update()
{
    cleanDirty();
    for(DDGLinkIterator it=inputs.begin(); it!=inputs.end(); ++it)
    {
        if ((*it)->isDirty())
        {
            (*it)->update();
        }
    }

    if (m_parent)
    {
#ifdef SOFA_DDG_TRACE
        if (getOwner())
            getOwner()->sout << "Data " << m_name << ": update from parent " << m_parent->m_name << getOwner()->sendl;
#endif
		m_data->updateFromParentValue(m_parent->m_data);

        // If the value is dirty clean it
        if(this->isDirty())
        {
            cleanDirty();
        }
    }
}

bool DDGDataNode::setParent(DDGDataNode* parent, const std::string& path)
{
    // First remove previous parents
    while (!this->inputs.empty())
        this->delInput(*this->inputs.begin());

    if (parent && !m_data->validParent(parent->m_data))
    {
        if (getOwner())
        {
            getOwner()->serr << "Invalid Data link from " << (parent->getOwner() ? parent->getOwner()->getName() : std::string("?")) << "." << parent->getName() << " to " << getOwner()->getName() << "." << getName() << getOwner()->sendl;
            if (!m_data->getValueTypeInfo()->ValidInfo())
                getOwner()->serr << "  Possible reason: destination Data " << getName() << " has an unknown type" << getOwner()->sendl;
            if (!parent->m_data->getValueTypeInfo()->ValidInfo())
                getOwner()->serr << "  Possible reason: source Data " << parent->getName() << " has an unknown type" << getOwner()->sendl;
        }
        return false;
    }

	if (!path.empty())
		m_parent.set(parent, path);
	else
		m_parent.set(parent);

    if (parent)
    {
        addInput(parent);
        setDirtyValue();
        if (!m_data->isCounterValid())
            update();

        m_counters[currentAspect()]++;
        m_isSets[currentAspect()] = true;
    }

	m_data->onParentChanged(parent ? parent->m_data : NULL);

    return true;
}

void DDGDataNode::removeParent()
{
	if (m_parent.get())
		delInput(m_parent.get());
}

/// Add a link.
/// Note that this method should only be called if the link was not initialized with the initLink method
void DDGDataNode::addLink(BaseLink* l)
{
    m_vecLink.push_back(l);
    //l->setOwner(this);
}

bool DDGDataNode::findDataLinkDest(DDGDataNode*& ptr, const std::string& path, const BaseLink* link)
{
	Base* owner = getOwner();
    if (owner)
	{
		BaseData* data = ptr->m_data;
        bool result = owner->findDataLinkDest(data, path, link);
		ptr = BaseData::ddg(data);
		return result;
	}
    else
    {
        std::cerr << "DATA LINK ERROR: no owner defined for Data " << getName() << ", cannot lookup Data link " << path << std::endl;
        return false;
    }
}

void DDGDataNode::doDelInput(DDGNode* n)
{
    if (m_parent == n)
        m_parent.set(NULL);
    DDGNode::doDelInput(n);
}

void DDGDataNode::copyAspect(int destAspect, int srcAspect)
{
    m_counters[destAspect] = m_counters[srcAspect];
    m_isSets[destAspect] = m_isSets[srcAspect];
    DDGNode::copyAspect(destAspect, srcAspect);
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        //std::cout << "  " << iLink->first;
        (*iLink)->copyAspect(destAspect, srcAspect);
    }
}

void DDGDataNode::releaseAspect(int aspect)
{
    for(VecLink::const_iterator iLink = m_vecLink.begin(); iLink != m_vecLink.end(); ++iLink)
    {
        (*iLink)->releaseAspect(aspect);
    }
}

void DDGDataNode::enableCreation(bool enable)
{
	m_creationEnabled[currentAspect()] = enable;
}

bool DDGDataNode::isCreationEnabled()
{
	return m_creationEnabled[currentAspect()];
}



sofa::core::objectmodel::Base* LinkTraitsPtrCasts<DDGDataNode>::getBase(sofa::core::objectmodel::DDGDataNode* n)
{
    if (!n) return NULL;
    return n->getOwner();
}

sofa::core::objectmodel::BaseData* LinkTraitsPtrCasts<DDGDataNode>::getData(sofa::core::objectmodel::DDGDataNode* n)
{
    if (!n) return NULL;
    return n->getData();
}



BaseData::BaseData(const char* h, DataFlags dataflags)
    : m_dataFlags(dataflags), m_ddg(NULL), m_owner(NULL)
{
	if (DDGDataNode::isCreationEnabled())
		m_ddg = new DDGDataNode(this, h);
}

BaseData::BaseData( const char* h, bool isDisplayed, bool isReadOnly)
    : m_dataFlags(FLAG_DEFAULT), m_ddg(NULL), m_owner(NULL)
{
    setFlag(FLAG_DISPLAYED,isDisplayed);
    setFlag(FLAG_READONLY,isReadOnly);
    
	if (DDGDataNode::isCreationEnabled())
		m_ddg = new DDGDataNode(this, h);
}

BaseData::BaseData( const BaseInitData& init)
    : m_dataFlags(init.dataFlags), m_ddg(NULL), m_owner(init.owner)
{
    if (init.data && init.data != this)
    {
        std::cerr << "CODE ERROR: initData POINTER MISMATCH: field name \"" << init.name << "\"";
        if (init.owner)
            std::cerr << " created by class " << init.owner->getClassName();
        std::cerr << "!...aborting" << std::endl;
        sofa::helper::BackTrace::dump();
        exit( EXIT_FAILURE );
    }

	if (DDGDataNode::isCreationEnabled())
		m_ddg = new DDGDataNode(init);
    
    if (m_owner)
		m_owner->addData(this);
}

BaseData::~BaseData()
{
	if (m_ddg)
		delete m_ddg;
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
	if (m_ddg)
	{
		return m_ddg->setParent(parent ? parent->m_ddg : NULL, path);
	}
	return false;
}

bool BaseData::setParent(const std::string& path)
{
	if (m_ddg)
	{
		BaseData* parent = NULL;
		bool resolved = findDataLinkDest(parent, path, &m_ddg->m_parent);
		return m_ddg->setParent(parent ? parent->m_ddg : NULL, path) && resolved;
    }
    return false;
}

void BaseData::removeParent()
{
	if (m_ddg)
		m_ddg->removeParent();
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
        && (!m.empty() || m_owner->f_printLog.getValue())
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

bool BaseData::findDataLinkDest(BaseData*& ptr, const std::string& path, const BaseLink* link)
{
    if (m_owner)
        return m_owner->findDataLinkDest(ptr, path, link);
    else
    {
        std::cerr << "DATA LINK ERROR: no owner defined for Data " << getName() << ", cannot lookup Data link " << path << std::endl;
        return false;
    }
}

void BaseData::copyAspect(int destAspect, int srcAspect)
{
	if (m_ddg)
		m_ddg->copyAspect(destAspect, srcAspect);
}

void BaseData::releaseAspect(int aspect)
{
	if (m_ddg)
		m_ddg->releaseAspect(aspect);
}

void BaseData::cleanDdg()
{
	if (m_ddg)
	{
		if (m_ddg->getInputs().empty() && m_ddg->getOutputs().empty())
		{
			delete m_ddg;
			m_ddg = NULL;
		}
	}
}

std::string BaseData::decodeTypeName(const std::type_info& t)
{
    return BaseClass::decodeTypeName(t);
}

} // namespace objectmodel

} // namespace core

} // namespace sofa

