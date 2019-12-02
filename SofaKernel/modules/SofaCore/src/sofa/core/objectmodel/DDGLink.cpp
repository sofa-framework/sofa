#include "DDGLink.h"

namespace sofa
{
namespace core
{
namespace objectmodel
{

BaseDDGLink::BaseDDGLink(const BaseDDGLink::InitDDGLink &init)
    : m_name(init.name),
      m_help(init.help),
      m_group(init.group),
      m_linkedBase(init.linkedBase),
      m_owner(init.owner),
      m_dataFlags(init.dataFlags)
{
    addLink(&inputs);
    addLink(&outputs);
    m_counters.assign(0);

    if (m_owner)
    {
        m_owner->addDDGLink(this, m_name);
        if (m_linkedBase)
            m_linkedBase->addDDGLinkOwner(m_owner);
    }
}

BaseDDGLink::~BaseDDGLink()
{

}

void BaseDDGLink::setOwner(Base* owner)
{
    if (m_linkedBase)
    {
        m_linkedBase->removeDDGLinkOwner(m_owner);
        m_linkedBase->addDDGLinkOwner(owner);
    }
    m_owner = owner;
}

void BaseDDGLink::set(Base* linkedBase)
{
    if (m_linkedBase)
        m_linkedBase->removeDDGLinkOwner(m_owner);
    m_linkedBase = linkedBase;
    if (m_linkedBase)
    {
        linkedBase->addDDGLinkOwner(m_owner);
        addInput(&m_linkedBase->d_componentstate);
    }
    ++m_counters[size_t(currentAspect())];
    setDirtyOutputs();
}

Base* BaseDDGLink::get()
{
    update();
    return m_linkedBase;
}

void BaseDDGLink::update()
{
    for(DDGLinkIterator it=inputs.begin(); it!=inputs.end(); ++it)
    {
        if ((*it)->isDirty())
        {
            (*it)->update();
        }
    }
    ++m_counters[size_t(currentAspect())];
    cleanDirty();
}

const std::string& BaseDDGLink::getName() const
{
    return m_name;
}

Base* BaseDDGLink::getOwner() const
{
    return m_owner;
}

BaseData* BaseDDGLink::getData() const
{
    return nullptr;
}

std::string BaseDDGLink::getPathName() const
{
    if (!m_owner)
        return getName();

    std::string pathname = m_owner->getPathName();
    return pathname + "." + getName();
}

} // namespace objectmodel
} // namespace core
} // namespace sofa
