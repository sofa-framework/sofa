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

    std::cout << "constructing ddglink with name " << m_name << std::endl;
    if (m_owner)
    {
        std::cout << "adding ddglink with name " << m_name << " to " << m_owner->getName() << std::endl;
        m_owner->addDDGLink(this, m_name);
        std::cout << m_owner->findGlobalDDGLink(m_name).size() << std::endl;
    }
}

BaseDDGLink::~BaseDDGLink()
{

}

void BaseDDGLink::setOwner(Base* owner)
{
    m_owner = owner;
}

void BaseDDGLink::set(Base* linkedBase)
{
    m_linkedBase = linkedBase;
    addInput(&m_linkedBase->d_componentstate);
    ++m_counters[size_t(currentAspect())];
    setDirtyOutputs();
}

void BaseDDGLink::set(const Base* linkedBase)
{
    /// storing the ptr as non-const.. but nowhere should the ptr be modified afterwards if manipulating a DDGLink<T>
    /// When manipulating BaseDDGLinks, be careful to use the correct getter or undefined behavior will occur.
    m_linkedBase = const_cast<Base*>(linkedBase);
    addInput(&m_linkedBase->d_componentstate);
    ++m_counters[size_t(currentAspect())];
    setDirtyOutputs();
}

const Base* BaseDDGLink::get() const
{
    const_cast <BaseDDGLink*> (this)->update();
    return m_linkedBase;
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

    std::string pathname = m_owner->name.getLinkPath();
    return pathname.substr(0, pathname.find_last_of(".")) + getName();
}

} // namespace objectmodel
} // namespace core
} // namespace sofa
