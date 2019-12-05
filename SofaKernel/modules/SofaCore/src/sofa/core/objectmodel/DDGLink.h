#pragma once

#include <sofa/core/objectmodel/DDGNode.h>
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/core.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{


/**
 * @brief The BaseLink class
 *
 * BaseLink inherits DDGNode, thus is part of the data dependency graph, thus can have inputs and outputs.
 * When setting a link, the linked base's componentState data is added as an input to the BaseLink,
 * which creates the connection between the BaseLink and the DDG.
 * any data, engine, etc. can then be connected as output.
 */
class SOFA_CORE_API BaseDDGLink : public DDGNode
{
public:
    /// Flags that describe some properties of a Data, and that can be OR'd together.
    /// \todo Probably remove FLAG_PERSISTENT, FLAG_ANIMATION_INSTANCE, FLAG_VISUAL_INSTANCE and FLAG_HAPTICS_INSTANCE, it looks like they are not used anywhere.
    enum DataFlagsEnum
    {
        FLAG_NONE       = 0,      ///< Means "no flag" when a value is required.
        FLAG_READONLY   = 1 << 0, ///< The Data will be read-only in GUIs.
        FLAG_DISPLAYED  = 1 << 1, ///< The Data will be displayed in GUIs.
        FLAG_PERSISTENT = 1 << 2, ///< The Data contains persistent information.
        FLAG_AUTOLINK   = 1 << 3, ///< The Data should be autolinked when using the src="..." syntax.
        FLAG_REQUIRED = 1 << 4, ///< True if the Data has to be set for the owner component to be valid (a warning is displayed at init otherwise)
        FLAG_ANIMATION_INSTANCE = 1 << 10,
        FLAG_VISUAL_INSTANCE = 1 << 11,
        FLAG_HAPTICS_INSTANCE = 1 << 12,
    };
    /// Bit field that holds flags value.
    typedef unsigned DataFlags;

    /// Default value used for flags.
    enum { FLAG_DEFAULT = FLAG_DISPLAYED | FLAG_PERSISTENT | FLAG_AUTOLINK };

    /// This internal class is used by the initLink() methods to store initialization parameters of a Data
    class InitDDGLink
    {
    public:
        InitDDGLink()
            : name(""),
              help(""),
              group(""),
              linkedBase(nullptr),
              owner(nullptr),
              dataFlags(FLAG_DEFAULT) {}
        std::string name;
        std::string help;
        std::string group;
        Base* linkedBase;
        Base* owner;
        DataFlags dataFlags;
    };

    explicit BaseDDGLink(const InitDDGLink& init);

    virtual ~BaseDDGLink() override;

    void setOwner(Base* owner);

    void set(Base* linkedBase);

    Base* get();

    virtual void update() override;

    virtual const std::string& getName() const override;

    virtual Base* getOwner() const override;

    virtual BaseData* getData() const override;

    std::string getPathName() const;

protected:
    std::string m_name {""};
    std::string m_help {""};
    std::string m_group {""};
    Base* m_linkedBase {nullptr};
    Base* m_owner {nullptr};
    BaseData::DataFlags m_dataFlags {BaseData::FLAG_DEFAULT};

private:
    /// Number of changes since creation
    sofa::helper::fixed_array<int, sofa::core::SOFA_DATA_MAX_ASPECTS> m_counters;
};


template <class T>
class DDGLink : public BaseDDGLink
{
  public:

    explicit DDGLink(const BaseDDGLink::InitDDGLink& init)
        : BaseDDGLink(init)
    {
    }

    virtual ~DDGLink()
    {
    }

    void set(T* linkedBase)
    {
        BaseDDGLink::set(linkedBase);
    }

    T* get()
    {
        return dynamic_cast<T*>(m_linkedBase);
    }
};

} // namespace objectmodel
} // namespace core
} // namespace sofa
