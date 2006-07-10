#ifndef SOFA_COMPONENTS_COMMON_FNDISPATCHER_H
#define SOFA_COMPONENTS_COMMON_FNDISPATCHER_H

#include <map>
#include <typeinfo>

namespace Sofa
{

namespace Components
{

namespace Common
{

template <class BaseClass, typename ResulT = void>
class BasicDispatcher
{
public:
    typedef ResulT (*F)(BaseClass &,BaseClass &);

protected:
    class TypeInfo
    {
    public:
        const std::type_info* pt;
        TypeInfo(const std::type_info& t) : pt(&t) { }
        operator const std::type_info&() const { return *pt; }
        bool operator==(const TypeInfo& t) const { return *pt == *t.pt; }
        bool operator<(const TypeInfo& t) const { return pt->before(*t.pt)!=0; }
    };
    typedef std::pair<TypeInfo,TypeInfo> KeyType;
    typedef std::map<KeyType, F> MapType;
    MapType callBackMap;
    virtual ~BasicDispatcher();
public:
    void add(const std::type_info& class1, const std::type_info& class2, F fun)
    {
        callBackMap[KeyType(class1,class2)] = fun;
    }

    void ignore(const std::type_info& class1, const std::type_info& class2)
    {
        callBackMap[KeyType(class1,class2)] = ignoreFn;
    }

    template <class ConcreteClass1,class ConcreteClass2,ResulT (*F)(ConcreteClass1&,ConcreteClass2&), bool symetric>
    void ignore()
    {
        this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass1), typeid(ConcreteClass2), &ignoreFn);
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass2), typeid(ConcreteClass1), &ignoreFn);
        }
    }

    virtual ResulT defaultFn(BaseClass& arg1, BaseClass& arg2);
    static ResulT ignoreFn(BaseClass& arg1, BaseClass& arg2);
    ResulT go(BaseClass &arg1,BaseClass &arg2);
    /// Return true if a pair of argument correspond to a callback function (different than ignoreFn)
    bool isSupported(BaseClass &arg1, BaseClass &arg2);
};

template <class BaseClass, typename ResulT>
class FnDispatcher : public BasicDispatcher<BaseClass, ResulT>
{
public:

    template <class ConcreteClass1, class ConcreteClass2,ResulT (*F)(ConcreteClass1&,ConcreteClass2&), bool symetric>
    void add()
    {
        struct Local
        {
            static ResulT trampoline(BaseClass &arg1,BaseClass &arg2)
            {
                return F(static_cast<ConcreteClass1 &> (arg1),
                        static_cast<ConcreteClass2 &> (arg2));
            }
            static ResulT trampolineR(BaseClass &arg1,BaseClass &arg2)
            {
                return trampoline (arg2, arg1);
            }
        };
        this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass1), typeid(ConcreteClass2), &Local::trampoline);
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::add(typeid(ConcreteClass2), typeid(ConcreteClass1), &Local::trampolineR);
        }
    }

    template <class ConcreteClass1, class ConcreteClass2, bool symetric>
    void ignore()
    {
        this->BasicDispatcher<BaseClass, ResulT>::ignore(typeid(ConcreteClass1), typeid(ConcreteClass2));
        if (symetric)
        {
            this->BasicDispatcher<BaseClass, ResulT>::ignore(typeid(ConcreteClass2), typeid(ConcreteClass1));
        }
    }
};


template <class BaseClass, typename ResulT>
class SingletonFnDispatcher : public FnDispatcher<BaseClass, ResulT>
{
protected:
    SingletonFnDispatcher();
public:
    static SingletonFnDispatcher<BaseClass, ResulT>* getInstance();
    /*
    template <class ConcreteClass1,class ConcreteClass2,ResulT (*F)(ConcreteClass1&,ConcreteClass2&), bool symetric>
    static void Add()
    {
    	getInstance()->add<ConcreteClass1,ConcreteClass2,F,symetric>();
    }

    template <class ConcreteClass1, class ConcreteClass2, bool symetric>
    static void Ignore()
    {
    	getInstance()->ignore<ConcreteClass1,ConcreteClass2,symetric>();
    }

    static ResulT Go(BaseClass &arg1,BaseClass &arg2)
    {
    	getInstance()->go(arg1,arg2);
    }
    */
};

} // namespace Common

} // namespace Components

} // namepsace Sofa

#endif
