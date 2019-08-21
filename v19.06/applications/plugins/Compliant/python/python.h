#ifndef COMPLIANT_MISC_PYTHON_H
#define COMPLIANT_MISC_PYTHON_H

#include <sofa/defaulttype/DataTypeInfo.h>

template<class T>
struct opaque {

    opaque() : data(0) { }
    
    T* data;
    
    friend std::istream& operator>>(std::istream& in , const opaque& ) {
        // TODO emit warning
        return in;
    }

    friend std::ostream& operator<<(std::ostream& out, const opaque& ) {
        return out << "#<opaque>";
    }

};


namespace sofa {
namespace defaulttype {

template<class T>
struct DataTypeInfo< opaque<T> > {
    // can't believe there is no default-impl i can override

    typedef opaque<T> DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    static const bool CopyOnWrite = false;
    static const bool ValidInfo = false;
    static const bool FixedSize = true;
    static const bool ZeroConstructor = true;
    static const bool SimpleCopy = true;
    static const bool SimpleLayout = true;
    static const bool Integer = false;
    static const bool Scalar = false;
    static const bool Text = false;
    static const bool Container = false;        

    static std::size_t size() { return 1; }
    static std::size_t size(const DataType& /*data*/) { return 1; }
    
    static std::size_t byteSize() { return sizeof(opaque<T>); }    
    static bool setSize(DataType& /*data*/, std::size_t /*size*/) { return false; }

    template <typename U>
    static void getValue(const DataType& /*data*/, std::size_t /*index*/, U& /*value*/)
    {
    }

    template<typename U>
    static void setValue(DataType& /*data*/, std::size_t /*index*/, const U& /*value*/)
    {
    }

    static void getValueString(const DataType& /*data*/, std::size_t /*index*/, std::string& /*value*/)
    {
    }

    static void setValueString(DataType& /*data*/, std::size_t /*index*/, const std::string& /*value*/)
    {
    }

    
    static const void* getValuePtr(const DataType& type)
    {
        return (void*) &type;
    }

    static void* getValuePtr(DataType& type)
    {
        return (void*) &type;
    }

    static const char* name() { return "opaque"; }
};

}
}







#endif
