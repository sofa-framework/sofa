#include "Binding_Vec.h"
#include <functional>
#include <type_traits>
#include <pybind11/operators.h>

#define BINDING_VEC_MAKE_NAME(N, T)                                            \
    std::string(std::string("Vec") + std::to_string(N) + typeid(T).name())

namespace pyVec {
template <int N, class T> std::string __str__(const Vec<N, T> &v, bool repr)
{
    std::stringstream s ;
    s.imbue(std::locale("C"));
    s << ((repr) ? (BINDING_VEC_MAKE_NAME(N, T) + "(") : ("("));
    s << v[0];
    for (size_t i = 1; i < v.size(); ++i)
        s <<  ", " << v[i];
    s <<  ")";
    return s.str();
}
} // namespace pyVec

template<typename T> T convertThenCast(const py::object& o);
template<> double convertThenCast<double>(const py::object& o)
{
    return py::cast<double>( py::float_(o) );
}
template<> int convertThenCast<int>(const py::object& o)
{
    return py::cast<int>( py::int_(o) );
}

template<class VecClass>
void setFromSequence(VecClass& v, const py::list& l)
{
    for (size_t i = 0; i < VecClass::size(); ++i)
    {
        v[i] = convertThenCast<typename VecClass::value_type>(l[i]);
    }
}

template<class VecClass>
VecClass* createFromSequence(const py::list& l)
{
    VecClass *v = new VecClass();
    setFromSequence(*v, l);
    return v;
}

template<class VecClass, int N>
py::tuple createTupleFrom(const VecClass& s)
{
    py::tuple t(N);
    for(unsigned int i=0;i<N;i++) { t[i] = s[i]; }
    return t;
}

template<class VecClass, int N>
void setFromPartialSequence(const VecClass& s, py::list t)
{
    for(unsigned int i=0;i<N;i++) { t[i] = s[i]; }
}

template <int N, class T>
py::class_<Vec<N,T>> addVec(py::module &m)
{
    typedef Vec<N, T> VecClass;
    py::class_<Vec<N, T>> p(m, BINDING_VEC_MAKE_NAME(N, T).c_str());

    p.def(py::init<>());                 // empty ctor
    p.def(py::init<const VecClass &>()); // copy ctor

    p.def(py::init([](py::args l) {
              /// Sequence based version
              if(l.size() == 1 && py::isinstance<py::list>(l[0]))
                return std::unique_ptr<VecClass>( createFromSequence<VecClass>(l[0]) );

              /// individual entries.
              return std::unique_ptr<VecClass>( createFromSequence<VecClass>(l) );
          }));

    p.def("set", [](VecClass &v, py::args l) {
        setFromSequence(v, l);
    });

    p.def("set", [](VecClass &v, py::list l) {
        setFromSequence(v, l);
    });

    p.def("__getitem__", [](const VecClass &v, size_t i) {
        if (i >= v.size())
            throw py::index_error();
        return v[i];
    });

    p.def("__setitem__", [](VecClass &v, size_t i, T d) {
        if (i >= v.size())
            throw py::index_error();
        T &val = v[i];
        val = d;
        return val;
    });

    /// Iterator protocol
    static size_t value = 0;
    p.def("__iter__", [](VecClass &v) {
        value = 0;
        return v;
    });
    p.def("__next__", [](VecClass &v) {
        if (value == v.size())
            throw py::stop_iteration();
        else
            return v[value++];
        return v[value];
    });

    p.def(py::self != py::self)
            .def(py::self == py::self)
            .def(py::self * py::self)
            .def(py::self + py::self)
            .def(py::self += py::self)
            .def(py::self - py::self)
            .def(py::self -= py::self);

    p.def("__mul__", [](double d, const VecClass &v) { return v * d; });
    p.def("__mul__", [](int d, const VecClass &v) { return v * d; });

    p.def(py::self * double());
    p.def(py::self * int());
    p.def("__imul__", [](VecClass &v, const float d) { v.eqmulscalar(d); return v; });
    p.def("__imul__", [](VecClass &v, const int d) { v.eqmulscalar(d); return v; });

    p.def(py::self / double());/// Add individual x,y,z
    if(N>=1){
        p.def_property("x",
                       [](VecClass &v) { return v[0]; },
        [](VecClass &v, double x) { v[0] = x; });
    }
    if(N>=2){
        p.def_property("y",
                       [](VecClass &v) { return v[1]; },
        [](VecClass &v, double y) { v[1] = y; });
    }
    if(N>=3){
        p.def_property("z",
                       [](VecClass &v) { return v[2]; },
        [](VecClass &v, double z) { v[2] = z; });
    }
    if(N>=4){
        p.def_property("w",
                       [](VecClass &v) { return v[3]; },
        [](VecClass &v, double w) { v[3] = w; });
    }
    p.def(py::self / int());
    p.def("__idiv__", [](VecClass &v, const double d) { v.eqdivscalar(d); return v; });
    p.def("__idiv__", [](VecClass &v, const int d) { v.eqdivscalar(d); return v; });

    p.def("__str__", [](VecClass &v) { return pyVec::__str__(v); });
    p.def("__repr__", [](VecClass &v) { return pyVec::__str__(v, true); });

    p.def("fill", &VecClass::fill, "r"_a);
    p.def("clear", &VecClass::clear);
    p.def("norm", &VecClass::norm);
    p.def("norm2", &VecClass::norm2);
    p.def("lNorm", &VecClass::lNorm, "l"_a);
    p.def("normalize", (bool (VecClass::*)(T)) & VecClass::normalize,
          "threshold"_a = std::numeric_limits<T>::epsilon());
    p.def("normalized", &VecClass::normalized);
    p.def("sum", &VecClass::sum);
    p.def("dot", [](const VecClass &self, const VecClass &b) {
        return sofa::defaulttype::dot(self, b);
    });



    /// Add paired elements.
    if(N>=2){
        p.def_property("xy",
                       &createTupleFrom<VecClass, 2>,
                       &setFromPartialSequence<VecClass, 2>);
    }
    if(N>=3){
        p.def_property("xyz",
                       &createTupleFrom<VecClass, 3>,
                       &setFromPartialSequence<VecClass, 3>);
    }
    if(N>=4){
        p.def_property("xyzw",
                       &createTupleFrom<VecClass, 4>,
                       &setFromPartialSequence<VecClass, 4>);
    }
    return p;
}

template<class T>
T addCross(T p)
{
    p.def("cross", [](typename T::type& a, typename T::type& b)
    {
        return sofa::defaulttype::cross(a,b);
    });
    return p;
}

template<class T>
void addVectorsFor(py::module& m)
{
    addVec<1, T>(m);
    addCross( addVec<2, T>(m) );
    addCross( addVec<3, T>(m) );
    addVec<4, T>(m);
    addVec<5, T>(m);
    addVec<6, T>(m);
    addVec<7, T>(m);
    addVec<8, T>(m);
    addVec<9, T>(m);
    addVec<10, T>(m);
    addVec<11, T>(m);
    addVec<12, T>(m);
}

void moduleAddVec(py::module &m)
{
    addVectorsFor<int>(m);
    addVectorsFor<double>(m);
}
