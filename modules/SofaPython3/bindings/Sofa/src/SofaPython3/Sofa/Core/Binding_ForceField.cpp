#include <pybind11/pybind11.h>
#include <pybind11/detail/init.h>
#include "Binding_BaseObject.h"
#include "Binding_ForceField.h"
#include "DataHelper.h"

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/simulation/Node.h>

PYBIND11_DECLARE_HOLDER_TYPE(TForceField,
                             sofapython3::py_shared_ptr<TForceField>, true)


namespace sofapython3
{
using sofa::core::objectmodel::ComponentState;
using sofa::core::behavior::MechanicalState;
using sofa::core::behavior::ForceField;
using sofa::defaulttype::Vec3dTypes;

class ForceField_Trampoline  : public ForceField<Vec3dTypes>, public PythonTrampoline
{
public:
    ForceField_Trampoline() {}
    virtual ~ForceField_Trampoline(){}

    virtual void init() override
    {
        Inherit1::init();

        if (!mstate.get())
            mstate.set(dynamic_cast< MechanicalState<DataTypes>* >(getContext()->getMechanicalState()));

        if(!mstate.get())
            throw py::type_error("Missing mechanical state.");

        PYBIND11_OVERLOAD(void, ForceField, init,);
    }

    virtual void addForce(const MechanicalParams* mparams,  DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override
    {
        BaseData* xx = const_cast<BaseData*>(static_cast<const BaseData*>(&x));
        BaseData* vv = const_cast<BaseData*>(static_cast<const BaseData*>(&v));
        PYBIND11_OVERLOAD_PURE(void, ForceField, addForce, py::none(), toPython(&f,true), toPython(xx,true), toPython(vv,true));
    }

    virtual void addDForce(const MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx ) override
    {
        BaseData* dxx = const_cast<BaseData*>(static_cast<const BaseData*>(&dx));
        PYBIND11_OVERLOAD_PURE(void, ForceField, addDForce,
                               toPython(&df,true), toPython(dxx,true),
                               py::cast(mparams->kFactor()), py::cast(mparams->bFactor()));
    }

    /*virtual void addMBKdx(const MechanicalParams* mparams, MultiVecDerivId dfId) override
    {
        PYBIND11_OVERLOAD_PURE(void, ForceField, addMBKdx, py::none(), py::none() );
    }*/
    virtual void addKToMatrix(const MechanicalParams* mparams, const MultiMatrixAccessor* dfId) override
    {
        PYBIND11_OVERLOAD_PURE(void, ForceField, addKToMatrix, py::none(), py::none() );
    }

    virtual void updateForceMask() override
    {
        #ifdef SOFA_USE_MASK
            PYBIND11_OVERLOAD_PURE(void, ForceField, updateForceMask,);
        #else
           PYBIND11_OVERLOAD(void, ForceField, updateForceMask,);
        #endif
    }
    virtual SReal getPotentialEnergy( const MechanicalParams* mparams,
                                      const DataVecCoord& x) const override {}


    virtual std::string getClassName() const override
    {
        return pyobject->ob_type->tp_name;
    }
};


void moduleAddForceField(py::module &m) {
    py::class_<ForceField<Vec3dTypes>,
            ForceField_Trampoline, BaseObject,
            py_shared_ptr<ForceField<Vec3dTypes>>> f(m, "ForceField",
                                                     py::dynamic_attr(),
                                                     py::multiple_inheritance());

    f.def(py::init([](py::args& args, py::kwargs& kwargs)
    {
              ForceField_Trampoline* c = new ForceField_Trampoline();
              c->f_listening.setValue(true);

              if(args.size() != 0)
              {
                  if(args.size()==1) c->setName(py::cast<std::string>(args[0]));
                  else throw py::type_error("Only one un-named arguments can be provided.");
              }

              py::object cc = py::cast(c);
              for(auto& kv : kwargs)
              {
                  std::string key = py::cast<std::string>(kv.first);
                  py::object value = py::reinterpret_borrow<py::object>(kv.second);
                  if( key == "name")
                  {
                      if( args.size() != 0 )
                      {
                          throw py::type_error("The name is setted twice as a "
                          "named argument='"+py::cast<std::string>(value)+"' and as a"
                          "positional argument='"+py::cast<std::string>(args[0])+"'.");
                      }
                  }
                  BindingBase::SetAttr(cc, key, value);
              }
              return c;
          }));
}



}
