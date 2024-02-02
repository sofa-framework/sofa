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
#pragma once
#include <sofa/component/linearsolver/direct/config.h>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/type/vector.h>


namespace sofa::component::linearsolver::direct
{

/**
 * A base wrapper class for all solvers based on the Eigen library
 */
struct SOFA_COMPONENT_LINEARSOLVER_DIRECT_API BaseEigenSolverProxy
{
    virtual ~BaseEigenSolverProxy(){}

    [[nodiscard]] virtual Eigen::ComputationInfo info() const = 0;

    template <class Real>
    using EigenVectorXdMap = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 1> >;

    virtual void solve(const EigenVectorXdMap<float>& b, EigenVectorXdMap<float>& x) const = 0;
    virtual void solve(const EigenVectorXdMap<double>& b, EigenVectorXdMap<double>& x) const = 0;

    template <class Real>
    using EigenSparseMatrix = Eigen::SparseMatrix<Real, Eigen::RowMajor>;

    template <class Real>
    using EigenSparseMatrixMap = Eigen::Map<EigenSparseMatrix<Real>>;

    virtual void analyzePattern(const EigenSparseMatrixMap<float>& a) = 0;
    virtual void analyzePattern(const EigenSparseMatrixMap<double>& a) = 0;

    virtual void factorize(const EigenSparseMatrixMap<float>& a) = 0;
    virtual void factorize(const EigenSparseMatrixMap<double>& a) = 0;
};


/**
 * This class is the bridge between an Eigen solver class and the abstact
 * class BaseEigenSolverProxy.
 *
 * Using this class, all Eigen solvers can inherit from the same base class,
 * hence runtime polymorphism is possible. Only a subset of the Eigen solver
 * methods are available.
 */
template <typename RealObject>
class EigenSolverWrapper : public BaseEigenSolverProxy
{
public:
    EigenSolverWrapper() : m_realObject() {}

    [[nodiscard]] Eigen::ComputationInfo info() const override
    {
        return m_realObject.info();
    }

    void solve(const EigenVectorXdMap<float>& b, EigenVectorXdMap<float>& x) const override
    {
        solve_t(b, x);
    }
    void solve(const EigenVectorXdMap<double>& b, EigenVectorXdMap<double>& x) const override
    {
        solve_t(b, x);
    }

    void analyzePattern(const EigenSparseMatrixMap<float>& a) override
    {
        analyzePattern_t(a);
    }
    void analyzePattern(const EigenSparseMatrixMap<double>& a) override
    {
        analyzePattern_t(a);
    }

    void factorize(const EigenSparseMatrixMap<float>& a) override
    {
        factorize_t(a);
    }
    void factorize(const EigenSparseMatrixMap<double>& a) override
    {
        factorize_t(a);
    }

private:
    RealObject m_realObject;

    template<class Scalar>
    void analyzePattern_t(const EigenSparseMatrixMap<Scalar>& a)
    {
        if constexpr (std::is_same_v<typename RealObject::Scalar, Scalar>)
        {
            m_realObject.analyzePattern(a);
        }
        else
        {
            msg_error("EigenSolverProxy") << "Mixing floating-point type: forbidden";
        }
    }

    template<class Scalar>
    void factorize_t(const EigenSparseMatrixMap<Scalar>& a)
    {
        if constexpr (std::is_same_v<typename RealObject::Scalar, Scalar>)
        {
            m_realObject.factorize(a);
        }
        else
        {
            msg_error("EigenSolverProxy") << "Mixing floating-point type: forbidden";
        }
    }

    template<class Scalar>
    void solve_t(const EigenVectorXdMap<Scalar>& b, EigenVectorXdMap<Scalar>& x) const
    {
        if constexpr (std::is_same_v<typename RealObject::Scalar, Scalar>)
        {
            x = m_realObject.solve(b);
        }
        else
        {
            msg_error("EigenSolverProxy") << "Mixing floating-point type: forbidden";
        }
    }
};


/**
 * A factory to create Eigen solvers from strings (the name of the ordering
 * method, and the name of the floating-point type)
 */
class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API EigenSolverFactory
{
public:
    struct OrderingMethodName
    {
        std::string name;
        std::string scalar;

        bool operator<(const OrderingMethodName& other) const
        {
            return name + scalar < other.name + other.scalar;
        }
    };

    template<typename EigenClass>
    void registerType(const std::string& orderingMethodName)
    {
        using Scalar = typename EigenClass::Scalar;
        m_registeredTypes[join<Scalar>(orderingMethodName)] = [](){ return new EigenSolverWrapper<EigenClass>(); };
    }

    template<class Scalar>
    BaseEigenSolverProxy* getObject(const std::string& orderingMethodName )
    {
        return m_registeredTypes[join<Scalar>(orderingMethodName)]();
    }

    template<class Scalar>
    bool hasObject(const std::string& orderingMethodName )
    {
        return m_registeredTypes.find(join<Scalar>(orderingMethodName)) != m_registeredTypes.end();
    }

    [[nodiscard]]
    sofa::type::vector<OrderingMethodName> registeredObjects() const
    {
        sofa::type::vector<OrderingMethodName> list;
        list.reserve(m_registeredTypes.size());
        for(const auto& [key, _] : m_registeredTypes)
        {
            list.push_back(key);
        }
        return list;
    }

private:
    std::map<OrderingMethodName, std::function<BaseEigenSolverProxy*()>> m_registeredTypes;

    static OrderingMethodName join(const std::string& orderingMethodName, const std::string& scalarType)
    {
        return {orderingMethodName, scalarType};
    }

    template<class Scalar>
    OrderingMethodName join(const std::string& orderingMethodName)
    {
        return join(orderingMethodName, defaulttype::DataTypeInfo<Scalar>::name());
    }
};


/**
 * Base class for specialized Eigen solver factories. It is a singleton with a
 * single instance of EigenSolverFactory. It uses the CRTP to define a single
 * instance of EigenSolverFactory for each derived type.
 */
template <typename Derived>
class BaseMainEigenSolverFactory
{
protected:
    inline static std::mutex s_mutex {};
    static EigenSolverFactory& getFactory()
    {
        static EigenSolverFactory factory;
        return factory;
    }

public:

    template<class Scalar>
    using EigenSparseMatrix = Eigen::SparseMatrix<Scalar>;

    template<class Scalar>
    static BaseEigenSolverProxy* getSolver(const std::string & orderingMethodName )
    {
        std::lock_guard lock(s_mutex);
        return getFactory().getObject<Scalar>(orderingMethodName);
    }

    template<class Scalar>
    static bool hasSolver(const std::string & orderingMethodName )
    {
        std::lock_guard lock(s_mutex);
        return getFactory().hasObject<Scalar>(orderingMethodName);
    }

    static sofa::type::vector<EigenSolverFactory::OrderingMethodName> registeredSolvers()
    {
        std::lock_guard lock(s_mutex);
        return getFactory().registeredObjects();
    }
};


/**
 * Singleton factory dedicated to LDLT solvers
 */
class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API MainSimplicialLDLTFactory : public BaseMainEigenSolverFactory<MainSimplicialLDLTFactory>
{
public:
    ~MainSimplicialLDLTFactory();

    template<typename OrderingMethodType, class ScalarType>
    static void registerSolver(const std::string& orderingMethodName)
    {
        std::lock_guard lock(s_mutex);
        getFactory().registerType<
            Eigen::SimplicialLDLT<EigenSparseMatrix<ScalarType>, Eigen::Lower, OrderingMethodType> >(orderingMethodName);
    }
};

/**
 * Singleton factory dedicated to LLT solvers
 */
class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API MainSimplicialLLTFactory : public BaseMainEigenSolverFactory<MainSimplicialLLTFactory>
{
public:
    ~MainSimplicialLLTFactory();

    template<typename OrderingMethodType, class ScalarType>
    static void registerSolver(const std::string& orderingMethodName)
    {
        std::lock_guard lock(s_mutex);
        getFactory().registerType<
            Eigen::SimplicialLLT<EigenSparseMatrix<ScalarType>, Eigen::Lower, OrderingMethodType> >(orderingMethodName);
    }
};

/**
 * Singleton factory dedicated to QT solvers
 */
class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API MainQRFactory : public BaseMainEigenSolverFactory<MainQRFactory>
{
public:
    ~MainQRFactory();

    template<typename OrderingMethodType, class ScalarType>
    static void registerSolver(const std::string& orderingMethodName)
    {
        std::lock_guard lock(s_mutex);
        getFactory().registerType<
            Eigen::SparseQR<EigenSparseMatrix<ScalarType>, OrderingMethodType> >(orderingMethodName);
    }
};

/**
 * Singleton factory dedicated to LU solvers
 */
class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API MainLUFactory : public BaseMainEigenSolverFactory<MainLUFactory>
{
public:
    ~MainLUFactory();

    template<typename OrderingMethodType, class ScalarType>
    static void registerSolver(const std::string& orderingMethodName)
    {
        std::lock_guard lock(s_mutex);
        getFactory().registerType<
            Eigen::SparseLU<EigenSparseMatrix<ScalarType>, OrderingMethodType> >(orderingMethodName);
    }
};


}
