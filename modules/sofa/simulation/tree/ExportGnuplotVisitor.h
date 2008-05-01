#ifndef SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H
#define SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif


#include <sofa/simulation/tree/Visitor.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

class InitGnuplotVisitor : public Visitor
{
public:
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(component::System* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "initGnuplot";
    }
};

class ExportGnuplotVisitor : public Visitor
{
public:
    ExportGnuplotVisitor( double time );
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(component::System* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "exportGnuplot";
    }
protected:
    double m_time;
};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
