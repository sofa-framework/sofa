#ifndef SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H
#define SOFA_SIMULATION_TREE_EXPORTGNUPLOTACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif


#include <sofa/simulation/common/Visitor.h>

namespace sofa
{

namespace simulation
{


class InitGnuplotVisitor : public simulation::Visitor
{
public:
    std::string gnuplotDirectory;

    InitGnuplotVisitor(std::string dir = std::string("")) : gnuplotDirectory(dir) {}

    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "initGnuplot";
    }
};

class ExportGnuplotVisitor : public simulation::Visitor
{
public:
    ExportGnuplotVisitor( double time );
    /// This method calls the fwd* methods during the forward traversal. You typically do not overload it.
    virtual Result processNodeTopDown(simulation::Node* node);

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const
    {
        return "exportGnuplot";
    }
protected:
    double m_time;
};

} // namespace simulation

} // namespace sofa

#endif
