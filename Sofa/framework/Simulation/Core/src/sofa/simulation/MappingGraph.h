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

/**
 * @file MappingGraph.h
 * @brief Implements a mapping graph structure for simulating mechanical systems in SOFA.
 *
 * This header defines the core structures necessary to represent how various
 * physical components and behavioral states are linked together during simulation.
 * It allows for top-down (prerequisite check) and bottom-up (dependency accumulation)
 * traversals of a component hierarchy, ensuring that inputs are calculated correctly
 * before they are needed by dependent nodes or mappings.
 */

#pragma once
#include <sofa/simulation/Node.h>
#include <sofa/simulation/config.h>
#include <sofa/simulation/mappinggraph/ComponentGroupMappingGraphNode.h>
#include <sofa/simulation/mappinggraph/CallableVisitor.h>

#include <queue>

namespace sofa::simulation
{
class TaskScheduler;

enum class VisitorApplication
{
    ALL_NODES,
    ONLY_MAPPED_NODES,
    ONLY_MAIN_NODES
};

/**
 * @brief Represents the overall mechanical simulation graph structure (Mapping Graph).
 * 
 * This class builds and manages a dependency graph connecting all major components 
 * (MechanicalStates, Mappings, ForceFields, Masses) within an SOFA scene. 
 * It allows for systematic traversal (Top-down/Bottom-up) to determine the correct 
 * order of calculation required during simulation initialization or execution.
 */
class SOFA_SIMULATION_CORE_API MappingGraph
{
public:
    using MappingInputs = type::vector<core::behavior::BaseMechanicalState*>;

    /**
     * @brief Container struct holding lists of all potential input components 
     * collected from a scene context.
     */
    struct SOFA_SIMULATION_CORE_API InputLists
    {
        sofa::type::vector<core::behavior::BaseMechanicalState*> mechanicalStates; ///< All Mechanical State inputs.
        sofa::type::vector<core::BaseMapping*> mappings;                       ///< All Mapping components.
        sofa::type::vector<core::behavior::BaseForceField*> forceFields;       ///< All Force Field components.
        sofa::type::vector<core::behavior::BaseMass*> masses;                   ///< All Mass components.

        /**
         * @brief Creates InputLists from a context pointer.
         * @param node The SOFA object model base context associated with the component list.
         * @return A populated InputLists structure.
         */
        static InputLists makeFromNode(core::objectmodel::BaseContext* node);

        /**
         * @brief Creates InputLists from a shared pointer context.
         * @param node The SOFA object model base context smart pointer.
         * @return A populated InputLists structure.
         */
        static InputLists makeFromNode(core::objectmodel::BaseContext::SPtr node) { return makeFromNode(node.get()); }
    };

    /**
     * @brief Default constructor initializes an empty graph.
     */
    MappingGraph() = default;

    /**
     * @brief Constructs the graph using pre-collected input lists.
     * @param input The list of all components found in the scene.
     */
    explicit MappingGraph(const InputLists& input);

    /**
     * @brief Constructs the graph by traversing a starting node in the SOFA object model context.
     * @param node The root context node from which to build the graph.
     */
    explicit MappingGraph(core::objectmodel::BaseContext* node);

    void clear();

    /**
     * @brief Returns the root node used during the initial construction of the graph.
     * @return A pointer to the root object model context.
     */
    [[nodiscard]] core::objectmodel::BaseContext* getRootNode() const;

    /**
     * @brief Gets the list of all main mechanical states that are not used as outputs 
     * in any mapping (i.e., they are root inputs).
     * @return Const reference to the vector of non-mapped mechanical state pointers.
     */
    [[nodiscard]] const sofa::type::vector<core::behavior::BaseMechanicalState*>& getMainMechanicalStates() const;

    /**
     * @brief Recursively finds top-most mechanical states that are unmapped but serve as 
     * inputs to a mapping involving the provided state.
     * 
     * This search is recursive, handling multiple levels of dependencies.
     * 
     * @param state The starting mechanical state for dependency checking.
     * @return A list of top-most unmapped mechanical states required by `state`.
     */
    MappingInputs getTopMostMechanicalStates(core::behavior::BaseMechanicalState* state) const;

    /**
     * @brief Recursively finds top-most mechanical states that are unmapped but serve as 
     * inputs to a mapping involving the mechanical states associated with a given accessor.
     * 
     * This search is recursive, handling multiple levels of dependencies.
     * 
     * @param stateAccessor The starting state accessor for dependency checking.
     * @return A list of top-most unmapped mechanical states required by `stateAccessor`.
     */
    MappingInputs getTopMostMechanicalStates(core::behavior::StateAccessor* stateAccessor) const;

    /**
     * @brief Checks if any mapping exists anywhere in the graph structure.
     * @return True if at least one mapping is present, false otherwise.
     */
    [[nodiscard]] bool hasAnyMapping() const;

    /**
     * @brief Determines if a specific mechanical state is an output of any mapping node 
     * connected to the graph.
     * @param mstate The mechanical state to check.
     * @return True if `mstate` is an output, false otherwise.
     */
    bool hasAnyMappingInput(core::behavior::BaseMechanicalState* mstate) const;

    /**
     * @brief Determines if the mechanical states associated with a component are outputs 
     * of any mapping node connected to the graph.
     * @param stateAccessor The state accessor for the component to check.
     * @return True if the associated states are mapped output, false otherwise.
     */
    bool hasAnyMappingInput(core::behavior::StateAccessor* stateAccessor) const;

    /**
     * @brief Calculates the total number of degrees of freedom (DoF) contributed 
     * by all main mechanical states in the graph.
     * @return The sum of DoFs across all primary states.
     */
    [[nodiscard]] sofa::Size getTotalNbMainDofs() const;

    /**
     * @brief Finds the global matrix indices (row/column) where a specific state 
     * contributes its degrees of freedom.
     * @param mstate The mechanical state.
     * @return A pair representing (global row index, global column index).
     */
    type::Vec2u getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* mstate) const;

    /**
     * @brief Finds the global matrix indices where two specified states contribute 
     * their degrees of freedom. (Used for cross-axis checks).
     * @param a The first mechanical state.
     * @param b The second mechanical state.
     * @return A pair representing (global row index, global column index) for the combined contribution.
     */
    type::Vec2u getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* a,
                                              core::behavior::BaseMechanicalState* b) const;

    /**
     * @brief Retrieves all mapping components that depend on (receive input from) 
     * the provided mechanical state. This is used for bottom-up dependency checks.
     * @param mstate The mechanical state acting as an input source.
     * @return A vector of shared pointers to dependent BaseMapping nodes.
     */
    sofa::type::vector<core::BaseMapping*> getBottomUpMappingsFrom(
        core::behavior::BaseMechanicalState*) const;


    // ------------------------------------------------------------------
    // Top-down traversal: roots (unmapped states) → leaves (components).
    //
    // Ensures that a BaseMechanicalState is only processed after all mappings 
    // that produce it as output have been processed, and similarly for mappings 
    // and leaf components. This guarantees correct dependency order.
    void traverseTopDown(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    template<class Callable>
    void traverseTopDown_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverseTopDown(visitor, scope);
    }

    // ------------------------------------------------------------------
    // Bottom-up traversal: leaves → roots.
    //
    // Provides the reverse dependency ordering check, ensuring that prerequisite 
    // states are processed before the components that require them.
    void traverseBottomUp(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    template<class Callable>
    void traverseBottomUp_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverseBottomUp(visitor, scope);
    }

    /**
     * @brief Visit and process component groups without any specific order.
     * @param visitor The concrete visitor implementation.
     */
    void traverseComponentGroups(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    /**
     * @brief Visit and process component groups without any specific order, optionally coordinating
     * with a TaskScheduler to manage execution parallelism.
     * @param visitor The concrete visitor implementation.
     * @param taskScheduler Optional scheduler instance for tasks requiring explicit ordering.
     */
    void traverseComponentGroups(MappingGraphVisitor& visitor, TaskScheduler* taskScheduler) const;

    /**
     * @brief Checks if the graph has been successfully built and analyzed.
     * @return True if building is complete, false otherwise.
     */
    [[nodiscard]] bool isBuilt() const;

    // ------------------------------------------------------------------
    // Graph construction methods:
    // ------------------------------------------------------------------

    /**
     * @brief Builds the mapping graph using a provided set of input components.
     * @param input The collected list of all potential SOFA components.
     */
    void build(const InputLists& input);

    /**
     * @brief Builds the mapping graph by traversing and collecting components 
     * starting from a specific root node in the object model hierarchy.
     * @param rootNode The starting context node.
     */
    void build(core::objectmodel::BaseContext* rootNode);

private:
    ///< Root node used to start graph exploration during construction.
    core::objectmodel::BaseContext* m_rootNode { nullptr };

    bool m_isBuilt = false; ///< Flag indicating if the graph structure is finalized.
    bool m_hasAnyMapping = false; ///< Flag indicating if any mapping exists in the graph.

    sofa::Size m_totalNbMainDofs {}; ///< Total number of primary degrees of freedom managed by the system.

    /**
     * @brief Map storing the global indices (row, column) for each main mechanical state's contribution matrix block.
     */
    std::map<core::behavior::BaseMechanicalState*, type::Vec2u > m_positionInGlobalMatrix;

    // Graph ownership structures:
    std::vector<BaseMappingGraphNode::SPtr> m_allNodes; ///< All nodes in the graph.
    sofa::type::vector<core::behavior::BaseMechanicalState*> m_rootStates {}; ///< List of initial, unmapped mechanical states (graph roots).
    std::unordered_map<core::behavior::BaseMechanicalState*, BaseMappingGraphNode*> m_stateIndex; ///< Quick lookup for a state's node.
    std::vector<std::pair<
        std::vector<core::behavior::BaseMechanicalState::SPtr>,
        ComponentGroupMappingGraphNode::SPtr>> m_groupIndex; ///< Indexing mechanism for group nodes.

    // ------------------------------------------------------------------

    std::queue<BaseMappingGraphNode*> prepareRootForTraversal() const;

    /**
     * @brief Performs a breadth-first search (BFS) traversal, processing nodes in dependency order.
     * 
     * This static helper method is used for both top-down and bottom-up traversals.
     * 
     * @param ready The queue of nodes that are currently ready to be visited/processed.
     */
    template<class Callable>
    static void processQueue(std::queue<BaseMappingGraphNode*>& ready, const Callable& f)
    {
        while (!ready.empty())
        {
            BaseMappingGraphNode* current = ready.front();
            ready.pop();

            f(current);

            for (auto& child : current->m_children)
            {
                --(child->m_pendingCount);
                if (child->m_pendingCount == 0)
                {
                    ready.push(child.get());
                }
            }
        }
    }

    /**
     * @brief Locates or creates a component group node encompassing the given set of states.
     * @param states The mechanical states belonging to the group.
     * @return A shared pointer to the found/created ComponentGroupMappingGraphNode.
     */
    ComponentGroupMappingGraphNode::SPtr findGroupNode(const std::vector<core::behavior::BaseMechanicalState::SPtr>& states);

    /**
     * @brief Locates or creates a component group node for a single state's context.
     * @param state The mechanical state defining the scope of the group.
     * @return A shared pointer to the found/created ComponentGroupMappingGraphNode.
     */
    ComponentGroupMappingGraphNode::SPtr findInGroupNodes(const core::behavior::BaseMechanicalState::SPtr state);

    /**
     * @brief Finds the graph node corresponding to a raw mechanical state pointer.
     * @param raw The mechanical state raw pointer.
     * @return Pointer to the associated BaseMappingGraphNode, or nullptr if not found.
     */
    BaseMappingGraphNode* findStateNode(core::behavior::BaseMechanicalState* raw) const;

    /**
     * @brief Adds a directed edge between two nodes in the graph structure (from -> to).
     * 
     * Both 'from' and 'to' pointers are added to the respective parent/child lists, 
     * ensuring that the graph manages ownership of all nodes via `SPtr`.
     * @param from The starting node.
     * @param to   The ending node.
     */
    static void addEdge(BaseMappingGraphNode* from, BaseMappingGraphNode* to);
};


/**
 * @brief Manages Jacobian matrix contributions for a single mechanical state.
 * 
 * This class holds and retrieves the Jacobian matrices calculated during 
 * graph construction. It maps input states (which require Jacobians) to their 
 * corresponding Jacobian calculation object.
 * 
 * @tparam JacobianMatrixType The concrete type used for the Jacobian matrix structure.
 */
template<class JacobianMatrixType>
class MappingJacobians
{
    const core::behavior::BaseMechanicalState& m_mappedState; ///< The mechanical state whose Jacobians are being managed.

    /**
     * @brief Map from an input mechanical state to its calculated Jacobian matrix.
     */
    std::map< core::behavior::BaseMechanicalState*, std::shared_ptr<JacobianMatrixType> > m_map;

public:
    /**
     * @brief Deleted constructor enforces usage via the parameterized constructor.
     */
    MappingJacobians() = delete;

    /**
     * @brief Constructs the Jacobian manager for a specific mechanical state.
     * @param mappedState Reference to the mechanical state being managed.
     */
    MappingJacobians(const core::behavior::BaseMechanicalState& mappedState) : m_mappedState(mappedState) {}

    /**
     * @brief Associates a calculated Jacobian matrix with a top-most parent state.
     * @param jacobian The shared pointer to the Jacobian matrix.
     * @param topMostParent The mechanical state that uses this Jacobian as input.
     */
    void addJacobianToTopMostParent(std::shared_ptr<JacobianMatrixType> jacobian, core::behavior::BaseMechanicalState* topMostParent)
    {
        m_map[topMostParent] = jacobian;
    }

    /**
     * @brief Retrieves the Jacobian matrix associated with a given mechanical state.
     * @param mstate The state whose Jacobian is requested.
     * @return A shared pointer to the Jacobian, or nullptr if not found.
     */
    std::shared_ptr<JacobianMatrixType> getJacobianFrom(core::behavior::BaseMechanicalState* mstate) const
    {
        const auto it = m_map.find(mstate);
        if (it != m_map.end())
            return it->second;
        return nullptr;
    }
};

}
