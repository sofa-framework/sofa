#include <sofa/simulation/MappingGraph2.h>

#include <sofa/simulation/task/ParallelForEach.h>

namespace sofa::simulation
{

MappingGraph2::InputLists MappingGraph2::InputLists::makeFromNode(core::objectmodel::BaseContext* node)
{
    InputLists inputLists;
    if (node)
    {
        node->getObjects(inputLists.mechanicalStates, core::objectmodel::BaseContext::SearchDirection::SearchDown);
        node->getObjects(inputLists.mappings, core::objectmodel::BaseContext::SearchDirection::SearchDown);
        node->getObjects(inputLists.forceFields, core::objectmodel::BaseContext::SearchDirection::SearchDown);
        node->getObjects(inputLists.masses, core::objectmodel::BaseContext::SearchDirection::SearchDown);
    }
    return inputLists;
}

MappingGraph2::MappingGraph2(const InputLists& input)
{
    build(input);
}

MappingGraph2::MappingGraph2(core::objectmodel::BaseContext* node)
{
    build(node);
}

core::objectmodel::BaseContext* MappingGraph2::getRootNode() const
{
    return m_rootNode;
}

const sofa::type::vector<core::behavior::BaseMechanicalState*>&
MappingGraph2::getMainMechanicalStates() const
{
    return m_roots;
}
MappingGraph2::MappingInputs MappingGraph2::getTopMostMechanicalStates(
    core::behavior::BaseMechanicalState* state) const
{
    auto* sn = findStateNode(state);
    if (sn)
    {
        struct CollectInput : public MappingGraphVisitor
        {
            void visit(core::behavior::BaseMechanicalState& state) override
            {
                inputs.push_back(&state);
            }

            MappingInputs inputs;
        } visitor;

        for (auto& node : m_allNodes)
        {
            node->m_pendingCount = 0;
        }

        std::queue<BaseMappingGraphNode*> nodes;
        nodes.push(sn);

        while (!nodes.empty())
        {
            BaseMappingGraphNode* current = nodes.front();
            nodes.pop();

            ++(current->m_pendingCount);

            if (current->m_parents.empty())
            {
                current->accept(visitor);
            }

            for (auto& parent : current->m_parents)
            {
                if (parent->m_pendingCount == 0)
                {
                    nodes.push(parent.get());
                }
            }
        }

        return visitor.inputs;
    }

    return {};
}

bool MappingGraph2::hasAnyMapping() const
{
    return m_hasAnyMapping;
}
bool MappingGraph2::hasAnyMappingInput(core::behavior::BaseMechanicalState* mstate) const
{
    if (m_rootNode == nullptr)
    {
        msg_error("MappingGraph") << "Graph is not built yet";
        return false;
    }

    if (mstate == nullptr)
    {
        msg_error("MappingGraph") << "Requested mechanical state is not valid : cannot get its position in the global matrix";
        return false;
    }

    //only main (non mapped) mechanical states are in this map
    return !m_positionInGlobalMatrix.contains(mstate);
}

sofa::Size MappingGraph2::getTotalNbMainDofs() const
{
    return m_totalNbMainDofs;
}

type::Vec2u MappingGraph2::getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* mstate) const
{
    if (m_rootNode == nullptr)
    {
        msg_error("MappingGraph") << "Graph is not built yet";
        return type::Vec2u{};
    }

    if (mstate == nullptr)
    {
        msg_error("MappingGraph") << "Requested mechanical state is not valid : cannot get its position in the global matrix";
        return type::Vec2u{};
    }

    if (const auto it = m_positionInGlobalMatrix.find(mstate); it != m_positionInGlobalMatrix.end())
        return it->second;

    msg_error("MappingGraph") << "Requested mechanical state (" << mstate->getPathName() <<
        ") is probably mapped or unknown from the graph: only main mechanical states have an associated submatrix in the global matrix";
    return type::Vec2u{};
}

type::Vec2u MappingGraph2::getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* a,
                                                     core::behavior::BaseMechanicalState* b) const
{
    const auto pos_a = getPositionInGlobalMatrix(a);
    const auto pos_b = getPositionInGlobalMatrix(b);
    return {pos_a[0], pos_b[1]};
}

sofa::type::vector<core::BaseMapping*> MappingGraph2::getBottomUpMappingsFrom(
    core::behavior::BaseMechanicalState* state) const
{
    auto* sn = findStateNode(state);
    if (sn)
    {
        struct CollectMapping : public MappingGraphVisitor
        {
            void visit(core::BaseMapping& mapping) override
            {
                mappings.push_back(&mapping);
            }

            sofa::type::vector<core::BaseMapping*> mappings;
        } visitor;

        for (auto& node : m_allNodes)
        {
            node->m_pendingCount = 0;
        }

        std::queue<BaseMappingGraphNode*> nodes;
        nodes.push(sn);

        while (!nodes.empty())
        {
            BaseMappingGraphNode* current = nodes.front();
            nodes.pop();

            ++(current->m_pendingCount);

            if (current->m_parents.empty())
            {
                current->accept(visitor);
            }

            for (auto& parent : current->m_parents)
            {
                if (parent->m_pendingCount == 0)
                {
                    nodes.push(parent.get());
                }
            }
        }

        return visitor.mappings;
    }

    return {};
}

void MappingGraph2::traverseTopDown(MappingGraphVisitor& visitor) const
{
    // pending count = number of parents not yet visited.
    std::queue<BaseMappingGraphNode*> ready;
    for (auto& node : m_allNodes)
    {
        node->m_pendingCount = static_cast<int>(node->m_parents.size());
        if (node->m_pendingCount == 0)
        {
            ready.push(node.get());
        }
    }

    processQueue(ready, visitor, /*topDown=*/true);
}

void MappingGraph2::traverseBottomUp(MappingGraphVisitor& visitor) const
{
    std::queue<BaseMappingGraphNode*> ready;
    for (auto& node : m_allNodes)
    {
        node->m_pendingCount = static_cast<int>(node->m_parents.size());
        if (node->m_pendingCount == 0)
        {
            ready.push(node.get());
        }
    }

    sofa::type::vector<BaseMappingGraphNode*> nodes;
    while (!ready.empty())
    {
        BaseMappingGraphNode* current = ready.front();
        nodes.push_back(current);
        ready.pop();

        for (auto& child : current->m_children)
        {
            --(child->m_pendingCount);
            if (child->m_pendingCount == 0)
            {
                ready.push(child.get());
            }
        }
    }

    for (auto it = nodes.crbegin(); it != nodes.crend(); ++it)
    {
        (*it)->accept(visitor);
    }
}

void MappingGraph2::traverseComponentGroups(MappingGraphVisitor& visitor) const
{
    for (auto& [states, node] : m_groupIndex)
    {
        for (auto& child : node->m_children)
        {
            child->accept(visitor);
        }
    }
}
void MappingGraph2::traverseComponentGroups(MappingGraphVisitor& visitor,
                                            TaskScheduler* taskScheduler) const
{
    if (taskScheduler)
    {
        sofa::simulation::parallelForEach(*taskScheduler, m_groupIndex.begin(), m_groupIndex.end(),
            [&visitor](const auto& pair)
            {
                for (auto& child : pair.second->m_children)
                {
                    child->accept(visitor);
                }
            });
    }
    else
    {
        traverseComponentGroups(visitor);
    }
}

void MappingGraph2::processQueue(std::queue<BaseMappingGraphNode*>& ready, MappingGraphVisitor& visitor,
                                 bool topDown)
{
    while (!ready.empty())
    {
        BaseMappingGraphNode* current = ready.front();
        std::cout << "Processing node: " << current->getName() << std::endl;
        ready.pop();

        current->accept(visitor);

        const auto& neighbours = topDown ? current->m_children : current->m_parents;
        for (auto& neighbour : neighbours)
        {
            --(neighbour->m_pendingCount);
            if (neighbour->m_pendingCount == 0)
            {
                ready.push(neighbour.get());
            }
        }
    }
}


bool MappingGraph2::isBuilt() const { return m_isBuilt; }

template<class TComponent>
MappingGraphNode<TComponent>::SPtr makeMappingGraphNode(typename TComponent::SPtr s)
{
    return MappingGraphNode<TComponent>::SPtr( new MappingGraphNode<TComponent>(s) );
}

void MappingGraph2::build(const InputLists& input)
{
    m_hasAnyMapping = input.mappings.size() > 0;

    // 1. Create one wrapper node per object; index state nodes by raw ptr.
    std::vector<MappingGraphNode<sofa::core::behavior::BaseMechanicalState>*> mechanicalStateNodes;
    for (auto& s : input.mechanicalStates)
    {
        auto node = makeMappingGraphNode<sofa::core::behavior::BaseMechanicalState>(s);
        mechanicalStateNodes.push_back(node.get());
        m_stateIndex[s] = node.get();
        m_allNodes.push_back(std::move(node));
    }

    // Collect (leafNode*, connected states) for every leaf component type.
    std::vector<std::pair<BaseMappingGraphNode*, std::vector<core::behavior::BaseMechanicalState::SPtr>>>
        leafConnections;

    const auto processComponents = [&leafConnections, this]<class TComponent>(const sofa::type::vector<TComponent*>& components)
    {
        for (const auto& component : components)
        {
            auto node = makeMappingGraphNode<TComponent>(component);

            const auto& states = component->getMechanicalStates();
            std::vector<core::behavior::BaseMechanicalState::SPtr> statesVector{states.begin(),
                                                                                states.end()};
            leafConnections.emplace_back(node.get(), statesVector);

            m_allNodes.push_back(std::move(node));
        }
    };

    processComponents(input.forceFields);
    processComponents(input.masses);

    // 2. Wire leaf component edges:  connectedState → leafComponent
    for (auto& [leafNode, states] : leafConnections)
    {
        auto sn = findGroupNode(states);
        if (sn)
        {
            addEdge(sn.get(), leafNode);
        }
    }

    std::vector<BaseMappingGraphNode*> mappingNodePtrs;
    for (auto& m : input.mappings)
    {
        auto node = makeMappingGraphNode<sofa::core::BaseMapping>(m);
        mappingNodePtrs.push_back(node.get());
        m_allNodes.push_back(std::move(node));
    }

    // 3. Wire mapping edges:  inputState → mapping → outputState
    for (size_t i = 0; i < input.mappings.size(); ++i)
    {
        BaseMappingGraphNode* mappingNode = mappingNodePtrs[i];
        for (auto& s : input.mappings[i]->getMechFrom())
        {
            if (auto groupNode = findInGroupNodes(s))
            {
                addEdge(groupNode.get(), mappingNode);
            }
            else if (auto* sn = findStateNode(s))
            {
                addEdge(sn, mappingNode);
            }
        }

        for (auto& s : input.mappings[i]->getMechTo())
        {
            if (auto* sn = findStateNode(s))
            {
                addEdge(mappingNode, sn);
            }
        }
    }

    // 4. Roots: MechanicalState nodes with no parents.
    m_totalNbMainDofs = 0;
    for (const auto& node : mechanicalStateNodes)
    {
        if (node->m_parents.empty())
        {
            m_roots.push_back(node->m_component.get());
            m_positionInGlobalMatrix[node->m_component.get()] = type::Vec2u(m_totalNbMainDofs, m_totalNbMainDofs);
            m_totalNbMainDofs += node->m_component->getMatrixSize();
        }
    }

    m_isBuilt = true;
}

void MappingGraph2::build(core::objectmodel::BaseContext* rootNode)
{
    if (rootNode)
    {
        build(InputLists::makeFromNode(rootNode));
    }
    m_rootNode = rootNode;
}

ComponentGroupMappingGraphNode::SPtr MappingGraph2::findGroupNode(
    const std::vector<core::behavior::BaseMechanicalState::SPtr>& states)
{
    auto it = std::find_if(m_groupIndex.begin(), m_groupIndex.end(),
        [&states](auto& group){ return group.first == states; });
    if (it != m_groupIndex.end())
        return it->second;

    std::cout << "Creating group node for " << sofa::helper::join(states.begin(), states.end(), [](auto state){ return state->getName(); }, ", ") << std::endl;
    auto group = ComponentGroupMappingGraphNode::SPtr{new ComponentGroupMappingGraphNode};
    for (const auto& state : states)
    {
        if (auto* sn = findStateNode(state.get()))
        {
            addEdge(sn, group.get());
        }
    }
    m_groupIndex.emplace_back(states, group);
    m_allNodes.push_back(group);
    return group;
}

ComponentGroupMappingGraphNode::SPtr MappingGraph2::findInGroupNodes(
    const core::behavior::BaseMechanicalState::SPtr state)
{
    auto it = std::find_if(m_groupIndex.begin(), m_groupIndex.end(),
        [&state](auto& group)
        {
            return std::find(group.first.begin(), group.first.end(), state) != group.first.end();
        });
    if (it != m_groupIndex.end())
        return it->second;
    return nullptr;
}

BaseMappingGraphNode* MappingGraph2::findStateNode(core::behavior::BaseMechanicalState* raw) const
{
    auto it = m_stateIndex.find(raw);
    return (it != m_stateIndex.end()) ? it->second : nullptr;
}

void MappingGraph2::addEdge(BaseMappingGraphNode* from, BaseMappingGraphNode* to)
{
    from->m_children.push_back(to->shared_from_this());
    to->m_parents.push_back(from->shared_from_this());
}

}  // namespace sofa::simulation
