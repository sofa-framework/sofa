#include <sofa/simulation/MappingGraph2.h>

#include <sofa/simulation/task/ParallelForEach.h>

namespace sofa::simulation
{

MappingGraph2::InputLists MappingGraph2::InputLists::makeFromNode(sofa::simulation::Node::SPtr node)
{
    InputLists inputLists;
    node->getTreeObjects(inputLists.mechanicalStates);
    node->getTreeObjects(inputLists.mappings);
    node->getTreeObjects(inputLists.forceFields);
    node->getTreeObjects(inputLists.masses);
    return inputLists;
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
    std::copy_if(mechanicalStateNodes.begin(), mechanicalStateNodes.end(), std::back_inserter(m_roots),
        [](auto* node) { return node->m_parents.empty(); });

    m_isBuilt = true;
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
