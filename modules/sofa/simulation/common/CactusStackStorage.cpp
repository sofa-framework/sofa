#include <sofa/simulation/common/CactusStackStorage.h>

namespace sofa
{

namespace simulation
{

void CactusStackStorage::push(void* data)
{
    stack.push(data);
}
void* CactusStackStorage::pop()
{
    if (stack.empty()) return NULL;
    void* data = stack.top();
    stack.pop();
    return data;
}
void* CactusStackStorage::top() const
{
    if (stack.empty())
        if (up)
            return up->top();
        else
            return NULL;
    else
        return stack.top();
}
bool CactusStackStorage::empty() const
{
    return stack.empty() && (up == NULL || up->empty());
}

} // namespace simulation

} // namespace sofa

