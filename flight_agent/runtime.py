"""Choose real LangGraph or an explicit, small offline reference runner.

Reference runner is NOT LangGraph. It verifies node/control-flow behavior when
dependencies cannot be installed. It has no persistence/streaming/concurrency.
"""


class ReferenceGraph:
    def __init__(self, schema):
        self.nodes, self.edges, self.routes = {}, {}, {}

    def add_node(self, name, function):
        self.nodes[name] = function

    def add_edge(self, source, destination):
        self.edges[source] = destination

    def add_conditional_edges(self, source, router, mapping):
        self.routes[source] = (router, mapping)

    def compile(self):
        return self

    def invoke(self, state, config=None):
        state = dict(state)
        node = self.edges['__start__']
        for _ in range((config or {}).get('recursion_limit', 200)):
            if node == '__end__':
                return state
            state.update(self.nodes[node](state))
            if node in self.routes:
                router, mapping = self.routes[node]
                node = mapping[router(state)]
            else:
                node = self.edges[node]
        raise RuntimeError('Reference graph recursion budget exceeded')


def graph_components(runtime):
    if runtime not in ('auto', 'reference', 'langgraph'):
        raise ValueError('runtime must be auto, reference or langgraph')
    if runtime != 'reference':
        try:
            from langgraph.graph import StateGraph, START, END
            return StateGraph, START, END, 'langgraph'
        except ModuleNotFoundError as exc:
            if runtime == 'langgraph' or not exc.name.startswith('langgraph'):
                raise
    return ReferenceGraph, '__start__', '__end__', 'reference'
