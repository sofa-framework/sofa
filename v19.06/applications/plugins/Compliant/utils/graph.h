#ifndef COMPLIANT_UTILS_GRAPH_H
#define COMPLIANT_UTILS_GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>

namespace boost {


}

namespace utils {

template<class Vertex, class Edge, class Direction>
struct graph_traits {

	typedef boost::property<boost::vertex_color_t,
							boost::default_color_type,
							Vertex> vertex_properties;
	
	typedef boost::property< boost::edge_color_t,
							 boost::default_color_type,
							 Edge > edge_properties;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, Direction, vertex_properties, edge_properties > graph_type;

	
}; 

template<class Vertex, class Edge, class Direction = boost::undirectedS>
struct graph : graph_traits<Vertex, Edge, Direction>::graph_type {

	typedef typename graph_traits<Vertex, Edge, Direction>::graph_type base;
	
	typedef Vertex vertex_type;
	typedef Edge edge_type;
	typedef Direction direction_type;

	graph() { }
	graph(unsigned n) : base(n) { }
	
	
	// some handy typedefs
	typedef std::pair<typename graph::edge_iterator,
	                  typename graph::edge_iterator> edge_range;

	typedef std::pair<typename graph::in_edge_iterator,
	                  typename graph::in_edge_iterator> in_edge_range;

	typedef std::pair<typename graph::out_edge_iterator,
	                  typename graph::out_edge_iterator> out_edge_range;

	// easy properties
	static typename boost::edge_property_type<base>::type ep(const Edge& x) {
		return typename boost::edge_property_type<base>::type( boost::default_color_type(), x);
	};

	static typename boost::vertex_property_type<base>::type vp(const Vertex& x) {
		return typename boost::vertex_property_type<base>::type( boost::default_color_type(), x);
	};


};


// postfix (children first)
template<class F>
struct dfs_visitor : boost::default_dfs_visitor {
	F f;
	
	dfs_visitor(const F& f) : f(f) { }

	template < typename Vertex, typename Graph >
	void finish_vertex(Vertex u, const Graph & g) const {
		f(u, g);
	}
	
};


// template<class F>
// struct prefix_visitor : boost::default_dfs_visitor {
// 	F f;
	
// 	prefix_visitor(const F& f) : f(f) { }

// 	template < typename Vertex, typename Graph >
// 	void discover_vertex(Vertex u, const Graph & g) const {
// 		f(u, g);
// 	}
	
// };


template<class F>
struct bfs_visitor : boost::default_bfs_visitor {
	F f;
	
	bfs_visitor(const F& f) : f(f) { }

	template < typename Vertex, typename Graph >
	void discover_vertex(Vertex u, const Graph & g) const {
		f(u, g);
	}
	
};



// postfix (chidren first)
template<class G, class F>
void dfs(const G& g, const F& f) {
	dfs_visitor<F> vis(f);
	boost::depth_first_search(g, boost::visitor(vis));
}

// template<class G, class F>
// void prefix(const G& g, const F& f) {
// 	prefix_visitor<F> vis(f);
// 	boost::depth_first_search(g, boost::visitor(vis));
// }

template<class G, class F>
void bfs(const G& g, const F& f, unsigned root = 0) {
	bfs_visitor<F> vis(f);
	boost::breadth_first_search(g, root, boost::visitor(vis));
}




}

#endif
