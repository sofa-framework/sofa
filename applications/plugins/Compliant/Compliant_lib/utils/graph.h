#ifndef GRAPH_H
#define GRAPH_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/breadth_first_search.hpp>

namespace utils {

template<class Vertex, class Edge, class Direction>
struct graph_traits {
	
	typedef boost::property<boost::vertex_bundle_t, Vertex,
	                        boost::property< boost::vertex_color_t, boost::default_color_type> > vertex_properties;
    
	typedef boost::property<boost::edge_bundle_t, Edge, 
	                        boost::property< boost::edge_color_t, boost::default_color_type> > edge_properties;
    
	typedef boost::adjacency_list<boost::vecS, boost::vecS, Direction, vertex_properties, edge_properties > graph_type;
	
};

template<class Vertex, class Edge, class Direction = boost::undirectedS>
struct graph : graph_traits<Vertex, Edge, Direction>::graph_type {
	

};


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
