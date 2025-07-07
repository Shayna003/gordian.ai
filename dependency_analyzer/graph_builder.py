"""
Dependency Graph Builder - Roberto's Implementation
Parses package.json and package-lock.json files to build dependency graphs
"""

import json
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

class DependencyGraphBuilder:
    """Builds dependency graphs from npm package files"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.package_versions = {}
        self.dependency_types = {}
    
    def parse_package_json(self, package_json_content: str) -> Dict:
        """Parse package.json content"""
        try:
            return json.loads(package_json_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse package.json: {e}")
            return {}
    
    def parse_package_lock_json(self, package_lock_content: str) -> Dict:
        """Parse package-lock.json content"""
        try:
            return json.loads(package_lock_content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse package-lock.json: {e}")
            return {}
    
    def build_graph_from_package_json(self, package_json_content: str) -> nx.DiGraph:
        """Build dependency graph from package.json"""
        package_data = self.parse_package_json(package_json_content)
        if not package_data:
            return self.graph
        
        root_package = package_data.get('name', 'root-project')
        self.graph.add_node(root_package, 
                           version=package_data.get('version', '1.0.0'),
                           type='root')
        
        # Add different types of dependencies
        dependency_sections = {
            'dependencies': 'production',
            'devDependencies': 'development', 
            'peerDependencies': 'peer',
            'optionalDependencies': 'optional'
        }
        
        for section, dep_type in dependency_sections.items():
            dependencies = package_data.get(section, {})
            for dep_name, version in dependencies.items():
                self._add_dependency(root_package, dep_name, version, dep_type)
        
        return self.graph
    
    def build_graph_from_package_lock(self, package_lock_content: str) -> nx.DiGraph:
        """Build detailed dependency graph from package-lock.json"""
        lock_data = self.parse_package_lock_json(package_lock_content)
        if not lock_data:
            return self.graph
        
        root_package = lock_data.get('name', 'root-project')
        self.graph.add_node(root_package,
                           version=lock_data.get('version', '1.0.0'),
                           type='root')
        
        # Process packages from lockfile
        packages = lock_data.get('packages', {})
        dependencies = lock_data.get('dependencies', {})
        
        # Handle lockfile v2 format (packages) and v1 format (dependencies)
        if packages:
            self._process_packages_v2(packages, root_package)
        elif dependencies:
            self._process_dependencies_v1(dependencies, root_package)
        
        return self.graph
    
    def _process_packages_v2(self, packages: Dict, root_package: str):
        """Process package-lock.json v2 format"""
        for package_path, package_info in packages.items():
            if package_path == "":  # Root package
                continue
                
            # Extract package name from path
            package_name = package_path.split('/')[-1] if '/' in package_path else package_path
            package_name = package_name.replace('node_modules/', '')
            
            version = package_info.get('version', 'unknown')
            self.graph.add_node(package_name, 
                               version=version,
                               type='dependency',
                               path=package_path)
            
            # Add dependencies
            deps = package_info.get('dependencies', {})
            for dep_name, dep_version in deps.items():
                self._add_dependency(package_name, dep_name, dep_version, 'production')
    
    def _process_dependencies_v1(self, dependencies: Dict, root_package: str):
        """Process package-lock.json v1 format"""
        for dep_name, dep_info in dependencies.items():
            version = dep_info.get('version', 'unknown')
            self.graph.add_node(dep_name,
                               version=version,
                               type='dependency')
            
            # Add edge from root to this dependency
            self._add_dependency(root_package, dep_name, version, 'production')
            
            # Recursively process nested dependencies
            nested_deps = dep_info.get('dependencies', {})
            if nested_deps:
                self._process_nested_dependencies(dep_name, nested_deps)
    
    def _process_nested_dependencies(self, parent: str, dependencies: Dict):
        """Process nested dependencies recursively"""
        for dep_name, dep_info in dependencies.items():
            version = dep_info.get('version', 'unknown')
            self.graph.add_node(dep_name,
                               version=version,
                               type='dependency')
            
            self._add_dependency(parent, dep_name, version, 'production')
            
            # Recursively process further nested dependencies
            nested_deps = dep_info.get('dependencies', {})
            if nested_deps:
                self._process_nested_dependencies(dep_name, nested_deps)
    
    def _add_dependency(self, from_package: str, to_package: str, version: str, dep_type: str):
        """Add a dependency edge to the graph"""
        # Add the target node if it doesn't exist
        if not self.graph.has_node(to_package):
            self.graph.add_node(to_package, 
                               version=version,
                               type='dependency')
        
        # Add the edge
        self.graph.add_edge(from_package, to_package,
                           version=version,
                           dependency_type=dep_type)
        
        # Store metadata
        self.package_versions[to_package] = version
        self.dependency_types[f"{from_package}->{to_package}"] = dep_type
    
    def get_graph_stats(self) -> Dict:
        """Get statistics about the dependency graph"""
        return {
            'total_packages': self.graph.number_of_nodes(),
            'total_dependencies': self.graph.number_of_edges(),
            'is_connected': nx.is_weakly_connected(self.graph),
            'density': nx.density(self.graph),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0
        }
    
    def get_package_dependencies(self, package_name: str) -> List[str]:
        """Get direct dependencies of a package"""
        if package_name in self.graph:
            return list(self.graph.successors(package_name))
        return []
    
    def get_package_dependents(self, package_name: str) -> List[str]:
        """Get packages that depend on this package"""
        if package_name in self.graph:
            return list(self.graph.predecessors(package_name))
        return []
    
    def export_graph_data(self) -> Dict:
        """Export graph data for visualization"""
        nodes = []
        edges = []
        
        for node in self.graph.nodes(data=True):
            nodes.append({
                'id': node[0],
                'label': node[0],
                'version': node[1].get('version', 'unknown'),
                'type': node[1].get('type', 'dependency')
            })
        
        for edge in self.graph.edges(data=True):
            edges.append({
                'source': edge[0],
                'target': edge[1],
                'version': edge[2].get('version', 'unknown'),
                'dependency_type': edge[2].get('dependency_type', 'production')
            })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'stats': self.get_graph_stats()
        }
    
    def add_demo_cycles_for_testing(self):
        """Add some artificial cycles for demonstration purposes"""
        # This method is for demo purposes only
        
        # Create a simple 3-node cycle: A -> B -> C -> A
        self.graph.add_edge('demo-package-a', 'demo-package-b', dependency_type='production')
        self.graph.add_edge('demo-package-b', 'demo-package-c', dependency_type='production')
        self.graph.add_edge('demo-package-c', 'demo-package-a', dependency_type='production')
        
        # Create a 2-node cycle: D -> E -> D
        self.graph.add_edge('demo-package-d', 'demo-package-e', dependency_type='development')
        self.graph.add_edge('demo-package-e', 'demo-package-d', dependency_type='development')
        
        # Create a more complex cycle involving real packages from the uploaded file
        # This creates artificial cycles with some of the packages for demo
        if 'react' in self.graph.nodes and 'webpack' in self.graph.nodes:
            self.graph.add_edge('react', 'babel-loader', dependency_type='production')
            self.graph.add_edge('babel-loader', 'webpack', dependency_type='production')
            self.graph.add_edge('webpack', 'react', dependency_type='production')
        
        # Add some metadata
        for pkg in ['demo-package-a', 'demo-package-b', 'demo-package-c', 'demo-package-d', 'demo-package-e']:
            if not self.graph.has_node(pkg):
                self.graph.add_node(pkg, type='demo', version='1.0.0')
        
        logger.info("Added demo cycles for testing purposes")
