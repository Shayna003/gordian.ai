"""
Cycle Detector - Roberto's Implementation
Detects cyclic dependencies and finds strongly connected components
"""

import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CycleDetector:
    """Detects cycles and strongly connected components in dependency graphs"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.cycles = []
        self.strongly_connected_components = []
        self.cycle_analysis = {}
    
    def detect_all_cycles(self) -> List[List[str]]:
        """Detect all simple cycles in the dependency graph"""
        try:
            # Find all simple cycles (cycles with no repeated nodes except start/end)
            cycles = list(nx.simple_cycles(self.graph))
            self.cycles = cycles
            
            # Analyze each cycle
            self.cycle_analysis = self._analyze_cycles(cycles)
            
            logger.info(f"Found {len(cycles)} cycles in dependency graph")
            return cycles
            
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")
            return []
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components using Tarjan's algorithm"""
        try:
            # NetworkX uses Tarjan's algorithm by default
            sccs = list(nx.strongly_connected_components(self.graph))
            
            # Filter out single-node SCCs (unless they have self-loops)
            significant_sccs = []
            for scc in sccs:
                if len(scc) > 1:
                    significant_sccs.append(list(scc))
                elif len(scc) == 1:
                    # Check for self-loops
                    node = list(scc)[0]
                    if self.graph.has_edge(node, node):
                        significant_sccs.append(list(scc))
            
            self.strongly_connected_components = significant_sccs
            logger.info(f"Found {len(significant_sccs)} significant strongly connected components")
            return significant_sccs
            
        except Exception as e:
            logger.error(f"Error finding strongly connected components: {e}")
            return []
    
    def _analyze_cycles(self, cycles: List[List[str]]) -> Dict:
        """Analyze detected cycles for severity and impact"""
        analysis = {
            'total_cycles': len(cycles),
            'cycle_details': [],
            'severity_distribution': {'low': 0, 'medium': 0, 'high': 0, 'critical': 0},
            'affected_packages': set()
        }
        
        for i, cycle in enumerate(cycles):
            cycle_info = self._analyze_single_cycle(cycle, i)
            analysis['cycle_details'].append(cycle_info)
            analysis['severity_distribution'][cycle_info['severity']] += 1
            analysis['affected_packages'].update(cycle)
        
        analysis['affected_packages'] = list(analysis['affected_packages'])
        return analysis
    
    def _analyze_single_cycle(self, cycle: List[str], cycle_id: int) -> Dict:
        """Analyze a single cycle for severity and characteristics"""
        cycle_length = len(cycle)
        
        # Determine severity based on cycle characteristics
        severity = self._determine_cycle_severity(cycle)
        
        # Get dependency types in the cycle
        dependency_types = self._get_cycle_dependency_types(cycle)
        
        # Check if cycle involves root package
        involves_root = any(
            self.graph.nodes[node].get('type') == 'root' 
            for node in cycle 
            if node in self.graph.nodes
        )
        
        return {
            'id': cycle_id,
            'cycle': cycle,
            'length': cycle_length,
            'severity': severity,
            'dependency_types': dependency_types,
            'involves_root': involves_root,
            'description': self._generate_cycle_description(cycle, severity)
        }
    
    def _determine_cycle_severity(self, cycle: List[str]) -> str:
        """Determine the severity of a cycle"""
        cycle_length = len(cycle)
        
        # Check dependency types in cycle
        has_production_deps = False
        has_dev_deps = False
        
        for i in range(len(cycle)):
            current = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]
            
            if self.graph.has_edge(current, next_node):
                dep_type = self.graph[current][next_node].get('dependency_type', 'production')
                if dep_type == 'production':
                    has_production_deps = True
                elif dep_type == 'development':
                    has_dev_deps = True
        
        # Severity rules
        if cycle_length == 2 and has_production_deps:
            return 'critical'  # Direct circular dependency in production
        elif cycle_length <= 3 and has_production_deps:
            return 'high'      # Short cycle with production dependencies
        elif cycle_length <= 5 and has_production_deps:
            return 'medium'    # Medium cycle with production dependencies
        elif has_dev_deps and not has_production_deps:
            return 'low'       # Development-only cycles are less critical
        else:
            return 'medium'    # Default for other cases
    
    def _get_cycle_dependency_types(self, cycle: List[str]) -> List[str]:
        """Get the types of dependencies in a cycle"""
        dep_types = []
        
        for i in range(len(cycle)):
            current = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]
            
            if self.graph.has_edge(current, next_node):
                dep_type = self.graph[current][next_node].get('dependency_type', 'production')
                dep_types.append(dep_type)
        
        return dep_types
    
    def _generate_cycle_description(self, cycle: List[str], severity: str) -> str:
        """Generate a human-readable description of the cycle"""
        cycle_str = " → ".join(cycle + [cycle[0]])
        
        severity_descriptions = {
            'critical': 'Critical circular dependency that can cause runtime failures',
            'high': 'High-risk cycle that may cause build or runtime issues',
            'medium': 'Moderate cycle that should be reviewed and potentially refactored',
            'low': 'Low-risk cycle, primarily in development dependencies'
        }
        
        return f"{severity_descriptions.get(severity, 'Circular dependency')}: {cycle_str}"
    
    def get_cycle_breaking_suggestions(self) -> List[Dict]:
        """Suggest ways to break detected cycles"""
        suggestions = []
        
        for cycle_info in self.cycle_analysis.get('cycle_details', []):
            cycle = cycle_info['cycle']
            severity = cycle_info['severity']
            
            # Find potential breaking points
            breaking_suggestions = self._find_cycle_breaking_points(cycle)
            
            suggestions.append({
                'cycle_id': cycle_info['id'],
                'cycle': cycle,
                'severity': severity,
                'breaking_points': breaking_suggestions,
                'recommended_action': self._get_recommended_action(severity)
            })
        
        return suggestions
    
    def _find_cycle_breaking_points(self, cycle: List[str]) -> List[Dict]:
        """Find potential points where a cycle can be broken"""
        breaking_points = []
        
        for i in range(len(cycle)):
            current = cycle[i]
            next_node = cycle[(i + 1) % len(cycle)]
            
            if self.graph.has_edge(current, next_node):
                edge_data = self.graph[current][next_node]
                dep_type = edge_data.get('dependency_type', 'production')
                
                # Suggest breaking points based on dependency type
                if dep_type in ['development', 'optional']:
                    breaking_points.append({
                        'from': current,
                        'to': next_node,
                        'dependency_type': dep_type,
                        'suggestion': f"Consider making {dep_type} dependency optional or removing if not essential",
                        'impact': 'low'
                    })
                elif dep_type == 'peer':
                    breaking_points.append({
                        'from': current,
                        'to': next_node,
                        'dependency_type': dep_type,
                        'suggestion': f"Review peer dependency relationship - consider architectural refactoring",
                        'impact': 'medium'
                    })
                else:  # production dependency
                    breaking_points.append({
                        'from': current,
                        'to': next_node,
                        'dependency_type': dep_type,
                        'suggestion': f"Refactor to extract common functionality or use dependency injection",
                        'impact': 'high'
                    })
        
        return breaking_points
    
    def _get_recommended_action(self, severity: str) -> str:
        """Get recommended action based on cycle severity"""
        actions = {
            'critical': 'Immediate action required - this cycle can cause runtime failures',
            'high': 'High priority - review and refactor within current sprint',
            'medium': 'Medium priority - plan refactoring in upcoming releases',
            'low': 'Low priority - monitor and consider refactoring during maintenance'
        }
        return actions.get(severity, 'Review and assess impact')
    
    def get_analysis_summary(self) -> Dict:
        """Get a comprehensive summary of cycle analysis"""
        sccs = self.find_strongly_connected_components()
        cycles = self.detect_all_cycles()
        
        return {
            'graph_stats': {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(self.graph)
            },
            'cycle_analysis': self.cycle_analysis,
            'strongly_connected_components': {
                'count': len(sccs),
                'components': sccs,
                'largest_component_size': max([len(scc) for scc in sccs]) if sccs else 0
            },
            'recommendations': self.get_cycle_breaking_suggestions()
        }
    
    def is_dag(self) -> bool:
        """Check if the graph is a Directed Acyclic Graph (DAG)"""
        return nx.is_directed_acyclic_graph(self.graph)
    
    def get_topological_order(self) -> Optional[List[str]]:
        """Get topological ordering if graph is a DAG"""
        if self.is_dag():
            try:
                return list(nx.topological_sort(self.graph))
            except nx.NetworkXError:
                return None
        return None
