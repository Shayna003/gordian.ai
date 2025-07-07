"""
Dependency Visualizer - Roberto's Implementation
Creates interactive visualizations for dependency graphs and cycles
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, List, Tuple, Optional
import streamlit as st
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DependencyVisualizer:
    """Creates interactive visualizations for dependency analysis"""
    
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.layout_cache = {}
    
    def create_dependency_graph_plot(self, cycles: List[List[str]] = None, 
                                   highlight_sccs: List[List[str]] = None) -> go.Figure:
        """Create an interactive dependency graph visualization"""
        if self.graph.number_of_nodes() == 0:
            return self._create_empty_plot("No dependencies to visualize")
        
        # Calculate layout
        pos = self._get_graph_layout()
        
        # Prepare node data
        node_trace = self._create_node_trace(pos, cycles, highlight_sccs)
        
        # Prepare edge data
        edge_traces = self._create_edge_traces(pos, cycles)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces,
                       layout=self._get_plot_layout())
        
        return fig
    
    def _get_graph_layout(self) -> Dict:
        """Calculate graph layout using spring algorithm"""
        if 'spring' not in self.layout_cache:
            try:
                # Use spring layout for better visualization
                if self.graph.number_of_nodes() > 100:
                    # For large graphs, use a faster algorithm
                    pos = nx.spring_layout(self.graph, k=1, iterations=20)
                else:
                    pos = nx.spring_layout(self.graph, k=2, iterations=50)
                
                self.layout_cache['spring'] = pos
            except Exception as e:
                logger.error(f"Error calculating layout: {e}")
                # Fallback to circular layout
                pos = nx.circular_layout(self.graph)
                self.layout_cache['spring'] = pos
        
        return self.layout_cache['spring']
    
    def _create_node_trace(self, pos: Dict, cycles: List[List[str]] = None, 
                          highlight_sccs: List[List[str]] = None) -> go.Scatter:
        """Create node trace for the graph"""
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        # Identify nodes in cycles and SCCs
        cycle_nodes = set()
        scc_nodes = set()
        
        if cycles:
            for cycle in cycles:
                cycle_nodes.update(cycle)
        
        if highlight_sccs:
            for scc in highlight_sccs:
                scc_nodes.update(scc)
        
        for node in self.graph.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node information
                node_data = self.graph.nodes[node]
                version = node_data.get('version', 'unknown')
                node_type = node_data.get('type', 'dependency')
                
                # Determine node color and size based on characteristics
                color, size = self._get_node_style(node, node_type, cycle_nodes, scc_nodes)
                node_colors.append(color)
                node_sizes.append(size)
                
                # Create hover text
                dependencies = list(self.graph.successors(node))
                dependents = list(self.graph.predecessors(node))
                
                hover_text = f"<b>{node}</b><br>"
                hover_text += f"Version: {version}<br>"
                hover_text += f"Type: {node_type}<br>"
                hover_text += f"Dependencies: {len(dependencies)}<br>"
                hover_text += f"Dependents: {len(dependents)}"
                
                if node in cycle_nodes:
                    hover_text += "<br><b>⚠️ Part of cycle</b>"
                if node in scc_nodes:
                    hover_text += "<br><b>🔗 In SCC</b>"
                
                node_text.append(hover_text)
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in self.graph.nodes() if node in pos],
            textposition="middle center",
            textfont=dict(size=8),
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            name="Packages"
        )
    
    def _get_node_style(self, node: str, node_type: str, cycle_nodes: set, scc_nodes: set) -> Tuple[str, int]:
        """Determine node color and size based on its characteristics"""
        # Base size
        size = 15
        
        # Adjust size based on connections
        degree = self.graph.degree(node)
        if degree > 10:
            size = 25
        elif degree > 5:
            size = 20
        
        # Determine color
        if node in cycle_nodes and node in scc_nodes:
            color = '#FF4444'  # Red for nodes in both cycles and SCCs
        elif node in cycle_nodes:
            color = '#FF8800'  # Orange for cycle nodes
        elif node in scc_nodes:
            color = '#FFAA00'  # Yellow-orange for SCC nodes
        elif node_type == 'root':
            color = '#4444FF'  # Blue for root package
        else:
            color = '#44AA44'  # Green for regular dependencies
        
        return color, size
    
    def _create_edge_traces(self, pos: Dict, cycles: List[List[str]] = None) -> List[go.Scatter]:
        """Create edge traces for the graph"""
        edge_traces = []
        
        # Identify cycle edges
        cycle_edges = set()
        if cycles:
            for cycle in cycles:
                for i in range(len(cycle)):
                    current = cycle[i]
                    next_node = cycle[(i + 1) % len(cycle)]
                    cycle_edges.add((current, next_node))
        
        # Regular edges
        regular_edge_x = []
        regular_edge_y = []
        
        # Cycle edges
        cycle_edge_x = []
        cycle_edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = pos.get(edge[0], (0, 0))
            x1, y1 = pos.get(edge[1], (0, 0))
            
            if edge in cycle_edges:
                cycle_edge_x.extend([x0, x1, None])
                cycle_edge_y.extend([y0, y1, None])
            else:
                regular_edge_x.extend([x0, x1, None])
                regular_edge_y.extend([y0, y1, None])
        
        # Add regular edges trace
        if regular_edge_x:
            edge_traces.append(go.Scatter(
                x=regular_edge_x, y=regular_edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines',
                name="Dependencies"
            ))
        
        # Add cycle edges trace
        if cycle_edge_x:
            edge_traces.append(go.Scatter(
                x=cycle_edge_x, y=cycle_edge_y,
                line=dict(width=3, color='#FF4444'),
                hoverinfo='none',
                mode='lines',
                name="Cycle Dependencies"
            ))
        
        return edge_traces
    
    def _get_plot_layout(self) -> dict:
        """Get layout configuration for the plot"""
        return dict(
            title="Dependency Graph",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes for details. Red edges indicate cycles.",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color="#888", size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        return fig
    
    def create_cycle_analysis_chart(self, cycle_analysis: Dict) -> go.Figure:
        """Create a chart showing cycle analysis"""
        if not cycle_analysis or cycle_analysis.get('total_cycles', 0) == 0:
            return self._create_empty_plot("No cycles detected")
        
        severity_dist = cycle_analysis.get('severity_distribution', {})
        
        # Create bar chart of severity distribution
        severities = list(severity_dist.keys())
        counts = list(severity_dist.values())
        colors = ['#44AA44', '#FFAA00', '#FF8800', '#FF4444']  # green, yellow, orange, red
        
        fig = go.Figure(data=[
            go.Bar(
                x=severities,
                y=counts,
                marker_color=colors[:len(severities)],
                text=counts,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Cycle Severity Distribution",
            xaxis_title="Severity Level",
            yaxis_title="Number of Cycles",
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_scc_analysis_chart(self, sccs: List[List[str]]) -> go.Figure:
        """Create a chart showing strongly connected components analysis"""
        if not sccs:
            return self._create_empty_plot("No strongly connected components found")
        
        # Analyze SCC sizes
        scc_sizes = [len(scc) for scc in sccs]
        
        # Create histogram of SCC sizes
        fig = go.Figure(data=[
            go.Histogram(
                x=scc_sizes,
                nbinsx=min(10, max(scc_sizes)),
                marker_color='#4444FF',
                opacity=0.7
            )
        ])
        
        fig.update_layout(
            title="Strongly Connected Components Size Distribution",
            xaxis_title="Component Size (Number of Packages)",
            yaxis_title="Number of Components",
            plot_bgcolor='white'
        )
        
        return fig
    
    def display_cycle_details_table(self, cycle_analysis: Dict):
        """Display detailed cycle information in a table"""
        if not cycle_analysis or cycle_analysis.get('total_cycles', 0) == 0:
            st.info("No cycles detected in the dependency graph.")
            return
        
        cycle_details = cycle_analysis.get('cycle_details', [])
        
        # Prepare data for table
        table_data = []
        for detail in cycle_details:
            cycle_str = " → ".join(detail['cycle'] + [detail['cycle'][0]])
            table_data.append({
                'Cycle ID': detail['id'],
                'Cycle Path': cycle_str,
                'Length': detail['length'],
                'Severity': detail['severity'],
                'Involves Root': '✓' if detail['involves_root'] else '✗',
                'Description': detail['description']
            })
        
        df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def style_severity(val):
            colors = {
                'critical': 'background-color: #ffebee',
                'high': 'background-color: #fff3e0',
                'medium': 'background-color: #fffde7',
                'low': 'background-color: #f1f8e9'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(style_severity, subset=['Severity'])
        st.dataframe(styled_df, use_container_width=True)
    
    def display_scc_details_table(self, sccs: List[List[str]]):
        """Display detailed SCC information in a table"""
        if not sccs:
            st.info("No strongly connected components found.")
            return
        
        # Prepare data for table
        table_data = []
        for i, scc in enumerate(sccs):
            table_data.append({
                'SCC ID': i,
                'Size': len(scc),
                'Packages': ', '.join(scc),
                'Risk Level': self._assess_scc_risk(len(scc))
            })
        
        df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def style_risk(val):
            colors = {
                'High': 'background-color: #ffebee',
                'Medium': 'background-color: #fff3e0',
                'Low': 'background-color: #f1f8e9'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(style_risk, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True)
    
    def _assess_scc_risk(self, size: int) -> str:
        """Assess risk level based on SCC size"""
        if size >= 5:
            return 'High'
        elif size >= 3:
            return 'Medium'
        else:
            return 'Low'
