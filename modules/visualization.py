import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

class MoleculeVisualizer:
    def __init__(self):
        self.color_scheme = {
            'Medical': '#FF6B6B',
            'Environmental': '#4ECDC4',
            'Structural': '#45B7D1'
        }
        plt.style.use('default')

    def display_results(self, results: Dict):
        """Display comprehensive optimization results and visualizations."""
        if not results:
            st.error("No results to display")
            return

        st.header("Nanobot Design Results")

        # Display overall scores and feasibility
        self._display_overall_metrics(results)
        
        # Display molecular properties with production metrics
        self._display_properties_section(results)
        
        # Display quantum state visualization
        if 'state_vector' in results:
            self._display_quantum_state(results['state_vector'])
        
        # Display production feasibility analysis
        self._display_production_analysis(results)
        
        # Display task-specific results based on task_type
        task_type = results.get('task_type')
        if task_type == "Medical":
            self._display_medical_results(results)
        elif task_type == "Environmental":
            self._display_environmental_results(results)
        elif task_type == "Structural":
            self._display_structural_results(results)
        
        # Display optimization history
        if 'optimization_history' in results:
            self._display_optimization_history(results['optimization_history'])
            
        # Display real-world data comparisons
        if 'real_world_data' in results:
            self._display_real_world_comparisons(results)

    def _display_real_world_comparisons(self, results: Dict):
        """Display comparisons with real-world molecular data."""
        st.subheader("Real-World Data Comparisons")
        
        real_data = results.get('real_world_data', {})
        mol_data = real_data.get('molecular_data')
        protein_data = real_data.get('protein_data')
        materials_data = real_data.get('materials_data')
        
        if mol_data:
            st.markdown("### Molecular Properties Comparison")
            
            # Create comparison dataframe
            comparison_data = {
                'Property': ['Stability', 'Reactivity', 'Coherence', 'Bond Strength'],
                'Simulated': [
                    results['molecular_properties']['stability_index'],
                    results['molecular_properties']['reactivity_measure'],
                    results['molecular_properties']['coherence_measure'],
                    results['molecular_properties']['bond_strength']
                ],
                'Real-World': [
                    mol_data.get('stability_index', 'N/A'),
                    mol_data.get('reactivity_measure', 'N/A'),
                    mol_data.get('coherence_measure', 'N/A'),
                    mol_data.get('bond_strength', 'N/A')
                ]
            }
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df)
            
            # Create comparison plot
            fig = go.Figure()
            
            # Add simulated data
            fig.add_trace(go.Bar(
                name='Simulated',
                x=comparison_data['Property'],
                y=comparison_data['Simulated'],
                marker_color='rgb(55, 83, 109)'
            ))
            
            # Add real-world data where available
            real_values = []
            for val in comparison_data['Real-World']:
                try:
                    real_values.append(float(val))
                except:
                    real_values.append(None)
            
            fig.add_trace(go.Bar(
                name='Real-World',
                x=comparison_data['Property'],
                y=real_values,
                marker_color='rgb(26, 118, 255)'
            ))
            
            fig.update_layout(
                title='Simulated vs Real-World Properties',
                barmode='group',
                yaxis_title='Value',
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig)
        
        if protein_data:
            st.markdown("### Protein Structure Analysis")
            if 'sequence' in protein_data:
                st.text("Protein Sequence:")
                st.code(protein_data['sequence'])
            
            if 'secondary_structure' in protein_data:
                st.markdown("#### Secondary Structure Distribution")
                struct_data = protein_data['secondary_structure']
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(struct_data.keys()),
                    values=list(struct_data.values())
                )])
                
                fig.update_layout(title='Secondary Structure Distribution')
                st.plotly_chart(fig)
            
            if 'binding_sites' in protein_data:
                st.markdown("#### Binding Sites")
                for site in protein_data['binding_sites']:
                    st.markdown(f"Site {site['site_id']}: {', '.join(site['residues'])}")
        
        if materials_data:
            st.markdown("### Materials Properties")
            if isinstance(materials_data, dict):
                # Create materials properties table
                properties_table = []
                for key, value in materials_data.items():
                    if isinstance(value, (int, float)):
                        properties_table.append({
                            'Property': key.replace('_', ' ').title(),
                            'Value': f"{value:.4f}" if isinstance(value, float) else value
                        })
                
                if properties_table:
                    st.table(pd.DataFrame(properties_table))
                
                # Plot elastic properties if available
                if 'elastic_properties' in materials_data:
                    st.markdown("#### Elastic Properties")
                    elastic = materials_data['elastic_properties']
                    if isinstance(elastic, dict):
                        fig = go.Figure(data=[go.Bar(
                            x=list(elastic.keys()),
                            y=list(elastic.values())
                        )])
                        
                        fig.update_layout(
                            title='Elastic Properties',
                            xaxis_title='Property',
                            yaxis_title='Value (GPa)'
                        )
                        
                        st.plotly_chart(fig)

    def _display_overall_metrics(self, results: Dict):
        """Display overall performance metrics."""
        st.subheader("Overall Performance Metrics")
        
        # Create three columns for main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Optimization Score",
                f"{results.get('optimization_score', 0.0):.3f}",
                delta=None,
                help="Overall optimization score (0-1)"
            )
            
        with col2:
            feasibility = results.get('feasibility_report', {}).get('overall_score', 0.0)
            st.metric(
                "Production Feasibility",
                f"{feasibility:.3f}",
                delta=None,
                help="Production feasibility score (0-1)"
            )
            
        with col3:
            production_metrics = results.get('production_metrics', {})
            st.metric(
                "Estimated Yield",
                f"{production_metrics.get('production_yield', 0.0):.1%}",
                delta=None,
                help="Estimated production yield rate"
            )

    def _display_properties_section(self, results: Dict):
        """Display molecular properties with production implications."""
        st.subheader("Molecular Properties & Production Metrics")
        
        # Create tabs for different property categories
        tabs = st.tabs(["Core Properties", "Production Metrics", "Stability Analysis"])
        
        with tabs[0]:
            self._display_core_properties(results)
            
        with tabs[1]:
            self._display_production_metrics(results)
            
        with tabs[2]:
            self._display_stability_analysis(results)

    def _display_core_properties(self, results: Dict):
        """Display core molecular properties."""
        properties = results.get('molecular_properties', {})
        if not properties:
            st.warning("No molecular properties data available")
            return
            
        # Create radar chart for core properties
        fig = go.Figure()
        
        categories = ['Stability', 'Coherence', 'Bond Strength', 
                     'Electron Density', 'Energy Level']
        values = [
            properties.get('stability_index', 0),
            properties.get('coherence_measure', 0),
            properties.get('bond_strength', 0),
            properties.get('electron_density', 0),
            abs(results.get('final_energy', 0))
        ]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name='Core Properties'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig)

    def _display_production_metrics(self, results: Dict):
        """Display production-related metrics."""
        metrics = results.get('production_metrics', {})
        if not metrics:
            st.warning("No production metrics data available")
            return
            
        # Create bar chart for production metrics
        fig = go.Figure()
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.update_layout(
            title='Production Metrics Analysis',
            yaxis=dict(
                title='Score',
                range=[0, 1]
            ),
            xaxis=dict(
                title='Metric',
                tickangle=45
            )
        )
        
        st.plotly_chart(fig)

    def _display_stability_analysis(self, results: Dict):
        """Display stability analysis over time."""
        properties = results.get('molecular_properties', {})
        if not properties:
            st.warning("No stability data available")
            return
            
        # Create time series prediction for stability
        time_points = np.linspace(0, 100, 50)
        stability_curve = properties.get('stability_index', 0.7) * np.exp(-0.005 * time_points)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=stability_curve,
            mode='lines',
            name='Predicted Stability'
        ))
        
        fig.update_layout(
            title='Stability Prediction Over Time',
            xaxis=dict(title='Time (hours)'),
            yaxis=dict(
                title='Stability Index',
                range=[0, 1]
            )
        )
        
        st.plotly_chart(fig)

    def _display_quantum_state(self, state_vector: np.ndarray):
        """Visualize quantum state with production implications."""
        st.subheader("Quantum State Analysis")
        
        # Create tabs for different visualizations
        tabs = st.tabs(["Amplitude Distribution", "Phase Distribution", "Energy Levels"])
        
        with tabs[0]:
            self._plot_amplitude_distribution(state_vector)
            
        with tabs[1]:
            self._plot_phase_distribution(state_vector)
            
        with tabs[2]:
            self._plot_energy_levels(state_vector)

    def _plot_amplitude_distribution(self, state_vector: np.ndarray):
        """Plot amplitude distribution with manufacturing implications."""
        amplitudes = np.abs(state_vector) ** 2
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(amplitudes))),
            y=amplitudes,
            name='Amplitude'
        ))
        
        fig.update_layout(
            title='Amplitude Distribution (Production Stability Indicator)',
            xaxis_title='Basis State',
            yaxis_title='Probability'
        )
        
        st.plotly_chart(fig)

    def _plot_phase_distribution(self, state_vector: np.ndarray):
        """Plot phase distribution with coherence analysis."""
        phases = np.angle(state_vector)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(range(len(phases))),
            y=phases,
            name='Phase'
        ))
        
        fig.update_layout(
            title='Phase Distribution (Coherence Analysis)',
            xaxis_title='Basis State',
            yaxis_title='Phase (radians)'
        )
        
        st.plotly_chart(fig)

    def _plot_energy_levels(self, state_vector: np.ndarray):
        """Plot energy levels with stability implications."""
        energy_levels = np.sort(np.real(np.log(np.abs(state_vector) + 1e-10)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(energy_levels))),
            y=energy_levels,
            mode='lines+markers',
            name='Energy Levels'
        ))
        
        fig.update_layout(
            title='Energy Level Distribution (Stability Analysis)',
            xaxis_title='State Index',
            yaxis_title='Energy Level (log scale)'
        )
        
        st.plotly_chart(fig)

    def _display_production_analysis(self, results: Dict):
        """Display production feasibility analysis."""
        st.subheader("Production Feasibility Analysis")
        
        feasibility = results.get('feasibility_report', {})
        checks = feasibility.get('checks', {})
        
        # Create status indicators
        cols = st.columns(len(checks))
        for col, (check_name, status) in zip(cols, checks.items()):
            with col:
                color = 'green' if status else 'red'
                symbol = '✓' if status else '✗'
                st.markdown(
                    f"<div style='text-align: center; color: {color};'>"
                    f"<h1>{symbol}</h1>"
                    f"{check_name.replace('_', ' ').title()}</div>",
                    unsafe_allow_html=True
                )
        
        # Display recommendations if any
        if 'recommendations' in feasibility:
            st.markdown("### Recommendations for Production")
            for rec in feasibility['recommendations']:
                st.warning(rec)

    def _display_optimization_history(self, history: List[Dict]):
        """Display optimization history and convergence."""
        st.subheader("Optimization Convergence Analysis")
        
        # Extract optimization metrics
        iterations = list(range(len(history)))
        energies = [h['energy'] for h in history]
        penalties = [h['penalty'] for h in history]
        
        # Create convergence plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=iterations,
            y=energies,
            mode='lines+markers',
            name='Energy'
        ))
        
        fig.add_trace(go.Scatter(
            x=iterations,
            y=penalties,
            mode='lines+markers',
            name='Constraint Penalty'
        ))
        
        fig.update_layout(
            title='Optimization Convergence',
            xaxis_title='Iteration',
            yaxis_title='Value'
        )
        
        st.plotly_chart(fig)

    def _display_task_specific_results(self, results: Dict):
        """Display task-specific analysis results."""
        task_type = results.get('task_type', '')
        
        if task_type == "Medical":
            self._display_medical_results(results)
        elif task_type == "Environmental":
            self._display_environmental_results(results)
        elif task_type == "Structural":
            self._display_structural_results(results)

    def _display_medical_results(self, results: Dict):
        """Display medical application specific results."""
        st.subheader("Medical Application Analysis")
        
        # Get medical metrics from optimized configuration
        config = results.get('optimized_configuration', {})
        
        # Create metrics dictionary with safe gets
        metrics = {
            'Biocompatibility': results.get('biocompatibility_score', 0.8),
            'Targeting': config.get('targeting_efficiency', 0.7),
            'Payload': config.get('payload_capacity', 0.6),
            'Clearance': config.get('clearance_rate', 0.5)
        }
        
        # Create columns for metrics display
        cols = st.columns(len(metrics))
        for col, (metric_name, value) in zip(cols, metrics.items()):
            with col:
                st.metric(
                    label=metric_name,
                    value=f"{value:.2f}",
                    help=f"{metric_name} score (0-1)"
                )
        
        # Display radar chart
        self._create_metric_visualization(metrics, "Medical Properties")

    def _display_environmental_results(self, results: Dict):
        """Display environmental application specific results."""
        st.subheader("Environmental Impact Analysis")
        
        profile = results.get('degradation_profile', {})
        metrics = {
            'Degradation Rate': profile.get('degradation_rate', 0.5),
            'Environmental Persistence': profile.get('environmental_persistence', 0.3),
            'Interaction Potential': profile.get('interaction_potential', 0.7)
        }
        
        # Create columns for metrics display
        cols = st.columns(len(metrics))
        for col, (metric_name, value) in zip(cols, metrics.items()):
            with col:
                st.metric(
                    label=metric_name,
                    value=f"{value:.2f}",
                    help=f"{metric_name} score (0-1)"
                )
        
        self._create_metric_visualization(metrics, "Environmental Properties")

    def _display_structural_results(self, results: Dict):
        """Display structural application specific results."""
        st.subheader("Structural Properties Analysis")
        
        stability = results.get('structural_stability', {})
        metrics = {
            'Structural Integrity': stability.get('structural_integrity', 0.8),
            'Flexibility': stability.get('flexibility_index', 0.6),
            'Assembly Efficiency': stability.get('assembly_efficiency', 0.7)
        }
        
        # Create columns for metrics display
        cols = st.columns(len(metrics))
        for col, (metric_name, value) in zip(cols, metrics.items()):
            with col:
                st.metric(
                    label=metric_name,
                    value=f"{value:.2f}",
                    help=f"{metric_name} score (0-1)"
                )
        
        self._create_metric_visualization(metrics, "Structural Properties")

    def _create_metric_visualization(self, metrics: Dict, title: str):
        """Create a visualization for specific metrics."""
        if not metrics:
            st.warning(f"No {title.lower()} data available")
            return
            
        fig = go.Figure()
        
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Add first value again to close the polygon
        values_plot = values + [values[0]]
        categories_plot = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values_plot,
            theta=categories_plot,
            fill='toself'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title=title
        )
        
        st.plotly_chart(fig)

    # Rest of the visualization methods remain unchanged
