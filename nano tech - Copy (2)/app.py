import streamlit as st
import numpy as np
from modules.input_handler import InputHandler, TaskParameters
from modules.quantum_simulator import QuantumSimulator
from modules.visualization import MoleculeVisualizer
from modules.optimization import TaskOptimizer
from modules.data_exporter import NanobotDataExporter
import os

class NanobotDesigner:
    def __init__(self):
        self.input_handler = InputHandler()
        self.quantum_simulator = QuantumSimulator()
        self.visualizer = MoleculeVisualizer()
        self.optimizer = TaskOptimizer()
        self.data_exporter = NanobotDataExporter()

    def run_app(self):
        st.title("Quantum Nanobot Designer")
        st.sidebar.header("Configuration Options")

        # Task Selection
        task_type = st.sidebar.selectbox(
            "Select Nanobot Application",
            ["Medical", "Environmental", "Structural"]
        )

        # Common Parameters
        st.sidebar.subheader("General Parameters")
        size_constraint = st.sidebar.slider("Size Constraint (nm)", 1, 100, 50)
        energy_efficiency = st.sidebar.slider("Energy Efficiency", 0.0, 1.0, 0.7)

        # Task-specific parameters
        params_dict = {
            'task_type': task_type,
            'size_constraint': size_constraint,
            'energy_efficiency': energy_efficiency
        }

        if task_type == "Medical":
            st.sidebar.subheader("Medical Parameters")
            params_dict.update({
                'biocompatibility': st.sidebar.slider("Biocompatibility", 0.7, 1.0, 0.8),
                'targeting_specificity': st.sidebar.slider("Targeting Specificity", 0.6, 1.0, 0.7),
                'payload_capacity': st.sidebar.slider("Payload Capacity", 0.4, 0.8, 0.5),
                'clearance_rate': st.sidebar.slider("Clearance Rate", 0.3, 0.7, 0.4),
                'immune_response': st.sidebar.slider("Immune Response", 0.0, 0.3, 0.2)
            })
            
        elif task_type == "Environmental":
            st.sidebar.subheader("Environmental Parameters")
            params_dict.update({
                'pollutant_affinity': st.sidebar.slider("Pollutant Affinity", 0.6, 1.0, 0.8),
                'environmental_stability': st.sidebar.slider("Environmental Stability", 0.7, 1.0, 0.8),
                'degradation_rate': st.sidebar.slider("Degradation Rate", 0.4, 0.8, 0.6),
                'weather_resistance': st.sidebar.slider("Weather Resistance", 0.6, 1.0, 0.7),
                'ph_tolerance': st.sidebar.slider("pH Tolerance", 0.5, 1.0, 0.7)
            })
            
        elif task_type == "Structural":
            st.sidebar.subheader("Structural Parameters")
            params_dict.update({
                'mechanical_strength': st.sidebar.slider("Mechanical Strength", 0.7, 1.0, 0.8),
                'flexibility': st.sidebar.slider("Flexibility", 0.5, 0.9, 0.7),
                'thermal_stability': st.sidebar.slider("Thermal Stability", 0.6, 1.0, 0.8),
                'stress_tolerance': st.sidebar.slider("Stress Tolerance", 0.7, 1.0, 0.8),
                'fatigue_resistance': st.sidebar.slider("Fatigue Resistance", 0.6, 1.0, 0.7)
            })

        if st.button("Generate Design"):
            with st.spinner("Running quantum simulation..."):
                try:
                    # Process parameters
                    params = self.input_handler.collect_parameters(params_dict)
                    
                    # Run quantum simulation
                    quantum_results = self.quantum_simulator.run_simulation(params)
                    
                    # Add task type to results
                    quantum_results['task_type'] = task_type
                    
                    # Ensure optimized_configuration exists with defaults
                    if 'optimized_configuration' not in quantum_results:
                        quantum_results['optimized_configuration'] = {
                            'targeting_efficiency': 0.7,
                            'payload_capacity': 0.7,
                            'clearance_rate': 0.5
                        }
                    
                    # Optimize results
                    optimized_results = self.optimizer.optimize(quantum_results, params)
                    optimized_results['task_type'] = task_type
                    
                    # Display success message
                    st.success("Design Generated Successfully!")
                    
                    # Display visualizations first
                    self.visualizer.display_results(optimized_results)
                    
                    st.markdown("---")
                    
                    # Generate and offer CSV download
                    csv_data = self.data_exporter.generate_csv_data(optimized_results, task_type)
                    csv_path = self.data_exporter.export_to_csv(csv_data)
                    
                    with open(csv_path, 'r') as f:
                        csv_contents = f.read()
                    
                    st.download_button(
                        label="📥 Download Complete Design Specifications (CSV)",
                        data=csv_contents,
                        file_name=f"nanobot_design_{task_type.lower()}.csv",
                        mime="text/csv",
                        help="Download detailed nanobot specifications including all parameters and measurements"
                    )
                    
                    # Clean up temporary file
                    try:
                        os.remove(csv_path)
                    except:
                        pass
                    
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")
                    st.error("Please adjust your parameters and try again.")

if __name__ == "__main__":
    app = NanobotDesigner()
    app.run_app()
