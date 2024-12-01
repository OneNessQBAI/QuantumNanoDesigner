import streamlit as st
import cirq
import numpy as np
from modules.input_handler import InputHandler, TaskParameters
from modules.quantum_simulator import QuantumSimulator
from modules.visualization import MoleculeVisualizer
from modules.optimization import TaskOptimizer

class NanobotDesigner:
    def __init__(self):
        self.input_handler = InputHandler()
        self.quantum_simulator = QuantumSimulator()
        self.visualizer = MoleculeVisualizer()
        self.optimizer = TaskOptimizer()

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
                'biocompatibility': st.sidebar.slider("Biocompatibility", 0.0, 1.0, 0.8),
                'targeting_specificity': st.sidebar.slider("Targeting Specificity", 0.0, 1.0, 0.7),
                'payload_capacity': st.sidebar.slider("Payload Capacity", 0.0, 1.0, 0.5)
            })
            
        elif task_type == "Environmental":
            st.sidebar.subheader("Environmental Parameters")
            params_dict.update({
                'pollutant_affinity': st.sidebar.slider("Pollutant Affinity", 0.0, 1.0, 0.8),
                'environmental_stability': st.sidebar.slider("Environmental Stability", 0.0, 1.0, 0.7)
            })
            
        elif task_type == "Structural":
            st.sidebar.subheader("Structural Parameters")
            params_dict.update({
                'mechanical_strength': st.sidebar.slider("Mechanical Strength", 0.0, 1.0, 0.8),
                'flexibility': st.sidebar.slider("Flexibility", 0.0, 1.0, 0.6)
            })

        if st.button("Generate Design"):
            with st.spinner("Running quantum simulation..."):
                try:
                    # Process parameters
                    params = self.input_handler.collect_parameters(params_dict)
                    
                    # Run quantum simulation
                    quantum_results = self.quantum_simulator.run_simulation(params)
                    
                    # Optimize results
                    optimized_design = self.optimizer.optimize(quantum_results, params)
                    
                    # Display results
                    st.success("Design Generated Successfully!")
                    self.visualizer.display_results(optimized_design)
                    
                except Exception as e:
                    st.error(f"Error during simulation: {str(e)}")

if __name__ == "__main__":
    app = NanobotDesigner()
    app.run_app()
