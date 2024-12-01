import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

class MoleculeVisualizer:
    def __init__(self):
        self.color_scheme = {
            'Medical': '#FF6B6B',
            'Environmental': '#4ECDC4',
            'Structural': '#45B7D1'
        }

    def display_results(self, results: Dict):
        """Display the optimization results and molecular visualizations."""
        st.header("Nanobot Design Results")

        # Display molecular properties
        self._display_properties_section(results)
        
        # Display quantum state visualization
        self._display_quantum_state(results['state_vector'])
        
        # Display molecular structure
        self._display_molecular_structure(results)
        
        # Display task-specific results
        self._display_task_specific_results(results)

    def _display_properties_section(self, results: Dict):
        """Display the main molecular properties."""
        st.subheader("Molecular Properties")
        
        # Create three columns for properties
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Stability Index",
                f"{results['molecular_properties']['stability_index']:.3f}",
                delta=None
            )
            
        with col2:
            st.metric(
                "Reactivity Measure",
                f"{results['molecular_properties']['reactivity_measure']:.3f}",
                delta=None
            )
            
        with col3:
            st.metric(
                "Coherence Measure",
                f"{results['molecular_properties']['coherence_measure']:.3f}",
                delta=None
            )

    def _display_quantum_state(self, state_vector: np.ndarray):
        """Visualize the quantum state."""
        st.subheader("Quantum State Visualization")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot amplitude distribution
        amplitudes = np.abs(state_vector) ** 2
        ax1.bar(range(len(amplitudes)), amplitudes)
        ax1.set_title("Amplitude Distribution")
        ax1.set_xlabel("Basis State")
        ax1.set_ylabel("Probability")
        
        # Plot phase distribution
        phases = np.angle(state_vector)
        ax2.bar(range(len(phases)), phases)
        ax2.set_title("Phase Distribution")
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel("Phase")
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    def _display_molecular_structure(self, results: Dict):
        """Display the molecular structure using RDKit."""
        st.subheader("Predicted Molecular Structure")
        
        # Generate a sample molecule based on the properties
        # This is a simplified example - in a real application, 
        # you would generate the actual molecular structure based on the quantum simulation
        mol = self._generate_sample_molecule(results['molecular_properties'])
        
        if mol is not None:
            # Generate 2D coordinates for the molecule
            AllChem.Compute2DCoords(mol)
            
            # Convert the molecule to an image
            img = Draw.MolToImage(mol)
            
            # Display the image
            st.image(img, caption="Predicted Molecular Structure", use_column_width=True)

    def _generate_sample_molecule(self, properties: Dict) -> rdkit.Chem.Mol:
        """Generate a sample molecule based on the properties.
        This is a simplified example - in a real application, 
        you would generate the actual molecular structure based on the quantum simulation."""
        # Create a simple molecule as an example
        # In a real application, this would be based on the quantum simulation results
        smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Example SMILES string
        mol = Chem.MolFromSmiles(smiles)
        return mol

    def _display_task_specific_results(self, results: Dict):
        """Display task-specific analysis results."""
        st.subheader("Task-Specific Analysis")
        
        if 'biocompatibility_score' in results:
            self._display_medical_results(results)
        elif 'degradation_profile' in results:
            self._display_environmental_results(results)
        elif 'structural_stability' in results:
            self._display_structural_results(results)

    def _display_medical_results(self, results: Dict):
        """Display medical application specific results."""
        st.markdown("### Medical Application Analysis")
        
        # Display biocompatibility score
        st.metric(
            "Biocompatibility Score",
            f"{results['biocompatibility_score']:.3f}",
            delta=None
        )
        
        # Create a radar chart for medical properties
        self._create_radar_chart({
            'Biocompatibility': results['biocompatibility_score'],
            'Targeting': results['molecular_properties']['stability_index'],
            'Drug Delivery': results['molecular_properties']['coherence_measure']
        }, 'Medical Properties')

    def _display_environmental_results(self, results: Dict):
        """Display environmental application specific results."""
        st.markdown("### Environmental Impact Analysis")
        
        degradation = results['degradation_profile']
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Degradation Rate", f"{degradation['degradation_rate']:.3f}")
        with cols[1]:
            st.metric("Environmental Persistence", 
                     f"{degradation['environmental_persistence']:.3f}")
        with cols[2]:
            st.metric("Interaction Potential", 
                     f"{degradation['interaction_potential']:.3f}")

    def _display_structural_results(self, results: Dict):
        """Display structural application specific results."""
        st.markdown("### Structural Properties Analysis")
        
        stability = results['structural_stability']
        cols = st.columns(3)
        
        with cols[0]:
            st.metric("Structural Integrity", 
                     f"{stability['structural_integrity']:.3f}")
        with cols[1]:
            st.metric("Flexibility Index", 
                     f"{stability['flexibility_index']:.3f}")
        with cols[2]:
            st.metric("Assembly Efficiency", 
                     f"{stability['assembly_efficiency']:.3f}")

    def _create_radar_chart(self, properties: Dict, title: str):
        """Create a radar chart for the given properties."""
        # Prepare the data
        categories = list(properties.keys())
        values = list(properties.values())
        
        # Close the plot by appending the first value
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # close the plot
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(title)
        
        st.pyplot(fig)
        plt.close()