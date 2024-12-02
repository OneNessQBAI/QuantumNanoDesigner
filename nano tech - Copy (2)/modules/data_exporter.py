import csv
import numpy as np
from typing import Dict, List
import os
from datetime import datetime

class NanobotDataExporter:
    def __init__(self):
        self.synthesis_methods = {
            "Medical": "DNA origami self-assembly with gold nanoparticle cores",
            "Environmental": "Green synthesis using plant extracts",
            "Structural": "3D printing with biocompatible polymers"
        }
        
        self.functionalization = {
            "Medical": "PEG coating with targeting ligands",
            "Environmental": "Surface-active groups for pollutant binding",
            "Structural": "Cross-linking agents for enhanced stability"
        }
        
        self.target_molecules = {
            "Medical": "Cancer cell markers (e.g., HER2, CD44)",
            "Environmental": "Heavy metals, organic pollutants",
            "Structural": "Tissue scaffold proteins"
        }
        
        self.storage_conditions = {
            "Medical": "4°C in sterile PBS buffer",
            "Environmental": "Room temperature in sealed containers",
            "Structural": "Controlled humidity (40-60%) at 20-25°C"
        }

    def generate_csv_data(self, quantum_results: Dict, task_type: str) -> List[List[str]]:
        """Generate comprehensive nanobot data in CSV format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        molecular_props = quantum_results.get('molecular_properties', {})
        production_metrics = quantum_results.get('production_metrics', {})
        
        # Calculate shape based on quantum state
        state_vector = quantum_results.get('state_vector', np.zeros(8))
        shape = self._determine_shape(state_vector)
        
        # Generate material composition based on task type and properties
        material = self._determine_material(task_type, molecular_props)
        
        data = [
            ["Nanobot Design Specifications", timestamp],
            [""],
            ["1. General Specifications"],
            ["Application Type", task_type],
            ["Size (nm)", f"{np.random.uniform(20, 80):.1f}"],
            ["Shape/Geometry", shape],
            ["Material Composition", material],
            [""],
            ["2. Molecular Properties"],
            ["Stability Index", f"{molecular_props.get('stability_index', 0.7):.3f}"],
            ["Coherence Measure", f"{molecular_props.get('coherence_measure', 0.6):.3f}"],
            ["Reactivity Measure", f"{molecular_props.get('reactivity_measure', 0.3):.3f}"],
            ["Bond Strength", f"{molecular_props.get('bond_strength', 0.7):.3f}"],
            ["Electron Density", f"{molecular_props.get('electron_density', 0.5):.3f}"],
            ["Energy Levels", ", ".join([f"{e:.3f}" for e in molecular_props.get('energy_levels', [-0.5, -0.3, -0.1])])],
            [""],
            ["3. Production Metrics"],
            ["Manufacturability Score", f"{production_metrics.get('manufacturability_score', 0.8):.3f}"],
            ["Scalability Score", f"{production_metrics.get('scalability_score', 0.75):.3f}"],
            ["Cost-Efficiency", f"{production_metrics.get('cost_efficiency', 0.7):.3f}"],
            ["Estimated Yield", f"{production_metrics.get('production_yield', 0.85):.1%}"],
            ["Quality Consistency", f"{production_metrics.get('quality_consistency', 0.9):.3f}"],
            [""]
        ]
        
        # Add task-specific metrics
        data.extend(self._get_task_specific_metrics(task_type, quantum_results))
        
        # Add quantum state analysis
        data.extend([
            ["5. Quantum State Analysis"],
            ["Amplitude Distribution", ", ".join([f"{abs(x)**2:.3f}" for x in state_vector])],
            ["Phase Distribution", ", ".join([f"{np.angle(x):.3f}" for x in state_vector])],
            ["Energy Level Distribution", ", ".join([f"{e:.3f}" for e in molecular_props.get('energy_levels', [-0.5, -0.3, -0.1])])],
            [""]
        ])
        
        # Add design parameters
        data.extend([
            ["6. Design Parameters"],
            ["Input Size Constraint (nm)", "20-100"],
            ["Input Energy Efficiency", f"{quantum_results.get('final_energy', -0.5):.3f}"],
            ["Optimization Score", f"{quantum_results.get('optimization_score', 0.85):.3f}"],
            [""]
        ])
        
        # Add lab-specific instructions
        data.extend([
            ["7. Lab-Specific Instructions"],
            ["Synthesis Methods", self.synthesis_methods[task_type]],
            ["Functionalization Details", self.functionalization[task_type]],
            ["Target Molecules/Cells", self.target_molecules[task_type]],
            ["Assembly Protocol", self._generate_assembly_protocol(task_type)],
            [""]
        ])
        
        # Add safety and feasibility
        data.extend([
            ["8. Safety and Feasibility"],
            ["Toxicity Threshold", "< 0.1 IC50"],
            ["Regulatory Compliance", self._get_regulatory_info(task_type)],
            ["Storage Conditions", self.storage_conditions[task_type]],
            ["Stability Period", "6 months under specified conditions"]
        ])
        
        return data

    def export_to_csv(self, data: List[List[str]], filename: str = None) -> str:
        """Export data to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"nanobot_design_{timestamp}.csv"
        
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        
        return filepath

    def _determine_shape(self, state_vector: np.ndarray) -> str:
        """Determine nanobot shape based on quantum state."""
        amplitude_distribution = np.abs(state_vector) ** 2
        symmetry = np.std(amplitude_distribution)
        
        if symmetry < 0.1:
            return "Spherical"
        elif symmetry < 0.2:
            return "Cylindrical"
        else:
            return "Complex polyhedron"

    def _determine_material(self, task_type: str, properties: Dict) -> str:
        """Determine optimal material composition."""
        materials = {
            "Medical": [
                "Gold nanoparticles with DNA origami",
                "PEGylated liposomes",
                "Biodegradable polymers (PLGA)"
            ],
            "Environmental": [
                "Iron oxide nanoparticles",
                "Activated carbon composites",
                "Titanium dioxide matrices"
            ],
            "Structural": [
                "Carbon nanotubes",
                "Graphene oxide sheets",
                "Silicon-based frameworks"
            ]
        }
        
        # Select material based on properties
        stability = properties.get('stability_index', 0.7)
        
        return materials[task_type][int(stability * 2) % len(materials[task_type])]

    def _get_task_specific_metrics(self, task_type: str, results: Dict) -> List[List[str]]:
        """Get metrics specific to the task type."""
        data = [["4. Task-Specific Metrics"]]
        
        config = results.get('optimized_configuration', {})
        
        if task_type == "Medical":
            data.extend([
                ["Biocompatibility Score", f"{results.get('biocompatibility_score', 0.85):.3f}"],
                ["Targeting Specificity", f"{config.get('targeting_efficiency', 0.7):.3f}"],
                ["Payload Capacity", f"{config.get('payload_capacity', 0.7):.3f}"],
                ["Clearance Rate", f"{config.get('clearance_rate', 0.5):.3f}"],
                ["Immune Response", "Minimal"]
            ])
        elif task_type == "Environmental":
            degradation = results.get('degradation_profile', {})
            data.extend([
                ["Pollutant Binding Efficiency", f"{0.85:.3f}"],
                ["Degradation Rate", f"{degradation.get('degradation_rate', 0.3):.3f}"],
                ["Environmental Safety", f"{0.95:.3f}"],
                ["Weather Resistance", f"{0.8:.3f}"]
            ])
        else:  # Structural
            stability = results.get('structural_stability', {})
            data.extend([
                ["Mechanical Strength", f"{stability.get('structural_integrity', 0.85):.3f}"],
                ["Assembly Efficiency", f"{stability.get('assembly_efficiency', 0.8):.3f}"],
                ["Stress Tolerance", f"{0.75:.3f}"],
                ["Thermal Resistance", f"{0.8:.3f}"]
            ])
        
        data.append([""])
        return data

    def _generate_assembly_protocol(self, task_type: str) -> str:
        """Generate step-by-step assembly protocol."""
        protocols = {
            "Medical": "1. Prepare DNA origami scaffold\n2. Attach gold nanoparticles\n3. Add targeting ligands\n4. PEG coating\n5. Purification",
            "Environmental": "1. Prepare precursor solution\n2. Green synthesis\n3. Surface modification\n4. Stability testing\n5. Activity verification",
            "Structural": "1. Prepare polymer solution\n2. 3D printing setup\n3. Layer-by-layer assembly\n4. Cross-linking\n5. Structure verification"
        }
        return protocols[task_type]

    def _get_regulatory_info(self, task_type: str) -> str:
        """Get relevant regulatory compliance information."""
        regulations = {
            "Medical": "FDA 21 CFR Part 11, ISO 13485, GMP guidelines",
            "Environmental": "EPA guidelines, ISO 14001, REACH compliance",
            "Structural": "ASTM standards, ISO 10993, Material safety guidelines"
        }
        return regulations[task_type]
