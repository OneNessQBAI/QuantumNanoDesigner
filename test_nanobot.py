import unittest
import numpy as np
from modules.input_handler import InputHandler, TaskParameters
from modules.quantum_simulator import QuantumSimulator
from modules.optimization import TaskOptimizer
from modules.visualization import MoleculeVisualizer

class TestNanobotDesigner(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.input_handler = InputHandler()
        self.quantum_simulator = QuantumSimulator()
        self.optimizer = TaskOptimizer()
        self.visualizer = MoleculeVisualizer()

    def test_input_handler(self):
        """Test input handler parameter processing."""
        # Test medical task parameters
        raw_params = {
            'task_type': 'Medical',
            'size_constraint': 50,
            'energy_efficiency': 0.7,
            'biocompatibility': 0.8,
            'targeting_specificity': 0.7,
            'payload_capacity': 0.5
        }
        
        params = self.input_handler.collect_parameters(raw_params)
        self.assertEqual(params.task_type, 'Medical')
        self.assertEqual(params.size_constraint, 50)
        self.assertEqual(params.biocompatibility, 0.8)

    def test_quantum_simulator(self):
        """Test quantum simulation functionality."""
        # Create test parameters
        params = TaskParameters(
            task_type='Medical',
            size_constraint=50,
            energy_efficiency=0.7,
            biocompatibility=0.8,
            targeting_specificity=0.7,
            payload_capacity=0.5
        )
        
        # Run simulation
        results = self.quantum_simulator.run_simulation(params)
        
        # Check results structure
        self.assertIn('molecular_properties', results)
        self.assertIn('state_vector', results)
        self.assertIn('final_energy', results)

    def test_optimization(self):
        """Test optimization functionality."""
        # Create test quantum results
        quantum_results = {
            'molecular_properties': {
                'stability_index': 0.8,
                'reactivity_measure': 0.3,
                'coherence_measure': 0.7,
                'energy_levels': [-0.5, -0.3, -0.1, 0.1, 0.3]
            },
            'state_vector': np.random.random(64) + 1j * np.random.random(64),
            'final_energy': -1.5
        }
        
        # Create test parameters
        task_params = {
            'task_type': 'Medical',
            'size_constraint': 50,
            'energy_efficiency': 0.7
        }
        
        # Run optimization
        results = self.optimizer.optimize(quantum_results, task_params)
        
        # Check optimization results
        self.assertIn('optimized_configuration', results)
        self.assertIn('optimization_score', results)

    def test_medical_task_workflow(self):
        """Test complete workflow for medical task."""
        # Create input parameters
        raw_params = {
            'task_type': 'Medical',
            'size_constraint': 50,
            'energy_efficiency': 0.7,
            'biocompatibility': 0.8,
            'targeting_specificity': 0.7,
            'payload_capacity': 0.5
        }
        
        # Process parameters
        params = self.input_handler.collect_parameters(raw_params)
        
        # Run quantum simulation
        quantum_results = self.quantum_simulator.run_simulation(params)
        
        # Optimize results
        optimized_results = self.optimizer.optimize(quantum_results, raw_params)
        
        # Check final results
        self.assertIn('optimized_configuration', optimized_results)
        self.assertIn('optimization_score', optimized_results)
        self.assertGreater(optimized_results['optimization_score'], 0)

    def test_environmental_task_workflow(self):
        """Test complete workflow for environmental task."""
        # Create input parameters
        raw_params = {
            'task_type': 'Environmental',
            'size_constraint': 40,
            'energy_efficiency': 0.8,
            'pollutant_affinity': 0.8,
            'environmental_stability': 0.7
        }
        
        # Process parameters
        params = self.input_handler.collect_parameters(raw_params)
        
        # Run quantum simulation
        quantum_results = self.quantum_simulator.run_simulation(params)
        
        # Optimize results
        optimized_results = self.optimizer.optimize(quantum_results, raw_params)
        
        # Check final results
        self.assertIn('optimized_configuration', optimized_results)
        self.assertIn('optimization_score', optimized_results)
        self.assertGreater(optimized_results['optimization_score'], 0)

    def test_structural_task_workflow(self):
        """Test complete workflow for structural task."""
        # Create input parameters
        raw_params = {
            'task_type': 'Structural',
            'size_constraint': 60,
            'energy_efficiency': 0.9,
            'mechanical_strength': 0.8,
            'flexibility': 0.6
        }
        
        # Process parameters
        params = self.input_handler.collect_parameters(raw_params)
        
        # Run quantum simulation
        quantum_results = self.quantum_simulator.run_simulation(params)
        
        # Optimize results
        optimized_results = self.optimizer.optimize(quantum_results, raw_params)
        
        # Check final results
        self.assertIn('optimized_configuration', optimized_results)
        self.assertIn('optimization_score', optimized_results)
        self.assertGreater(optimized_results['optimization_score'], 0)

if __name__ == '__main__':
    unittest.main()
