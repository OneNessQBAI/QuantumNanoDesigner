import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TaskParameters:
    task_type: str
    size_constraint: float
    energy_efficiency: float
    
    # Medical parameters
    biocompatibility: float = 0.0
    targeting_specificity: float = 0.0
    payload_capacity: float = 0.0
    
    # Environmental parameters
    pollutant_affinity: float = 0.0
    environmental_stability: float = 0.0
    
    # Structural parameters
    mechanical_strength: float = 0.0
    flexibility: float = 0.0

    def generate_constraints_matrix(self) -> np.ndarray:
        """Generate constraints matrix for quantum optimization."""
        # Basic constraint matrix based on common parameters
        constraints = np.array([
            self.size_constraint / 100,  # Normalize size constraint
            self.energy_efficiency
        ])
        
        # Add task-specific constraints
        if self.task_type == "Medical":
            task_constraints = np.array([
                self.biocompatibility,
                self.targeting_specificity,
                self.payload_capacity
            ])
        elif self.task_type == "Environmental":
            task_constraints = np.array([
                self.pollutant_affinity,
                self.environmental_stability
            ])
        else:  # Structural
            task_constraints = np.array([
                self.mechanical_strength,
                self.flexibility
            ])
            
        return np.concatenate([constraints, task_constraints])

class InputHandler:
    def __init__(self):
        self.valid_task_types = ["Medical", "Environmental", "Structural"]
        
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        if params['task_type'] not in self.valid_task_types:
            raise ValueError(f"Invalid task type: {params['task_type']}")
            
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if key == 'size_constraint':
                    if not (1 <= value <= 100):
                        raise ValueError(f"Size constraint must be between 1 and 100 nm")
                else:
                    if not (0 <= value <= 1):
                        raise ValueError(f"Parameter {key} must be between 0 and 1")
        return True

    def collect_parameters(self, raw_params: Dict[str, Any]) -> TaskParameters:
        """Collect and process input parameters."""
        # Extract relevant parameters based on task type
        task_type = raw_params['task_type']
        params = {
            'task_type': task_type,
            'size_constraint': raw_params['size_constraint'],
            'energy_efficiency': raw_params['energy_efficiency']
        }
        
        # Add task-specific parameters
        if task_type == "Medical":
            params.update({
                'biocompatibility': raw_params['biocompatibility'],
                'targeting_specificity': raw_params['targeting_specificity'],
                'payload_capacity': raw_params['payload_capacity']
            })
        elif task_type == "Environmental":
            params.update({
                'pollutant_affinity': raw_params['pollutant_affinity'],
                'environmental_stability': raw_params['environmental_stability']
            })
        elif task_type == "Structural":
            params.update({
                'mechanical_strength': raw_params['mechanical_strength'],
                'flexibility': raw_params['flexibility']
            })
            
        # Validate parameters
        self.validate_parameters(params)
        
        # Create and return TaskParameters object
        return TaskParameters(**params)
