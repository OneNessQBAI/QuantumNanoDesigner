import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class TaskParameters:
    task_type: str
    size_constraint: float
    energy_efficiency: float
    
    # Medical parameters
    biocompatibility: float = 0.0
    targeting_specificity: float = 0.0
    payload_capacity: float = 0.0
    clearance_rate: float = 0.0
    immune_response: float = 0.0
    
    # Environmental parameters
    pollutant_affinity: float = 0.0
    environmental_stability: float = 0.0
    degradation_rate: float = 0.0
    weather_resistance: float = 0.0
    ph_tolerance: float = 0.0
    
    # Structural parameters
    mechanical_strength: float = 0.0
    flexibility: float = 0.0
    thermal_stability: float = 0.0
    stress_tolerance: float = 0.0
    fatigue_resistance: float = 0.0

    def generate_constraints_matrix(self) -> np.ndarray:
        """Generate comprehensive constraints matrix for quantum optimization."""
        # Base constraints for all types
        base_constraints = np.array([
            self.size_constraint / 100,  # Normalize size constraint
            self.energy_efficiency
        ])
        
        # Task-specific constraints with real-world considerations
        if self.task_type == "Medical":
            task_constraints = np.array([
                self.biocompatibility,
                self.targeting_specificity,
                self.payload_capacity,
                self.clearance_rate,
                self.immune_response
            ])
        elif self.task_type == "Environmental":
            task_constraints = np.array([
                self.pollutant_affinity,
                self.environmental_stability,
                self.degradation_rate,
                self.weather_resistance,
                self.ph_tolerance
            ])
        else:  # Structural
            task_constraints = np.array([
                self.mechanical_strength,
                self.flexibility,
                self.thermal_stability,
                self.stress_tolerance,
                self.fatigue_resistance
            ])
            
        return np.concatenate([base_constraints, task_constraints])

class InputHandler:
    def __init__(self):
        self.valid_task_types = ["Medical", "Environmental", "Structural"]
        self.load_validation_rules()
        
    def load_validation_rules(self):
        """Load validation rules for parameters."""
        self.validation_rules = {
            "Medical": {
                "biocompatibility": (0.7, 1.0),
                "targeting_specificity": (0.6, 1.0),
                "payload_capacity": (0.4, 0.8),
                "clearance_rate": (0.3, 0.7),
                "immune_response": (0.0, 0.3)
            },
            "Environmental": {
                "pollutant_affinity": (0.6, 1.0),
                "environmental_stability": (0.7, 1.0),
                "degradation_rate": (0.4, 0.8),
                "weather_resistance": (0.6, 1.0),
                "ph_tolerance": (0.5, 1.0)
            },
            "Structural": {
                "mechanical_strength": (0.7, 1.0),
                "flexibility": (0.5, 0.9),
                "thermal_stability": (0.6, 1.0),
                "stress_tolerance": (0.7, 1.0),
                "fatigue_resistance": (0.6, 1.0)
            }
        }

        # Common parameter rules
        self.common_rules = {
            "size_constraint": (1, 100),  # nm
            "energy_efficiency": (0.0, 1.0)
        }

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters against real-world constraints."""
        if params['task_type'] not in self.valid_task_types:
            raise ValueError(f"Invalid task type: {params['task_type']}")
            
        # Validate common parameters
        for param, (min_val, max_val) in self.common_rules.items():
            if param in params:
                value = params[param]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{param} must be between {min_val} and {max_val} "
                        f"(got {value})"
                    )
            
        # Validate task-specific parameters
        task_rules = self.validation_rules[params['task_type']]
        for param, (min_val, max_val) in task_rules.items():
            if param in params:
                value = params[param]
                if not (min_val <= value <= max_val):
                    raise ValueError(
                        f"{param} must be between {min_val} and {max_val} "
                        f"(got {value})"
                    )
        
        return True

    def collect_parameters(self, raw_params: Dict[str, Any]) -> TaskParameters:
        """Collect and process input parameters with real-world validation."""
        # Extract base parameters
        task_type = raw_params['task_type']
        params = {
            'task_type': task_type,
            'size_constraint': raw_params['size_constraint'],
            'energy_efficiency': raw_params['energy_efficiency']
        }
        
        # Add task-specific parameters with defaults
        if task_type == "Medical":
            params.update({
                'biocompatibility': raw_params.get('biocompatibility', 0.7),
                'targeting_specificity': raw_params.get('targeting_specificity', 0.6),
                'payload_capacity': raw_params.get('payload_capacity', 0.5),
                'clearance_rate': raw_params.get('clearance_rate', 0.4),
                'immune_response': raw_params.get('immune_response', 0.2)
            })
        elif task_type == "Environmental":
            params.update({
                'pollutant_affinity': raw_params.get('pollutant_affinity', 0.7),
                'environmental_stability': raw_params.get('environmental_stability', 0.7),
                'degradation_rate': raw_params.get('degradation_rate', 0.5),
                'weather_resistance': raw_params.get('weather_resistance', 0.6),
                'ph_tolerance': raw_params.get('ph_tolerance', 0.6)
            })
        elif task_type == "Structural":
            params.update({
                'mechanical_strength': raw_params.get('mechanical_strength', 0.8),
                'flexibility': raw_params.get('flexibility', 0.6),
                'thermal_stability': raw_params.get('thermal_stability', 0.7),
                'stress_tolerance': raw_params.get('stress_tolerance', 0.7),
                'fatigue_resistance': raw_params.get('fatigue_resistance', 0.7)
            })
            
        # Validate parameters
        self.validate_parameters(params)
        
        # Create and return TaskParameters object
        return TaskParameters(**params)

    def get_parameter_recommendations(self, task_type: str) -> Dict[str, Dict[str, float]]:
        """Get recommended parameter ranges for a specific task type."""
        if task_type not in self.valid_task_types:
            raise ValueError(f"Invalid task type: {task_type}")
            
        task_rules = self.validation_rules[task_type]
        recommendations = {}
        
        # Add common parameters
        for param, (min_val, max_val) in self.common_rules.items():
            recommendations[param] = {
                'min': min_val,
                'max': max_val,
                'recommended': (min_val + max_val) / 2
            }
            
        # Add task-specific parameters
        for param, (min_val, max_val) in task_rules.items():
            recommendations[param] = {
                'min': min_val,
                'max': max_val,
                'recommended': (min_val + max_val) / 2
            }
            
        return recommendations
