import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass
from .input_handler import TaskParameters, InputHandler

@dataclass
class OptimizationConstraints:
    min_size: float
    max_size: float
    min_energy: float
    max_energy: float
    task_specific_constraints: Dict[str, Tuple[float, float]]
    production_constraints: Dict[str, float]

class TaskOptimizer:
    def __init__(self):
        self.task_weights = {
            "Medical": {
                "biocompatibility": 0.35,
                "targeting_efficiency": 0.25,
                "payload_delivery": 0.20,
                "stability": 0.20
            },
            "Environmental": {
                "pollutant_binding": 0.30,
                "degradation_rate": 0.25,
                "environmental_safety": 0.25,
                "stability": 0.20
            },
            "Structural": {
                "mechanical_strength": 0.30,
                "assembly_efficiency": 0.25,
                "stability": 0.25,
                "thermal_resistance": 0.20
            }
        }
        
        self.production_thresholds = {
            "stability_minimum": 0.65,
            "coherence_minimum": 0.60,
            "reactivity_maximum": 0.40,
            "energy_tolerance": 0.15
        }
        
        self.input_handler = InputHandler()

    def optimize(self, quantum_results: Dict, task_params: Union[Dict, TaskParameters]) -> Dict:
        """Multi-objective optimization with real-world constraints."""
        # Convert dictionary to TaskParameters if necessary
        if isinstance(task_params, dict):
            task_params = self.input_handler.collect_parameters(task_params)
        
        # Extract relevant properties
        molecular_properties = quantum_results['molecular_properties']
        state_vector = quantum_results['state_vector']
        
        # Initialize constraints with production requirements
        constraints = self._initialize_constraints(task_params)
        
        # Add task type to results
        quantum_results['task_type'] = task_params.task_type
        
        # Perform multi-objective optimization
        optimized_config = self._optimize_configuration(
            molecular_properties,
            state_vector,
            task_params,
            constraints
        )
        
        # Validate production feasibility
        feasibility_report = self._validate_production_feasibility(optimized_config)
        
        # Calculate final scores
        optimization_score = self._calculate_optimization_score(
            optimized_config,
            task_params.task_type
        )
        
        # Combine results with feasibility metrics
        return {
            **quantum_results,
            'optimized_configuration': optimized_config,
            'optimization_score': optimization_score,
            'feasibility_report': feasibility_report,
            'production_metrics': self._calculate_production_metrics(optimized_config)
        }

    def _initialize_constraints(self, task_params: TaskParameters) -> OptimizationConstraints:
        """Initialize comprehensive constraints including production requirements."""
        base_constraints = OptimizationConstraints(
            min_size=1e-9,  # 1 nm
            max_size=100e-9,  # 100 nm
            min_energy=0.0,
            max_energy=1.0,
            task_specific_constraints={},
            production_constraints=self.production_thresholds.copy()
        )
        
        # Add task-specific constraints with real-world considerations
        if task_params.task_type == "Medical":
            base_constraints.task_specific_constraints.update({
                "biocompatibility": (0.75, 1.0),  # Increased minimum for safety
                "targeting_efficiency": (0.70, 1.0),
                "payload_delivery": (0.65, 1.0),
                "clearance_rate": (0.50, 0.80)
            })
            base_constraints.production_constraints.update({
                "toxicity_threshold": 0.15,
                "immune_response_threshold": 0.20
            })
        elif task_params.task_type == "Environmental":
            base_constraints.task_specific_constraints.update({
                "pollutant_binding": (0.70, 1.0),
                "degradation_rate": (0.45, 0.75),
                "environmental_safety": (0.85, 1.0),
                "weather_resistance": (0.70, 1.0)
            })
            base_constraints.production_constraints.update({
                "temperature_tolerance": 0.75,
                "ph_tolerance": 0.70
            })
        else:  # Structural
            base_constraints.task_specific_constraints.update({
                "mechanical_strength": (0.80, 1.0),
                "assembly_efficiency": (0.70, 1.0),
                "stability": (0.85, 1.0),
                "stress_tolerance": (0.75, 1.0)
            })
            base_constraints.production_constraints.update({
                "load_bearing_capacity": 0.80,
                "fatigue_resistance": 0.75
            })
        return base_constraints

    def _create_initial_guess(self, molecular_properties: Dict) -> np.ndarray:
        """Create initial guess for optimization parameters."""
        return np.array([
            50e-9,  # Initial size
            0.5,    # Initial energy level
            molecular_properties['stability_index'],
            molecular_properties['reactivity_measure'],
            molecular_properties['coherence_measure']
        ])

    def _create_optimization_bounds(self, constraints: OptimizationConstraints) -> List[Tuple[float, float]]:
        """Create bounds for optimization parameters."""
        return [
            (constraints.min_size, constraints.max_size),
            (constraints.min_energy, constraints.max_energy),
            (0.0, 1.0),  # Stability
            (0.0, 1.0),  # Reactivity
            (0.0, 1.0)   # Coherence
        ]

    def _optimize_configuration(
        self,
        molecular_properties: Dict,
        state_vector: np.ndarray,
        task_params: TaskParameters,
        constraints: OptimizationConstraints
    ) -> Dict:
        """Optimize molecular configuration with multi-objective consideration."""
        def objective_function(x: np.ndarray) -> float:
            config = self._create_configuration(x, molecular_properties)
            
            # Calculate primary objectives with weighted importance
            stability_score = config['stability_index'] * 1.2  # Increased weight
            task_score = self._calculate_task_score(config, task_params.task_type)
            production_score = (
                0.5 * self._calculate_manufacturability(config) +  # Increased weight
                0.3 * self._calculate_scalability(config) +
                0.2 * self._calculate_cost_efficiency(config)
            )
            
            # Weighted combination of objectives with stability bias
            total_score = (
                0.45 * stability_score +  # Increased weight
                0.35 * task_score +
                0.20 * production_score
            )
            
            # Enhanced penalty calculation
            penalty = self._calculate_constraint_violations(config, constraints) * 1.5  # Increased penalty
            
            return -total_score + penalty  # Negative because we maximize
        
        # Initial guess based on molecular properties with stability bias
        x0 = self._create_initial_guess(molecular_properties)
        
        # Optimization bounds
        bounds = self._create_optimization_bounds(constraints)
        
        # Use differential evolution with enhanced settings
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=300,      # Increased iterations
            popsize=30,       # Increased population
            mutation=(0.6, 1.4),  # Adjusted mutation range
            recombination=0.9,    # Higher recombination
            seed=42,
            tol=0.0001,      # Tighter tolerance
            polish=True,      # Enable polishing
            strategy='best1bin',
            updating='immediate',
            disp=True        # Show progress
        )
        
        return self._create_configuration(result.x, molecular_properties)

    def _create_configuration(self, params: np.ndarray, molecular_properties: Dict) -> Dict:
        """Create a complete molecular configuration."""
        config = {
            'size': params[0],
            'energy_level': params[1],
            'stability_index': molecular_properties['stability_index'],
            'reactivity_measure': molecular_properties['reactivity_measure'],
            'coherence_measure': molecular_properties['coherence_measure'],
            'electron_density': molecular_properties['electron_density'],
            'bond_strength': molecular_properties['bond_strength']
        }
        
        # Add derived properties
        config.update({
            'thermal_stability': self._calculate_thermal_stability(config),
            'chemical_stability': self._calculate_chemical_stability(config),
            'structural_integrity': self._calculate_structural_integrity(config)
        })
        
        return config

    def _calculate_production_metrics(self, config: Dict) -> Dict:
        """Calculate comprehensive production-related metrics."""
        return {
            'manufacturability_score': self._calculate_manufacturability(config),
            'scalability_score': self._calculate_scalability(config),
            'cost_efficiency': self._calculate_cost_efficiency(config),
            'quality_consistency': self._calculate_quality_consistency(config),
            'production_yield': self._calculate_production_yield(config)
        }

    def _calculate_manufacturability(self, config: Dict) -> float:
        """Calculate manufacturability score based on multiple factors."""
        stability_factor = config['stability_index'] ** 1.5
        complexity_factor = 1 - (config['reactivity_measure'] * 0.5)
        consistency_factor = config['coherence_measure'] ** 1.2
        
        return np.clip(
            0.4 * stability_factor +
            0.3 * complexity_factor +
            0.3 * consistency_factor,
            0, 1
        )

    def _calculate_scalability(self, config: Dict) -> float:
        """Assess production scalability."""
        return np.clip(
            0.4 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']) +
            0.3 * config['bond_strength'],
            0, 1
        )

    def _calculate_cost_efficiency(self, config: Dict) -> float:
        """Estimate production cost efficiency."""
        complexity_penalty = config['reactivity_measure'] * 0.3
        stability_bonus = config['stability_index'] * 0.4
        yield_factor = self._calculate_production_yield(config) * 0.3
        
        return np.clip(
            1 - complexity_penalty + stability_bonus + yield_factor,
            0, 1
        )

    def _calculate_quality_consistency(self, config: Dict) -> float:
        """Estimate production quality consistency."""
        return np.clip(
            0.4 * config['coherence_measure'] +
            0.3 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']),
            0, 1
        )

    def _calculate_production_yield(self, config: Dict) -> float:
        """Estimate production yield rate."""
        stability_factor = config['stability_index'] ** 1.2
        reactivity_penalty = config['reactivity_measure'] * 0.2
        coherence_factor = config['coherence_measure'] * 0.3
        
        return np.clip(
            stability_factor - reactivity_penalty + coherence_factor,
            0, 1
        )

    def _validate_production_feasibility(self, config: Dict) -> Dict:
        """Validate configuration against production requirements."""
        feasibility_checks = {
            'stability_check': config['stability_index'] >= self.production_thresholds['stability_minimum'],
            'coherence_check': config['coherence_measure'] >= self.production_thresholds['coherence_minimum'],
            'reactivity_check': config['reactivity_measure'] <= self.production_thresholds['reactivity_maximum'],
            'energy_check': abs(config['energy_level']) <= self.production_thresholds['energy_tolerance']
        }
        
        feasibility_score = sum(feasibility_checks.values()) / len(feasibility_checks)
        
        return {
            'checks': feasibility_checks,
            'overall_score': feasibility_score,
            'recommendations': self._generate_feasibility_recommendations(config, feasibility_checks)
        }

    def _generate_feasibility_recommendations(self, config: Dict, checks: Dict) -> List[str]:
        """Generate specific recommendations for improving production feasibility."""
        recommendations = []
        
        if not checks['stability_check']:
            recommendations.append(
                f"Increase stability index (current: {config['stability_index']:.2f}) "
                f"to meet minimum threshold of {self.production_thresholds['stability_minimum']}"
            )
            
        if not checks['coherence_check']:
            recommendations.append(
                f"Improve coherence measure (current: {config['coherence_measure']:.2f}) "
                f"to meet minimum threshold of {self.production_thresholds['coherence_minimum']}"
            )
            
        if not checks['reactivity_check']:
            recommendations.append(
                f"Reduce reactivity measure (current: {config['reactivity_measure']:.2f}) "
                f"to below maximum threshold of {self.production_thresholds['reactivity_maximum']}"
            )
            
        return recommendations

    def _calculate_optimization_score(self, config: Dict, task_type: str) -> float:
        """Calculate the final optimization score."""
        # Calculate base score from molecular properties
        base_score = (
            config['stability_index'] * 0.4 +
            config['coherence_measure'] * 0.3 +
            (1 - config['reactivity_measure']) * 0.3
        )
        
        # Calculate task-specific score
        if task_type == "Medical":
            task_score = (
                self._calculate_biocompatibility(config) * 0.4 +
                self._calculate_targeting_efficiency(config) * 0.3 +
                self._calculate_payload_capacity(config) * 0.3
            )
        elif task_type == "Environmental":
            task_score = (
                self._calculate_binding_efficiency(config) * 0.4 +
                (1 - self._calculate_degradation_rate(config)) * 0.3 +
                (1 - self._calculate_environmental_impact(config)) * 0.3
            )
        else:  # Structural
            task_score = (
                self._calculate_mechanical_strength(config) * 0.4 +
                self._calculate_assembly_efficiency(config) * 0.3 +
                self._calculate_structural_integrity(config) * 0.3
            )
        
        # Combine scores and ensure positive result
        combined_score = (0.6 * base_score + 0.4 * task_score)
        
        # Convert to positive score (0 to 1 range)
        return (combined_score + 1) / 2

    def _calculate_biocompatibility(self, config: Dict) -> float:
        """Calculate biocompatibility with production considerations."""
        return (
            0.4 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']) +
            0.3 * config['coherence_measure']
        ) ** 1.5

    def _calculate_targeting_efficiency(self, config: Dict) -> float:
        """Calculate targeting efficiency for production."""
        return np.clip(
            0.4 * config['coherence_measure'] +
            0.3 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']),
            0, 1
        )

    def _calculate_payload_capacity(self, config: Dict) -> float:
        """Calculate payload capacity with size considerations."""
        size_normalized = config['size'] / 100e-9
        return size_normalized * config['stability_index']

    def _calculate_binding_efficiency(self, config: Dict) -> float:
        """Calculate binding efficiency for production."""
        return np.clip(
            0.4 * config['reactivity_measure'] +
            0.3 * config['coherence_measure'] +
            0.3 * config['stability_index'],
            0, 1
        )

    def _calculate_degradation_rate(self, config: Dict) -> float:
        """Calculate degradation rate for production conditions."""
        return np.clip(
            0.5 * config['reactivity_measure'] +
            0.3 * (1 - config['stability_index']) +
            0.2 * (1 - config['bond_strength']),
            0, 1
        )

    def _calculate_environmental_impact(self, config: Dict) -> float:
        """Calculate environmental impact for production."""
        return np.clip(
            0.4 * (1 - config['stability_index']) +
            0.3 * config['reactivity_measure'] +
            0.3 * (1 - config['coherence_measure']),
            0, 1
        )

    def _calculate_mechanical_strength(self, config: Dict) -> float:
        """Calculate mechanical strength for production."""
        return np.clip(
            0.4 * config['stability_index'] +
            0.3 * config['bond_strength'] +
            0.3 * (1 - config['reactivity_measure']),
            0, 1
        )

    def _calculate_assembly_efficiency(self, config: Dict) -> float:
        """Calculate assembly efficiency for production."""
        return np.clip(
            0.4 * config['coherence_measure'] +
            0.3 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']),
            0, 1
        )

    def _calculate_task_score(self, config: Dict, task_type: str) -> float:
        """Calculate task-specific optimization score."""
        weights = self.task_weights[task_type]
        
        if task_type == "Medical":
            return (
                weights['biocompatibility'] * self._calculate_biocompatibility(config) +
                weights['targeting_efficiency'] * self._calculate_targeting_efficiency(config) +
                weights['payload_delivery'] * self._calculate_payload_capacity(config) +
                weights['stability'] * config['stability_index']
            )
        elif task_type == "Environmental":
            return (
                weights['pollutant_binding'] * self._calculate_binding_efficiency(config) +
                weights['degradation_rate'] * (1 - self._calculate_degradation_rate(config)) +
                weights['environmental_safety'] * (1 - self._calculate_environmental_impact(config)) +
                weights['stability'] * config['stability_index']
            )
        else:  # Structural
            return (
                weights['mechanical_strength'] * self._calculate_mechanical_strength(config) +
                weights['assembly_efficiency'] * self._calculate_assembly_efficiency(config) +
                weights['stability'] * config['stability_index'] +
                weights['thermal_resistance'] * config['thermal_stability']
            )

    def _calculate_constraint_violations(self, config: Dict, constraints: OptimizationConstraints) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Size constraints
        if config['size'] < constraints.min_size or config['size'] > constraints.max_size:
            penalty += 100.0
            
        # Energy constraints
        if config['energy_level'] < constraints.min_energy or config['energy_level'] > constraints.max_energy:
            penalty += 100.0
            
        # Task-specific constraints
        for param, (min_val, max_val) in constraints.task_specific_constraints.items():
            if param in config:
                value = config[param]
                if value < min_val:
                    penalty += 50.0 * (min_val - value)
                elif value > max_val:
                    penalty += 50.0 * (value - max_val)
                    
        return penalty

    def _calculate_thermal_stability(self, config: Dict) -> float:
        """Calculate thermal stability for production conditions."""
        return np.clip(
            0.5 * config['stability_index'] +
            0.3 * config['bond_strength'] +
            0.2 * (1 - config['reactivity_measure']),
            0, 1
        )

    def _calculate_chemical_stability(self, config: Dict) -> float:
        """Calculate chemical stability under production conditions."""
        return np.clip(
            0.4 * config['stability_index'] +
            0.3 * (1 - config['reactivity_measure']) +
            0.3 * config['coherence_measure'],
            0, 1
        )

    def _calculate_structural_integrity(self, config: Dict) -> float:
        """Calculate structural integrity for production processes."""
        return np.clip(
            0.4 * config['stability_index'] +
            0.3 * config['bond_strength'] +
            0.3 * config['coherence_measure'],
            0, 1
        )
