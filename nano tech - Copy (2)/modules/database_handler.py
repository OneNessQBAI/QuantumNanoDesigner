import pubchempy as pcp
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import requests
import json

class MolecularDatabase:
    def __init__(self):
        self.pubchem_cache = {}
        self.pdb_cache = {}
        self.materials_cache = {}
        
    def fetch_molecular_data(self, compound_name: str) -> Dict:
        """Fetch molecular data from PubChem database."""
        try:
            if compound_name in self.pubchem_cache:
                return self.pubchem_cache[compound_name]
            
            # Search PubChem for compound
            compounds = pcp.get_compounds(compound_name, 'name')
            if not compounds:
                return None
            
            compound = compounds[0]
            mol = Chem.MolFromSmiles(compound.canonical_smiles)
            
            # Calculate molecular properties
            properties = {
                'molecular_weight': compound.molecular_weight,
                'xlogp': compound.xlogp,
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'hbd': compound.hbond_donor_count,
                'hba': compound.hbond_acceptor_count,
                'tpsa': Descriptors.TPSA(mol),
                'charge': Chem.GetFormalCharge(mol),
                'rings': Descriptors.RingCount(mol)
            }
            
            # Calculate quantum properties
            quantum_props = self._calculate_quantum_properties(mol)
            properties.update(quantum_props)
            
            self.pubchem_cache[compound_name] = properties
            return properties
            
        except Exception as e:
            print(f"Error fetching PubChem data: {str(e)}")
            return None

    def fetch_protein_data(self, pdb_id: str) -> Dict:
        """Fetch protein structure data from Protein Data Bank."""
        try:
            if pdb_id in self.pdb_cache:
                return self.pdb_cache[pdb_id]
            
            # Fetch PDB data
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url)
            if response.status_code != 200:
                return None
            
            # Parse PDB file
            structure_data = {
                'sequence': self._extract_sequence(response.text),
                'secondary_structure': self._analyze_secondary_structure(response.text),
                'binding_sites': self._identify_binding_sites(response.text)
            }
            
            self.pdb_cache[pdb_id] = structure_data
            return structure_data
            
        except Exception as e:
            print(f"Error fetching PDB data: {str(e)}")
            return None

    def fetch_materials_data(self, material_id: str) -> Dict:
        """Fetch materials data from Materials Project database."""
        try:
            if material_id in self.materials_cache:
                return self.materials_cache[material_id]
            
            # Materials Project API endpoint
            api_key = "YOUR_MATERIALS_PROJECT_API_KEY"  # Replace with actual API key
            url = f"https://materialsproject.org/rest/v2/materials/{material_id}/vasp"
            headers = {"X-API-KEY": api_key}
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Extract relevant properties
            properties = {
                'formula': data['pretty_formula'],
                'band_gap': data['band_gap'],
                'density': data['density'],
                'elastic_properties': data.get('elastic', {}),
                'formation_energy': data['formation_energy_per_atom']
            }
            
            self.materials_cache[material_id] = properties
            return properties
            
        except Exception as e:
            print(f"Error fetching materials data: {str(e)}")
            return None

    def _calculate_quantum_properties(self, mol) -> Dict:
        """Calculate quantum mechanical properties using RDKit."""
        try:
            # Generate 3D coordinates
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Calculate quantum properties
            properties = {
                'stability_index': self._estimate_stability(mol),
                'reactivity_measure': self._estimate_reactivity(mol),
                'coherence_measure': self._estimate_coherence(mol),
                'bond_strength': self._estimate_bond_strength(mol)
            }
            
            return properties
            
        except Exception as e:
            print(f"Error calculating quantum properties: {str(e)}")
            return {}

    def _estimate_stability(self, mol) -> float:
        """Estimate molecular stability."""
        try:
            # Use molecular descriptors to estimate stability
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            molecular_weight = Descriptors.ExactMolWt(mol)
            
            # Normalize and combine factors
            stability = (
                0.4 * (1 - np.exp(-aromatic_rings/2)) +  # Aromatic stability
                0.3 * (1 - rotatable_bonds/20) +         # Conformational stability
                0.3 * (1 - np.exp(-molecular_weight/500)) # Size-based stability
            )
            
            return np.clip(stability, 0, 1)
            
        except:
            return 0.5  # Default value

    def _estimate_reactivity(self, mol) -> float:
        """Estimate molecular reactivity."""
        try:
            # Calculate reactivity based on molecular features
            charge = abs(Chem.GetFormalCharge(mol))
            radical_electrons = Descriptors.NumRadicalElectrons(mol)
            surface_area = Descriptors.TPSA(mol)
            
            # Normalize and combine factors
            reactivity = (
                0.4 * (charge/2) +                    # Charge contribution
                0.4 * (radical_electrons/2) +         # Radical contribution
                0.2 * (surface_area/200)             # Surface area contribution
            )
            
            return np.clip(reactivity, 0, 1)
            
        except:
            return 0.5  # Default value

    def _estimate_coherence(self, mol) -> float:
        """Estimate quantum coherence."""
        try:
            # Use molecular symmetry and conjugation as coherence indicators
            conjugated_bonds = Descriptors.NumAliphaticCarbocycles(mol)
            symmetry_score = self._calculate_symmetry_score(mol)
            
            coherence = (
                0.6 * symmetry_score +
                0.4 * (1 - np.exp(-conjugated_bonds/5))
            )
            
            return np.clip(coherence, 0, 1)
            
        except:
            return 0.5  # Default value

    def _estimate_bond_strength(self, mol) -> float:
        """Estimate average bond strength."""
        try:
            # Calculate bond strength based on bond types
            bonds = mol.GetBonds()
            total_strength = 0
            
            for bond in bonds:
                if bond.GetBondType() == Chem.BondType.SINGLE:
                    strength = 1
                elif bond.GetBondType() == Chem.BondType.DOUBLE:
                    strength = 2
                elif bond.GetBondType() == Chem.BondType.TRIPLE:
                    strength = 3
                elif bond.GetBondType() == Chem.BondType.AROMATIC:
                    strength = 1.5
                else:
                    strength = 1
                    
                total_strength += strength
            
            avg_strength = total_strength / len(bonds) if bonds else 1
            return np.clip(avg_strength/3, 0, 1)  # Normalize to [0,1]
            
        except:
            return 0.5  # Default value

    def _calculate_symmetry_score(self, mol) -> float:
        """Calculate molecular symmetry score."""
        try:
            # Use RDKit's symmetry perception
            symmetry_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
            unique_classes = len(set(symmetry_classes))
            total_atoms = mol.GetNumAtoms()
            
            # More unique classes = less symmetry
            symmetry_score = 1 - (unique_classes / total_atoms)
            return np.clip(symmetry_score, 0, 1)
            
        except:
            return 0.5  # Default value

    def _extract_sequence(self, pdb_text: str) -> str:
        """Extract protein sequence from PDB file."""
        sequence = ""
        for line in pdb_text.split('\n'):
            if line.startswith("SEQRES"):
                sequence += line[19:].strip()
        return sequence

    def _analyze_secondary_structure(self, pdb_text: str) -> Dict:
        """Analyze protein secondary structure."""
        structure = {
            'helix': 0,
            'sheet': 0,
            'loop': 0
        }
        
        for line in pdb_text.split('\n'):
            if line.startswith("HELIX"):
                structure['helix'] += 1
            elif line.startswith("SHEET"):
                structure['sheet'] += 1
                
        return structure

    def _identify_binding_sites(self, pdb_text: str) -> List[Dict]:
        """Identify potential binding sites."""
        binding_sites = []
        current_site = None
        
        for line in pdb_text.split('\n'):
            if line.startswith("SITE"):
                site_id = line[11:14].strip()
                residues = line[18:].strip()
                
                if current_site and current_site['site_id'] != site_id:
                    binding_sites.append(current_site)
                    current_site = None
                
                if not current_site:
                    current_site = {
                        'site_id': site_id,
                        'residues': []
                    }
                
                current_site['residues'].extend(residues.split())
        
        if current_site:
            binding_sites.append(current_site)
            
        return binding_sites
