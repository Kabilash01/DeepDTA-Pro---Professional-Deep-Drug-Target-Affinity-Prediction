"""
Molecular Visualization Module
3D and 2D visualization tools for drug molecules
"""

import streamlit as st
import py3Dmol
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io
import base64
from PIL import Image
import numpy as np
from typing import Optional, Tuple

def create_3d_molecule_viewer(smiles: str, width: int = 400, height: int = 400) -> str:
    """
    Create a 3D molecular viewer using py3Dmol
    
    Args:
        smiles: SMILES string of the molecule
        width: Width of the viewer
        height: Height of the viewer
    
    Returns:
        HTML string for the 3D viewer
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.OptimizeMolecule(mol)
        
        # Get SDF format for 3D visualization
        sdf_block = Chem.MolToMolBlock(mol)
        
        # Create py3Dmol viewer
        viewer_html = f"""
        <div id="3dmolviewer_{hash(smiles) % 10000}" style="height: {height}px; width: {width}px; position: relative;"></div>
        
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        
        <script>
        $(document).ready(function() {{
            let element = $('#3dmolviewer_{hash(smiles) % 10000}');
            let config = {{ backgroundColor: 'white' }};
            let viewer = $3Dmol.createViewer(element, config);
            
            let moldata = `{sdf_block}`;
            
            viewer.addModel(moldata, "sdf");
            viewer.setStyle({{}}, {{stick: {{colorscheme: "default", radius: 0.15}}, 
                                 sphere: {{colorscheme: "default", scale: 0.25}}}});
            viewer.addPropertyLabels("atom", {{fontColor: "black", fontOpacity: 0.8, fontSize: 12}});
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(1.2, 1000);
        }});
        </script>
        """
        
        return viewer_html
        
    except Exception as e:
        print(f"Error creating 3D viewer: {e}")
        return None

def create_2d_molecule_image(smiles: str, width: int = 300, height: int = 300) -> Optional[Image.Image]:
    """
    Create a 2D molecular structure image using RDKit
    
    Args:
        smiles: SMILES string of the molecule
        width: Width of the image
        height: Height of the image
    
    Returns:
        PIL Image object
    """
    try:
        # Convert SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create 2D drawing
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # Convert to PIL Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
        
    except Exception as e:
        print(f"Error creating 2D image: {e}")
        return None

def get_molecular_properties(smiles: str) -> dict:
    """
    Calculate molecular properties using RDKit
    
    Args:
        smiles: SMILES string of the molecule
    
    Returns:
        Dictionary of molecular properties
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        properties = {
            'Molecular Weight': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Descriptors.MolLogP(mol), 2),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Heavy Atoms': mol.GetNumHeavyAtoms(),
        }
        
        return properties
        
    except Exception as e:
        print(f"Error calculating properties: {e}")
        return {}

def display_molecule_viewer(smiles: str, title: str = "Molecular Structure"):
    """
    Display molecule with both 2D and 3D visualizations in Streamlit
    
    Args:
        smiles: SMILES string of the molecule
        title: Title for the visualization
    """
    st.subheader(title)
    
    # Validate SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["🧪 2D Structure", "🌐 3D Interactive", "📊 Properties"])
    
    with tab1:
        # 2D molecular structure
        img_2d = create_2d_molecule_image(smiles, width=400, height=400)
        if img_2d:
            st.image(img_2d, caption=f"2D Structure: {smiles}", use_column_width=True)
        else:
            st.error("Failed to generate 2D structure")
    
    with tab2:
        # 3D interactive viewer
        viewer_html = create_3d_molecule_viewer(smiles, width=500, height=400)
        if viewer_html:
            components.html(viewer_html, height=450)
            st.caption("🖱️ Click and drag to rotate, scroll to zoom")
        else:
            st.error("Failed to generate 3D structure")
    
    with tab3:
        # Molecular properties
        properties = get_molecular_properties(smiles)
        if properties:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Molecular Weight", f"{properties.get('Molecular Weight', 'N/A')} Da")
                st.metric("LogP", properties.get('LogP', 'N/A'))
                st.metric("TPSA", f"{properties.get('TPSA', 'N/A')} Ų")
                st.metric("Rotatable Bonds", properties.get('Rotatable Bonds', 'N/A'))
            
            with col2:
                st.metric("H-Bond Donors", properties.get('H-Bond Donors', 'N/A'))
                st.metric("H-Bond Acceptors", properties.get('H-Bond Acceptors', 'N/A'))
                st.metric("Aromatic Rings", properties.get('Aromatic Rings', 'N/A'))
                st.metric("Heavy Atoms", properties.get('Heavy Atoms', 'N/A'))
            
            # Drug-likeness assessment
            st.subheader("💊 Drug-likeness Assessment")
            
            # Lipinski's Rule of Five
            mw = properties.get('Molecular Weight', 0)
            logp = properties.get('LogP', 0)
            hbd = properties.get('H-Bond Donors', 0)
            hba = properties.get('H-Bond Acceptors', 0)
            
            lipinski_violations = 0
            lipinski_rules = []
            
            if mw > 500:
                lipinski_violations += 1
                lipinski_rules.append("❌ Molecular Weight > 500 Da")
            else:
                lipinski_rules.append("✅ Molecular Weight ≤ 500 Da")
            
            if logp > 5:
                lipinski_violations += 1
                lipinski_rules.append("❌ LogP > 5")
            else:
                lipinski_rules.append("✅ LogP ≤ 5")
            
            if hbd > 5:
                lipinski_violations += 1
                lipinski_rules.append("❌ H-Bond Donors > 5")
            else:
                lipinski_rules.append("✅ H-Bond Donors ≤ 5")
            
            if hba > 10:
                lipinski_violations += 1
                lipinski_rules.append("❌ H-Bond Acceptors > 10")
            else:
                lipinski_rules.append("✅ H-Bond Acceptors ≤ 10")
            
            # Display results
            if lipinski_violations == 0:
                st.success("🎉 Passes Lipinski's Rule of Five - Good drug-likeness!")
            elif lipinski_violations == 1:
                st.warning("⚠️ 1 Lipinski violation - Moderate drug-likeness")
            else:
                st.error(f"❌ {lipinski_violations} Lipinski violations - Poor drug-likeness")
            
            for rule in lipinski_rules:
                st.write(rule)
        else:
            st.error("Failed to calculate molecular properties")

def create_protein_structure_viewer(sequence: str, width: int = 500, height: int = 400) -> str:
    """
    Create a simple protein structure visualization
    Note: This creates a mock visualization since we don't have actual 3D structures
    """
    try:
        # Calculate some basic properties
        length = len(sequence)
        hydrophobic_aa = set('AILVMFWY')
        polar_aa = set('STYNQ')
        charged_aa = set('DEKHR')
        
        hydrophobic_count = sum(1 for aa in sequence if aa in hydrophobic_aa)
        polar_count = sum(1 for aa in sequence if aa in polar_aa)
        charged_count = sum(1 for aa in sequence if aa in charged_aa)
        
        # Create a simple helical representation
        viewer_html = f"""
        <div id="proteinviewer_{hash(sequence) % 10000}" style="height: {height}px; width: {width}px; position: relative; background: linear-gradient(45deg, #f0f8ff, #e6f3ff); border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-direction: column;">
            <div style="text-align: center; color: #333;">
                <h3>🧬 Protein Sequence Visualization</h3>
                <div style="margin: 20px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <div style="font-size: 14px; margin: 10px 0;">
                        <strong>Length:</strong> {length} amino acids
                    </div>
                    <div style="font-size: 14px; margin: 10px 0;">
                        <span style="color: #ff6b6b;">●</span> Hydrophobic: {hydrophobic_count} ({hydrophobic_count/length*100:.1f}%)
                    </div>
                    <div style="font-size: 14px; margin: 10px 0;">
                        <span style="color: #4ecdc4;">●</span> Polar: {polar_count} ({polar_count/length*100:.1f}%)
                    </div>
                    <div style="font-size: 14px; margin: 10px 0;">
                        <span style="color: #45b7d1;">●</span> Charged: {charged_count} ({charged_count/length*100:.1f}%)
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 100px; overflow-y: auto;">
                        {sequence[:100]}{"..." if len(sequence) > 100 else ""}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return viewer_html
        
    except Exception as e:
        print(f"Error creating protein viewer: {e}")
        return None

def display_protein_visualization(sequence: str, title: str = "Protein Structure"):
    """
    Display protein sequence visualization
    """
    st.subheader(title)
    
    if not sequence:
        st.error("❌ No protein sequence provided")
        return
    
    # Create protein viewer
    viewer_html = create_protein_structure_viewer(sequence)
    if viewer_html:
        components.html(viewer_html, height=450)
    
    # Display sequence analysis
    with st.expander("🔬 Sequence Analysis"):
        st.text_area("Protein Sequence", sequence, height=100)
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Length", f"{len(sequence)} AA")
        
        with col2:
            hydrophobic = sum(1 for aa in sequence if aa in 'AILVMFWY')
            st.metric("Hydrophobic", f"{hydrophobic/len(sequence)*100:.1f}%")
        
        with col3:
            polar = sum(1 for aa in sequence if aa in 'STYNQ')
            st.metric("Polar", f"{polar/len(sequence)*100:.1f}%")
        
        with col4:
            charged = sum(1 for aa in sequence if aa in 'DEKHR')
            st.metric("Charged", f"{charged/len(sequence)*100:.1f}%")