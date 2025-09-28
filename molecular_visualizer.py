"""
Molecular Visualization Tools
3D and 2D molecular structure visualization using py3Dmol and RDKit
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol
import io
import base64
from PIL import Image

def create_3d_molecule_viewer(smiles, width=400, height=400, style='stick'):
    """
    Create 3D molecular structure viewer using py3Dmol
    
    Args:
        smiles: SMILES string of the molecule
        width: Width of the viewer
        height: Height of the viewer
        style: Visualization style ('stick', 'sphere', 'cartoon', 'surface')
    
    Returns:
        HTML component for Streamlit
    """
    try:
        # Generate 3D coordinates
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        mol = Chem.AddHs(mol)
        # Generate 3D coordinates
        if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
            # If embedding fails, try without randomSeed
            AllChem.EmbedMolecule(mol)
        
        # Optimize molecule geometry using MMFF force field
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            # If MMFF fails, try UFF
            try:
                AllChem.UFFOptimizeMolecule(mol)
            except:
                # If all optimization fails, continue without optimization
                pass
        
        # Convert to SDF format for py3Dmol
        sdf_string = Chem.MolToMolBlock(mol)
        
        # Create py3Dmol viewer
        viewer_id = f"mol3d_{abs(hash(smiles)) % 10000}"
        
        viewer_html = f"""
        <div id="{viewer_id}" style="width:{width}px; height:{height}px; position:relative; border: 1px solid #ddd; border-radius: 5px;"></div>
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <script>
        (function() {{
            // Wait for 3Dmol to load
            function init3DMol() {{
                if (typeof $3Dmol === 'undefined') {{
                    setTimeout(init3DMol, 100);
                    return;
                }}
                
                var element = document.getElementById('{viewer_id}');
                if (!element) {{
                    setTimeout(init3DMol, 100);
                    return;
                }}
                
                var viewer = $3Dmol.createViewer(element, {{
                    backgroundColor: 'white',
                    defaultcolors: $3Dmol.rasmolElementColors
                }});
                
                var sdf = `{sdf_string}`;
                viewer.addModel(sdf, 'sdf');
                
                // Set style based on parameter
                var style = '{style}';
                if (style === 'stick') {{
                    viewer.setStyle({{}}, {{stick: {{radius: 0.15, colorscheme: "default"}}, sphere: {{scale: 0.25, colorscheme: "default"}}}});
                }} else if (style === 'sphere') {{
                    viewer.setStyle({{}}, {{sphere: {{scale: 0.3, colorscheme: "default"}}}});
                }} else if (style === 'cartoon') {{
                    viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
                }} else if (style === 'surface') {{
                    viewer.setStyle({{}}, {{stick: {{radius: 0.1}}, sphere: {{scale: 0.2}}}});
                    viewer.addSurface($3Dmol.VDW, {{opacity: 0.7, color: 'white'}});
                }}
                
                viewer.zoomTo();
                viewer.render();
                viewer.zoom(1.2, 1000);
            }}
            
            init3DMol();
        }})();
        </script>
        """
        
        return viewer_html
        
    except Exception as e:
        st.error(f"Error creating 3D visualization: {e}")
        return None

def create_2d_molecule_image(smiles, width=300, height=300, highlight_atoms=None):
    """
    Create 2D molecular structure image using RDKit
    
    Args:
        smiles: SMILES string of the molecule
        width: Image width
        height: Image height
        highlight_atoms: List of atom indices to highlight
    
    Returns:
        PIL Image object
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create drawer
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        
        # Set drawing options
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.addAtomIndices = False
        opts.bondLineWidth = 2
        
        # Highlight atoms if specified
        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)
        
        drawer.FinishDrawing()
        
        # Convert to PIL Image
        img_data = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(img_data))
        
        return img
        
    except Exception as e:
        st.error(f"Error creating 2D visualization: {e}")
        return None

def create_molecular_graph_network(smiles, layout='spring'):
    """
    Create network graph of molecular structure using NetworkX
    
    Args:
        smiles: SMILES string of the molecule
        layout: Graph layout ('spring', 'circular', 'kamada_kawai')
    
    Returns:
        Plotly figure object
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add atoms as nodes
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), 
                      element=atom.GetSymbol(),
                      formal_charge=atom.GetFormalCharge(),
                      hybridization=str(atom.GetHybridization()))
        
        # Add bonds as edges
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), 
                      bond.GetEndAtomIdx(),
                      bond_type=str(bond.GetBondType()))
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_elements = [mol.GetAtomWithIdx(i).GetSymbol() for i in G.nodes()]
        node_colors = [get_element_color(element) for element in node_elements]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_elements,
            textposition="middle center",
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
        
        # Create hover text
        node_text = []
        for node in G.nodes():
            atom = mol.GetAtomWithIdx(node)
            text = f"Atom {node}<br>"
            text += f"Element: {atom.GetSymbol()}<br>"
            text += f"Formal Charge: {atom.GetFormalCharge()}<br>"
            text += f"Hybridization: {atom.GetHybridization()}<br>"
            text += f"Degree: {atom.GetDegree()}"
            node_text.append(text)
        
        node_trace.hovertext = node_text
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=f'Molecular Graph: {smiles}', font=dict(size=16)),
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Molecular structure as a graph network",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating molecular graph: {e}")
        return None

def get_element_color(element):
    """Get color for chemical element"""
    colors = {
        'C': '#909090',  # Carbon - gray
        'N': '#3050F8',  # Nitrogen - blue
        'O': '#FF0D0D',  # Oxygen - red
        'H': '#FFFFFF',  # Hydrogen - white
        'S': '#FFFF30',  # Sulfur - yellow
        'P': '#FF8000',  # Phosphorus - orange
        'F': '#90E050',  # Fluorine - green
        'Cl': '#1DC51D', # Chlorine - green
        'Br': '#A62929', # Bromine - brown
        'I': '#940094',  # Iodine - purple
    }
    return colors.get(element, '#FF69B4')  # Default pink

def calculate_molecular_properties(smiles):
    """Calculate molecular properties and descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        properties = {
            'Molecular Weight': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'H-Bond Donors': Descriptors.NumHDonors(mol),
            'H-Bond Acceptors': Descriptors.NumHAcceptors(mol),
            'Rotatable Bonds': Descriptors.NumRotatableBonds(mol),
            'Aromatic Rings': Descriptors.NumAromaticRings(mol),
            'Heavy Atoms': Descriptors.HeavyAtomCount(mol),
            'Formal Charge': Chem.rdmolops.GetFormalCharge(mol),
            'Complexity': Descriptors.BertzCT(mol)
        }
        
        return properties
        
    except Exception as e:
        st.error(f"Error calculating properties: {e}")
        return {}

def create_property_radar_chart(properties):
    """Create radar chart for molecular properties"""
    if not properties:
        return None
    
    # Normalize properties for radar chart
    categories = list(properties.keys())
    values = list(properties.values())
    
    # Normalize values to 0-1 scale for better visualization
    normalized_values = []
    for i, (cat, val) in enumerate(zip(categories, values)):
        if cat == 'Molecular Weight':
            normalized_values.append(min(val / 500, 1.0))  # Normalize by 500
        elif cat == 'LogP':
            normalized_values.append((val + 5) / 10)  # Scale -5 to 5 range
        elif cat == 'TPSA':
            normalized_values.append(min(val / 200, 1.0))  # Normalize by 200
        elif cat in ['H-Bond Donors', 'H-Bond Acceptors']:
            normalized_values.append(min(val / 10, 1.0))  # Normalize by 10
        elif cat == 'Rotatable Bonds':
            normalized_values.append(min(val / 20, 1.0))  # Normalize by 20
        elif cat == 'Heavy Atoms':
            normalized_values.append(min(val / 50, 1.0))  # Normalize by 50
        elif cat == 'Complexity':
            normalized_values.append(min(val / 1000, 1.0))  # Normalize by 1000
        else:
            normalized_values.append(min(val / 10, 1.0))  # Default normalization
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name='Molecular Properties',
        line_color='rgb(32, 201, 151)',
        fillcolor='rgba(32, 201, 151, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Molecular Properties Radar Chart"
    )
    
    return fig

def create_fingerprint_visualization(smiles, fp_type='morgan'):
    """Create molecular fingerprint visualization"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate fingerprint
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        elif fp_type == 'topological':
            fp = Chem.RDKFingerprint(mol, fpSize=1024)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        
        # Convert to numpy array
        fp_array = np.array(fp)
        
        # Create heatmap
        fp_matrix = fp_array.reshape(32, 32)  # Reshape to 32x32 for visualization
        
        fig = px.imshow(
            fp_matrix,
            color_continuous_scale='viridis',
            title=f'{fp_type.title()} Fingerprint Heatmap',
            labels={'color': 'Bit Value'}
        )
        
        fig.update_layout(
            xaxis_title="Fingerprint Dimension (X)",
            yaxis_title="Fingerprint Dimension (Y)"
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating fingerprint visualization: {e}")
        return None

def similarity_analysis(smiles_list):
    """Analyze similarity between multiple molecules"""
    if len(smiles_list) < 2:
        return None
    
    try:
        # Generate fingerprints
        fps = []
        valid_smiles = []
        
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                fps.append(fp)
                valid_smiles.append(smiles)
        
        if len(fps) < 2:
            return None
        
        # Calculate similarity matrix
        from rdkit import DataStructs
        
        n = len(fps)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    similarity_matrix[i][j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        
        # Create heatmap
        fig = px.imshow(
            similarity_matrix,
            x=[f"Mol {i+1}" for i in range(n)],
            y=[f"Mol {i+1}" for i in range(n)],
            color_continuous_scale='RdYlBu_r',
            title='Molecular Similarity Matrix (Tanimoto Coefficient)',
            aspect='auto'
        )
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                fig.add_annotation(
                    x=j, y=i,
                    text=f"{similarity_matrix[i][j]:.3f}",
                    showarrow=False,
                    font=dict(color="white" if similarity_matrix[i][j] < 0.5 else "black")
                )
        
        return fig, similarity_matrix
        
    except Exception as e:
        st.error(f"Error in similarity analysis: {e}")
        return None, None