import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import select
from uvsib.db.tables import *
from uvsib.db.utils import *

def fetch_oer_data(reaction, composition):
    """
    Fetch reaction data from the database.
    """
    rows = query_by_columns(DBSurfaceMLAdsorbate, {"reaction": reaction, "composition":composition})
    oer_data = []
    for row in rows:
#        slab_row = query_by_columns(DBSurface, {"id":row.surface_id})[0]
        oer_data.append({
                'composition': row.composition,
                'site_type': row.site_type,
                'overpotential': row.eta,
                'dG': row.dG,
                'reaction_path': row.reaction_path,
                'ads_coord': row.ads_coord,
                'repeat': row.repeat,
#                'miller_index': slab_row.slab['miller_index']
            })
    return oer_data

def prepare_volcano_data(oer_data):
    """
    Prepare data for volcano plot.
    Returns
    -------
    tuple
        (descriptors, overpotentials, compositions, sites)
    """
    descriptors = []
    overpotentials = []
    compositions = []
    sites = []
    
    for entry in oer_data:
        dG_array = entry['dG']
        # Index 2 is *OH, index 1 is *O
        descriptor =  dG_array[1] - dG_array[2]
        descriptors.append(descriptor)
        overpotentials.append(entry['overpotential'])
        compositions.append(entry['composition'])
        sites.append(entry['site_type'])
    return descriptors, overpotentials, compositions, sites

def plot_volcano(descriptors, overpotentials, compositions, sites,
                 output_file='oer_volcano.png',
                 title='OER Volcano Plot'):
    """
    Create and save OER volcano plot.
    
    Parameters
    ----------
    descriptors : list
        List of descriptor values
    overpotentials : list
        List of overpotential values (in eV)
    compositions : list, optional
        List of composition strings for labeling
    sites : list, optional
        List of site types for coloring
    output_file : str
        Output file path
    title : str
        Plot title
    """
    descriptors = np.array(descriptors)
    overpotentials = -1 * np.array(overpotentials)
    
    # Determine color based on site type
    colors = []
    site_colors = {'ontop': 'gray', 'bridge': 'gray', 'hollow': 'gray'}
    if sites:
        colors = [site_colors.get(site, 'gray') for site in sites]
    else:
        colors = 'blue'
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot data points
    scatter = ax.scatter(descriptors, overpotentials, c=colors, s=10, 
                        alpha=0.7, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('$\Delta G_{*O} - \Delta G_{*OH}$ (eV)', 
                 fontsize=12, fontweight='normal')
    ax.set_ylabel('-$\eta$ (V)', fontsize=12, fontweight='normal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Legend for site types
#    if sites:
#        from matplotlib.patches import Patch
#        legend_elements = [
#            Patch(facecolor='red', edgecolor='black', label='Ontop/Top'),
#            Patch(facecolor='blue', edgecolor='black', label='Bridge'),
#            Patch(facecolor='green', edgecolor='black', label='Hollow')
#        ]
#        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
#    else:
#        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, format='png')
    plt.close()


def print_data_summary(oer_data):
    """Print summary statistics of OER data."""
    descriptors, overpotentials, compositions, sites = prepare_volcano_data(
        oer_data)
    
    print("\n" + "="*70)
    print("OER DATA SUMMARY")
    print("="*70)
    print(f"Total entries: {len(oer_data)}")
    print(f"Valid entries for volcano plot: {len(descriptors)}")
    print(f"Descriptor range: {min(descriptors):.3f} to {max(descriptors):.3f} eV")
    print(f"Overpotential range: {min(overpotentials):.3f} to {max(overpotentials):.3f} V")
    
    # Find best catalyst
    best_idx = np.argmin(overpotentials)
    print(f"\nBest catalyst (lowest overpotential):")
    print(f"  Composition: {compositions[best_idx]}")
    print(f"  Site type: {sites[best_idx]}")
    print(f"  Descriptor: {descriptors[best_idx]:.3f} eV")
    print(f"  Overpotential: {overpotentials[best_idx]:.3f} V")
    
    # Statistics by site type
    if sites:
        print(f"\nStatistics by site type:")
        for site_type in set(sites):
            site_indices = [i for i, s in enumerate(sites) if s == site_type]
            site_overpotentials = [overpotentials[i] for i in site_indices]
            print(f"  {site_type}: {len(site_indices)} entries, "
                  f"avg overpotential = {np.mean(site_overpotentials):.3f} V")
    
    print("="*70 + "\n")


def main(reaction, composition):
    """Main function to generate OER volcano plot."""
    oer_data = fetch_oer_data(reaction, composition)
    # Print summary
    print_data_summary(oer_data)
    # Prepare data
    descriptors, overpotentials, compositions_list, sites = prepare_volcano_data(oer_data)
    # Generate plot
    plot_volcano(
        descriptors, overpotentials,
        compositions=compositions_list,
        sites=sites,
    )
    
if __name__ == "__main__":
    main(reaction="OER", composition="WO2")
