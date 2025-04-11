import numpy as np
import pandas as pd
import awkward as ak
import uproot
import glob
import os
from read_low_level import read_file
from sklearn.model_selection import train_test_split

def read_all_root_files(data_folder="data20250214/"):
    """Read all ROOT files and return combined data"""
    print(f"Reading ROOT files from {data_folder}...")
    
    # Get all .root files in the folder
    root_files = glob.glob(data_folder + "*.root")
    print(f"Found {len(root_files)} ROOT files.")
    
    # Lists to store data
    all_x_particles = []
    all_y = []
    
    # Process each file
    for file in root_files:
        try:
            # Check if file exists and has content
            if os.path.exists(file) and os.path.getsize(file) > 0:
                print(f"Processing {file}...")
                x_particles, x_event, y = read_file(
                    file, 
                    event_level_features=['eventWeight', 'MET'],  # Add MET here
                    labels=['DSID', 'truth_W_decay_mode', 'selection_category'],
                    max_num_particles=30
                )
                
                # Set particle type to -1 for zero-padded entries
                x_particles[:,4,:][(x_particles[:,:4,:]==0).all(axis=1)] = -1
                
                all_x_particles.append(x_particles)
                all_y.append(np.concatenate([y, x_event], axis=1))
            else:
                print(f"Skipping {file} - file is empty or doesn't exist")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Combine data
    if all_x_particles and all_y:
        x_particles = np.concatenate(all_x_particles, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Create DataFrame with all data
        columns = ['DSID', 'truth_W_decay_mode', 'selection_category', 'eventWeight', 'MET']
        all_df = pd.DataFrame(y, columns=columns)
        
        print(f"Total events: {len(all_df)}")
        
        return x_particles, all_df
    else:
        raise ValueError("No valid data found in any files")

def process_lvbb_data(x_particles, all_df):
    """Process data for LVBB channel with MET cut and proper signal/background separation"""
    print("\nProcessing LVBB data...")
    
    # LVBB selection based on selection_category
    lvbb_selected = all_df['selection_category'].isin([0, 8, 10])
    
    # Truth information
    lvbb_truth = (all_df['truth_W_decay_mode'] == 1)
    
    # Apply MET cut (30 GeV = 30000 MeV)
    met_selection = all_df['MET'] > 30000
    
    # Background identification (non-signal DSIDs)
    # Signal DSIDs are in range 510115-510124
    background_mask = ~all_df['DSID'].astype(str).str.startswith('510')
    
    # Define signal events: selected as lvbb AND truth is lvbb AND passes MET cut
    signal_mask = lvbb_selected & lvbb_truth & met_selection
    
    # Define background events: selected as lvbb BUT from background processes AND passes MET cut
    background_mask = lvbb_selected & background_mask & met_selection
    
    # Combined mask for events to use (both signal and background)
    combined_mask = signal_mask | background_mask
    
    # Extract data using the combined mask
    x_selected = x_particles[combined_mask].copy()
    
    # Create labels: 1 for signal, 0 for background
    y_selected = signal_mask[combined_mask].astype(int).values
    
    # Get weights
    weights_selected = np.abs(all_df.loc[combined_mask, 'eventWeight'].values)
    total_weight_before = weights_selected.sum()
    
    # Print statistics
    print(f"LVBB selected events: {np.sum(combined_mask)}")
    print(f"LVBB signal events: {np.sum(signal_mask & combined_mask)}")
    print(f"LVBB background events: {np.sum(background_mask & combined_mask)}")
    
    # Apply class weighting to compensate for imbalance
    signal_count = np.sum(y_selected == 1)
    background_count = np.sum(y_selected == 0)
    print(f"Signal to background ratio before reweighting: {signal_count / background_count:.4f}")

    # Calculate the total weight for each class before adjustment
    signal_total_weight = np.sum(weights_selected[y_selected == 1])
    background_total_weight = np.sum(weights_selected[y_selected == 0])
    print(f"Signal total weight: {signal_total_weight:.2f}, Background total weight: {background_total_weight:.2f}")

    # Compute adjustments to balance the classes
    signal_adjustment = 1.0
    background_adjustment = (signal_total_weight / background_total_weight)

    # Create the adjustment factor array
    adjustment_factors = np.ones_like(weights_selected)
    adjustment_factors[y_selected == 0] = background_adjustment

    # Apply the adjustments
    balanced_weights = weights_selected * adjustment_factors

    # Verify the balancing worked
    new_signal_weight = np.sum(balanced_weights[y_selected == 1])
    new_background_weight = np.sum(balanced_weights[y_selected == 0])
    print(f"After balancing - Signal weight: {new_signal_weight:.2f}, Background weight: {new_background_weight:.2f}")
    print(f"Effective signal to background weight ratio after reweighting: {new_signal_weight / new_background_weight:.4f}")

    # Use the balanced weights
    weights_selected = balanced_weights
    
    # Check that total weight is roughly preserved
    total_weight_after = np.sum(weights_selected)
    print(f"Total weight before: {total_weight_before:.2f}, after: {total_weight_after:.2f}")
    print(f"Effective signal to background weight ratio after reweighting: {np.sum(weights_selected[y_selected==1]) / np.sum(weights_selected[y_selected==0]):.4f}")
    
    
    # Normalize momentum
    for i in range(4):
        mask = x_selected[:, i, :] != 0
        x_selected[:, i, :][mask] = x_selected[:, i, :][mask] / 10000.0
    
    # Create directory for preprocessed data
    os.makedirs("preprocessed", exist_ok=True)
    
    # Save processed data
    np.savez(
        "preprocessed/lvbb_data.npz",
        x_particles=x_selected,
        y_labels=y_selected,
        weights=weights_selected
    )
    
    print(f"LVBB data saved - X shape: {x_selected.shape}, Y shape: {y_selected.shape}")
    
def process_qqbb_data(x_particles, all_df):
    """Process data for QQBB channel preserving original distribution but with class weights"""
    print("\nProcessing QQBB data with class weighting...")
    
    # QQBB selection
    qqbb_selected = all_df['selection_category'].isin([3, 9])
    qqbb_truth = (all_df['truth_W_decay_mode'] == 2)
    
    # Extract data - maintain original distribution, don't force equal counts
    x_selected = x_particles[qqbb_selected].copy()
    y_selected = qqbb_truth[qqbb_selected].astype(int).values
    weights_selected = np.abs(all_df.loc[qqbb_selected, 'eventWeight'].values)
    
    # Print statistics
    print(f"QQBB selected events: {np.sum(qqbb_selected)}")
    print(f"QQBB signal events: {np.sum(y_selected)}")
    print(f"QQBB background events: {len(y_selected) - np.sum(y_selected)}")
    
    # Calculate class weights to adjust for imbalance
    class_counts = np.bincount(y_selected)
    total = len(y_selected)
    class_weights = total / (class_counts * len(class_counts))
    print(f"Class distribution: {class_counts}")
    print(f"Class weight factors: {class_weights}")
    
    # Apply class weights to original event weights
    adjusted_weights = np.zeros_like(weights_selected)
    for i in range(len(y_selected)):
        adjusted_weights[i] = weights_selected[i] * class_weights[y_selected[i]]
    
    # Normalize momentum
    for i in range(4):
        mask = x_selected[:, i, :] != 0
        x_selected[:, i, :][mask] = x_selected[:, i, :][mask] / 10000.0
    
    # Create directory for preprocessed data
    os.makedirs("preprocessed", exist_ok=True)
    
    # Save processed data with both original and adjusted weights
    np.savez(
        "preprocessed/qqbb_data.npz",
        x_particles=x_selected,
        y_labels=y_selected,
        weights=weights_selected,  # Original event weights
        adjusted_weights=adjusted_weights  # Event weights × class weights
    )
    
    print(f"QQBB data saved - X shape: {x_selected.shape}, Y shape: {y_selected.shape}")

if __name__ == "__main__":
    # Read and combine all ROOT files
    x_particles, all_df = read_all_root_files()
    
    # Process data for lvbb channel with MET cut and DSID selection
    process_lvbb_data(x_particles, all_df)
    
    # Process data for qqbb channel with MET cut and DSID selection
    process_qqbb_data(x_particles, all_df)
    
    print("\nAll preprocessing complete!")