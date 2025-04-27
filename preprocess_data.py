import numpy as np
import pandas as pd
import awkward as ak
import uproot
import glob
import os
from read_low_level import read_file
from sklearn.model_selection import train_test_split

def load_and_save_original_data(data_folder="data20250214/", 
                              output_file="preprocessed/original_data.npz",
                              max_num_particles=30,
                              verbose=True):
    """
    Load all ROOT files from the specified folder and save the combined data to an npz file.
    
    Parameters:
    -----------
    data_folder : str
        Path to the folder containing the ROOT files
    output_file : str
        Path to save the npz file
    max_num_particles : int
        Maximum number of particles to consider per event
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    x_particles : numpy.ndarray
        Array of particle features with shape (n_events, 5, max_num_particles)
    all_df : pandas.DataFrame
        DataFrame containing event-level information
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all .root files in the folder
    root_files = glob.glob(data_folder + "*.root")
    
    if verbose:
        print(f"Found {len(root_files)} ROOT files.")
    
    # Lists to store data from all files
    all_x_particles = []
    all_y = []
    
    # Define the features to extract
    labels = ['DSID', 'truth_W_decay_mode', 'selection_category']
    event_level_features = ['eventWeight', 'MET']  # Added MET as it's useful for selection
    
    # Process each ROOT file
    for file in root_files:
        try:
            # Check if file exists and has content
            if os.path.exists(file) and os.path.getsize(file) > 0:
                if verbose:
                    print(f"Processing {file}...")
                
                # Read data from the file
                x_particles, x_event, y = read_file(
                    file, 
                    event_level_features=event_level_features, 
                    labels=labels, 
                    max_num_particles=max_num_particles
                )
                
                # Set particle type to -1 for zero-padded entries
                # This helps distinguish real particles from padding
                x_particles[:,4,:][(x_particles[:,:4,:]==0).all(axis=1)] = -1
                
                all_x_particles.append(x_particles)
                all_y.append(np.concatenate([y, x_event], axis=1))
                
                if verbose:
                    print(f"Successfully processed {file} - {x_particles.shape[0]} events")
            else:
                if verbose:
                    print(f"Skipping {file} - file is empty or doesn't exist")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    # Only create combined arrays if we have data
    if all_x_particles and all_y:
        # Combine data from all files
        x_particles = np.concatenate(all_x_particles, axis=0)
        y = np.concatenate(all_y, axis=0)
        
        # Create DataFrame from y data
        all_columns = labels + event_level_features
        all_df = pd.DataFrame(y, columns=all_columns)
        
        if verbose:
            print(f"Successfully processed {len(all_x_particles)} files")
            print(f"Total events: {len(all_df)}")
            print(f"x_particles shape: {x_particles.shape}")
            
            # Print signal/background statistics
            signal_mask = (all_df['DSID'] >= 500000) & (all_df['DSID'] < 600000)
            print(f"Signal events: {signal_mask.sum()} ({signal_mask.sum()/len(all_df)*100:.2f}%)")
            print(f"Background events: {(~signal_mask).sum()} ({(~signal_mask).sum()/len(all_df)*100:.2f}%)")
            
            # Print channel statistics
            lvbb_sel = all_df['selection_category'].isin([0, 8, 10])
            qqbb_sel = all_df['selection_category'].isin([3, 9])
            print(f"lvbb selected events: {lvbb_sel.sum()} ({lvbb_sel.sum()/len(all_df)*100:.2f}%)")
            print(f"qqbb selected events: {qqbb_sel.sum()} ({qqbb_sel.sum()/len(all_df)*100:.2f}%)")
            print(f"Other events: {(~(lvbb_sel | qqbb_sel)).sum()} ({(~(lvbb_sel | qqbb_sel)).sum()/len(all_df)*100:.2f}%)")
        
        # Save data to npz file
        np.savez(
            output_file,
            x_particles=x_particles,
            event_data=y,
            columns=all_columns
        )
        
        if verbose:
            print(f"Data saved to {output_file}")
        
        return x_particles, all_df
    else:
        print("No valid data found in any files")
        return None, None

def read_all_root_files(data_folder="data20250214/", use_cached=True, cache_file="preprocessed/original_data.npz"):
    """Read all ROOT files and return combined data, with option to use cached data"""
    print(f"Reading ROOT files from {data_folder}...")
    
    # Check if cached data exists and should be used
    if use_cached and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        data = np.load(cache_file)
        x_particles = data['x_particles']
        event_data = data['event_data']
        columns = data['columns']
        all_df = pd.DataFrame(event_data, columns=columns)
        print(f"Loaded cached data: {len(all_df)} events, x_particles shape: {x_particles.shape}")
        return x_particles, all_df
    
    # If no cached data or use_cached is False, load from ROOT files
    return load_and_save_original_data(data_folder, cache_file)

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
    print(f"QQBB selected with MET > 30 GeV: {np.sum(combined_mask)}")
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
    adjustment_factors[y_selected == 1] = signal_adjustment

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
    """Process data for QQBB channel with MET cut and class weighting"""
    print("\nProcessing QQBB data with MET cut and class weighting...")
    
    # QQBB selection
    qqbb_selected = all_df['selection_category'].isin([3, 9])
    qqbb_truth = (all_df['truth_W_decay_mode'] == 2)
    
    # Apply MET cut (30 GeV = 30000 MeV), same as lvbb
    met_selection = all_df['MET'] > 30000
    
    # Combine selections: must be selected as qqbb AND pass MET cut
    combined_mask = qqbb_selected & met_selection
    
    # Extract data using the combined mask
    x_selected = x_particles[combined_mask].copy()
    y_selected = qqbb_truth[combined_mask].astype(int).values
    weights_selected = np.abs(all_df.loc[combined_mask, 'eventWeight'].values)
    
    # Print statistics
    print(f"QQBB selected events: {np.sum(combined_mask)}")
    print(f"QQBB selected with MET > 30 GeV: {np.sum(combined_mask)}")
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

def analyze_dataset(all_df, x_particles):
    """Analyze dataset and print statistics"""
    print("\n===== Dataset Analysis =====")
    
    # Signal vs background
    signal_mask = (all_df['DSID'] >= 500000) & (all_df['DSID'] < 600000)
    print(f"Total events: {len(all_df)}")
    print(f"Signal events: {signal_mask.sum()} ({signal_mask.sum()/len(all_df)*100:.2f}%)")
    print(f"Background events: {(~signal_mask).sum()} ({(~signal_mask).sum()/len(all_df)*100:.2f}%)")
    
    # Selection categories
    lvbb_sel = all_df['selection_category'].isin([0, 8, 10])
    qqbb_sel = all_df['selection_category'].isin([3, 9])
    print(f"\nSelection categories:")
    print(f"lvbb selected events: {lvbb_sel.sum()} ({lvbb_sel.sum()/len(all_df)*100:.2f}%)")
    print(f"qqbb selected events: {qqbb_sel.sum()} ({qqbb_sel.sum()/len(all_df)*100:.2f}%)")
    print(f"Other events: {(~(lvbb_sel | qqbb_sel)).sum()} ({(~(lvbb_sel | qqbb_sel)).sum()/len(all_df)*100:.2f}%)")
    
    # Truth decay modes for signal events
    if signal_mask.sum() > 0:
        signal_df = all_df[signal_mask]
        lvbb_truth = (signal_df['truth_W_decay_mode'] == 1)
        qqbb_truth = (signal_df['truth_W_decay_mode'] == 2)
        other_truth = ~(lvbb_truth | qqbb_truth)
        
        print(f"\nTruth decay modes (signal events only):")
        print(f"lvbb truth: {lvbb_truth.sum()} ({lvbb_truth.sum()/len(signal_df)*100:.2f}%)")
        print(f"qqbb truth: {qqbb_truth.sum()} ({qqbb_truth.sum()/len(signal_df)*100:.2f}%)")
        print(f"Other/undefined: {other_truth.sum()} ({other_truth.sum()/len(signal_df)*100:.2f}%)")
        
        # Selection vs truth confusion matrix for signal events
        print(f"\nSelection vs Truth (signal events only):")
        print(f"Correct lvbb selection: {(lvbb_sel & lvbb_truth & signal_mask).sum()} events")
        print(f"Correct qqbb selection: {(qqbb_sel & qqbb_truth & signal_mask).sum()} events")
        print(f"lvbb selected but qqbb truth: {(lvbb_sel & qqbb_truth & signal_mask).sum()} events")
        print(f"qqbb selected but lvbb truth: {(qqbb_sel & lvbb_truth & signal_mask).sum()} events")
    
    # Mass point distribution for signal events
    if signal_mask.sum() > 0:
        print("\nMass point distribution (signal events):")
        mass_points = all_df.loc[signal_mask, 'DSID'].value_counts().sort_index()
        for mass_point, count in mass_points.items():
            print(f"DSID {mass_point}: {count} events")
    
    # Particle statistics
    print("\nParticle statistics:")
    # Count non-padding particles per event
    particle_counts = np.sum(x_particles[:, 4, :] != -1, axis=1)
    print(f"Average particles per event: {particle_counts.mean():.2f}")
    print(f"Min particles per event: {particle_counts.min()}")
    print(f"Max particles per event: {particle_counts.max()}")
    
    # Particle type distribution
    particle_types = x_particles[:, 4, :].flatten()
    particle_types = particle_types[particle_types != -1]  # Remove padding
    unique_types, type_counts = np.unique(particle_types, return_counts=True)
    print("\nParticle type distribution:")
    for type_val, count in zip(unique_types, type_counts):
        print(f"Type {int(type_val)}: {count} particles ({count/len(particle_types)*100:.2f}%)")

if __name__ == "__main__":
    # Read and combine all ROOT files, with option to use cached data
    x_particles, all_df = read_all_root_files(use_cached=True)
    
    # Analyze the dataset
    analyze_dataset(all_df, x_particles)
    
    # Process data for lvbb channel with MET cut and DSID selection
    process_lvbb_data(x_particles, all_df)
    
    # Process data for qqbb channel with MET cut and DSID selection
    process_qqbb_data(x_particles, all_df)
    
    print("\nAll preprocessing complete!")