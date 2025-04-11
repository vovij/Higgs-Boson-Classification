import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()
 
def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['ll_particle_px', 'll_particle_py', 'll_particle_pz', 'll_particle_e', 'll_particle_type'],
        event_level_features=['eventWeight', 'MET'],
        labels=['DSID', 'truth_W_decay_mode', 'selection_category'],
        new_inputs_labels=True):
    """Loads a single file from the JetClass dataset.
 
    **Arguments**
 
    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet.
        Jets with fewer particles will be zero-padded,
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded.
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded.
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded.
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl
 
    **Returns**
 
    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """
 
    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x
 
    if new_inputs_labels:
        table = uproot.open(filepath)['tree'].arrays()
        p4 = vector.zip({'px': table['ll_particle_px'],
                            'py': table['ll_particle_py'],
                            'pz': table['ll_particle_pz'],
                            'energy': table['ll_particle_e']})
        # pos = labels.index('dsid')
        # labels.remove('dsid')
        # labels.insert(pos, "DSID")
        # pos = labels.index('truth_decay_mode')
        # labels.remove('truth_decay_mode')
        # labels.insert(pos, "ll_truth_decay_mode")
    else:
        table = uproot.open(filepath)['ProcessedTree'].arrays()
        p4 = vector.zip({'px': table['px'],
                            'py': table['py'],
                            'pz': table['pz'],
                            'energy': table['e']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi
    table['part_px'] = p4.px
    table['part_py'] = p4.py
    table['part_pz'] = p4.pz
    table['part_energy'] = p4.energy
    table['part_mass'] = p4.mass
    if len(particle_features):
        print(particle_features)
        x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    else:
        x_particles = None
    if len(event_level_features):
        x_event = np.stack([ak.to_numpy(table[n]).astype('float32') for n in event_level_features], axis=1)
    else:
        x_event = None
    y = np.stack([(ak.to_numpy(table[n])=='lvbb')*1+(ak.to_numpy(table[n])=='qqbb')*2 if n=='truth_W_decay_mode' else ak.to_numpy(table[n]).astype('float') for n in labels], axis=1)
 
    return x_particles, x_event, y