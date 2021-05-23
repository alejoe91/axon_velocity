import numpy as np
from scipy.stats import linregress, kurtosis
from scipy.signal import resample_poly
from scipy.stats import pearsonr, median_abs_deviation
from sklearn import linear_model
import matplotlib.pylab as plt
import networkx as nx
from matplotlib import gridspec
import matplotlib as mpl

from .plotting import plot_template, plot_velocity
from .tools import distance_numpy


class AxonTracking:
    def __init__(self, template, locations, fs, upsample=1, init_delay=0,
                 detect_threshold=0.1, kurt_threshold=0.3, peak_std_threshold=1, peak_std_distance=30,
                 detection_type="relative", remove_isolated=True, min_selected_points=30, min_path_length=100,
                 min_path_points=5, r2_threshold=None, verbose=False):
        assert len(template) == len(locations), f"Mismatch between template size {len(template)} and " \
                                                f"number of locations {len(locations)}"

        self.branches = None
        self.locations = locations
        self._detect_threshold = detect_threshold
        self._kurt_threshold = kurt_threshold
        self._peak_std_threhsold = peak_std_threshold
        self._peak_std_distance = peak_std_distance
        assert detection_type in ["relative", "absolute"]
        self._detection_type = detection_type
        self._remove_isolated = remove_isolated
        self._min_selected_points = min_selected_points
        self._min_path_length = min_path_length
        self._min_path_points = min_path_points
        self._init_delay = init_delay
        self._upsample = upsample
        self._r2_threshold = r2_threshold
        if isinstance(verbose, bool):
            if verbose:
                self._verbose = 1
            else:
                self._verbose= 0
        else:
            self._verbose = verbose

        self.selected_channels = np.arange(len(locations))
        self.compute_velocity = True

        if self._upsample > 1:
            self.template = resample_poly(template, up=upsample, down=1, axis=1)
            self.fs = fs * self._upsample
        else:
            self.template = template
            self.fs = fs

        self._init_frames = np.round(init_delay / 1e3 * self.fs).astype(int)

        self.peak_times = np.argmin(self.template, 1)
        self.amplitudes = np.abs(np.min(self.template, axis=1))
        self.log_amplitudes = np.log10(self.amplitudes + 1)  # +1 is to avoid [0-1] values
        self.init_channel = np.argmax(self.amplitudes)

        self._selected_channels_detect = None
        self._selected_channels_kurt = None
        self._selected_channels_peakstd = None
        self._selected_channels_init = None
        self._vscale = 1.5 * np.max(self.amplitudes)

    def select_channels(self, init_delay=None, detect_threshold=None,
                        kurt_threshold=None, remove_isolated=None, neighbor_distance=50):
        self.selected_channels = np.arange(len(self.locations))
        if init_delay is not None:
            self._init_frames = np.round(init_delay / 1e3 * self.fs).astype(int)
        if detect_threshold is not None:
            self._detect_threshold = detect_threshold
        if kurt_threshold is not None:
            self._kurt_threshold = kurt_threshold
        if remove_isolated is not None:
            assert isinstance(remove_isolated, bool), "'remove_isolated' must be a boolean"
            self._remove_isolated = remove_isolated

        if self._verbose > 0:
            print("Channel selection\n")

        # initially all channels are selected
        selected_channels = self.selected_channels

        if self._verbose > 0:
            print(f"Initial channels: {len(self.locations)}")

        # filter amplitudes
        selected_channels_detect = set(selected_channels)
        if self._detect_threshold is not None:
            max_amplitude = np.max(self.amplitudes)
            if self._detection_type == "relative":
                threshold = self._detect_threshold * max_amplitude
            else:
                threshold = self._detect_threshold
            channels_above_threshold = np.where(self.amplitudes > threshold)[0]
            selected_channels_detect = set(selected_channels[channels_above_threshold])
            if self._verbose > 0:
                print(f"{len(selected_channels_detect)} after detection filter")
        self._selected_channels_detect = selected_channels_detect

        # filter kurtosis
        selected_channels_kurt = set(selected_channels)
        if self._kurt_threshold is not None:
            kurt = kurtosis(self.template[selected_channels], 1)
            kurt_idxs = np.where(kurt > self._kurt_threshold)[0]
            selected_channels_kurt = set(selected_channels[kurt_idxs])
            if self._verbose > 0:
                print(f"{len(selected_channels_kurt)} after kurtosis filter")
        self._selected_channels_kurt = selected_channels_kurt

        # filter peak std
        selected_channels_peakstd = set(selected_channels)
        if self._peak_std_threhsold is not None:
            peak_time_stds = np.zeros(len(selected_channels))
            x, y = self.locations[selected_channels].T
            for i in np.arange(len(selected_channels)):
                neighboring_chs_idxs = np.where((np.abs(x - x[i]) < self._peak_std_distance)
                                                 & (np.abs(y - y[i]) < self._peak_std_distance))[0]
                if len(neighboring_chs_idxs) > 1:
                    peak_time_stds[i] = np.std(self.peak_times[neighboring_chs_idxs] / self.fs * 1000)
            peak_time_idxs = np.where(peak_time_stds < self._peak_std_threhsold)
            selected_channels_peakstd = set(selected_channels[peak_time_idxs])
            if self._verbose > 0:
                print(f"{len(selected_channels_peakstd)} after peak std filter")
        self._selected_channels_peakstd = selected_channels_peakstd

        # filter init delay
        selected_channels_init_delay = set(selected_channels)
        if self._init_delay is not None:
            init_frame = np.argmin(self.template[self.init_channel])
            init_idxs = np.where(self.peak_times >= init_frame + self._init_frames)[0]
            selected_channels_init_delay = selected_channels[init_idxs]
            selected_channels_init_delay = set(selected_channels_init_delay)
            if self._verbose > 0:
                print(f"{len(selected_channels_init_delay)} after init_delay filter")

        self._selected_channels_init = selected_channels_init_delay
        self.selected_channels = selected_channels_detect.intersection(selected_channels_kurt,
                                                                       selected_channels_peakstd,
                                                                       selected_channels_init_delay)

        if self._remove_isolated:
            self.selected_channels = np.array(list(self.selected_channels))
            num_channels = len(self.selected_channels)
            if self._verbose > 0:
                print(f"Removing isolated channels")
            selected_channels_non_isolated = []
            for ch in self.selected_channels:
                distances = np.array([np.linalg.norm(self.locations[ch] - loc)
                                      for loc in self.locations[self.selected_channels]])
                # exclude same channel
                neighboring_chs_idxs = np.where((distances < neighbor_distance) & (distances > 0))[0]
                if len(neighboring_chs_idxs) > 0:
                    selected_channels_non_isolated.append(ch)
            self.selected_channels = np.array(selected_channels_non_isolated)
            if self._verbose > 0:
                print(f"Removed {num_channels - len(selected_channels_non_isolated)} isolated channels")

        self.selected_channels = list(self.selected_channels)
        if self.init_channel not in self.selected_channels:
            self.selected_channels.append(self.init_channel)

        self.selected_channels = np.array(self.selected_channels)

        if len(self.selected_channels) < self._min_selected_points:
            if self._verbose > 0:
                print("Discarded for minimum points after init delay")
            self.compute_velocity = False

    def compute_peak_std(self, neighbor_distance=30):
        peak_time_stds = np.zeros(len(self.locations))
        for i in np.arange(len(self.locations)):
            distances = np.array([np.linalg.norm(self.locations[i] - loc)
                                  for loc in self.locations])
            # exclude same channel
            neighboring_chs_idxs = np.where((distances < neighbor_distance) & (distances > 0))[0]
            if len(neighboring_chs_idxs) > 1:
                peak_time_stds[i] = np.std(self.peak_times[neighboring_chs_idxs])

        return peak_time_stds

    def plot_channel_selection(self):
        filters = {'detect': self._detect_threshold,
                   'kurtosis': self._kurt_threshold,
                   'peak_std': self._peak_std_threhsold,
                   'init_delay': self._init_delay}
        applied_filters = [k for (k, v) in filters.items() if v is not None]
        n_filters = len(applied_filters)

        fig, axes = plt.subplots(nrows=1, ncols=n_filters + 1)
        vscale = 1.5 * np.max(self.amplitudes)
        ax_id = 0

        for filt, thresh in filters.items():
            if thresh is not None:
                if filt == 'detect':
                    ax_detect = axes[ax_id]
                    ax_detect.set_title(f"Detection threshold: {self._detect_threshold}")
                    plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=vscale, ax=ax_detect)
                    plot_template(self.template, self.locations, colors='k', channels=self._selected_channels_detect,
                                  vscale=self._vscale, ax=ax_detect)
                elif filt == 'kurtosis':
                    ax_kurt = axes[ax_id]
                    ax_kurt.set_title(f"Kurtosis threshold: {self._kurt_threshold}")
                    plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=vscale, ax=ax_kurt)
                    plot_template(self.template, self.locations, colors='k', channels=self._selected_channels_kurt,
                                  vscale=self._vscale, ax=ax_kurt)
                elif filt == 'peak_std':
                    ax_peak = axes[ax_id]
                    ax_peak.set_title(f"Peak std threshold {self._peak_std_threhsold}")
                    plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=vscale, ax=ax_peak)
                    plot_template(self.template, self.locations, colors='k', channels=self._selected_channels_peakstd,
                                  vscale=self._vscale, ax=ax_peak)
                elif filt == 'init_delay':
                    ax_init = axes[ax_id]
                    ax_init.set_title(f"Init delay threshold: {self._init_delay}")
                    plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=vscale, ax=ax_init)
                    plot_template(self.template, self.locations, colors='k', channels=self._selected_channels_init,
                                  vscale=self._vscale, ax=ax_init)
                    plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                                  vscale=self._vscale, ax=ax_init)
                ax_id += 1

            ax_all = axes[ax_id]
            ax_all.set_title(f"All thresholds")
            plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=vscale, ax=ax_all)
            plot_template(self.template, self.locations, colors='k', channels=self.selected_channels,
                          vscale=self._vscale, ax=ax_all)
            plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                          vscale=self._vscale, ax=ax_all)

            fig.subplots_adjust(top=0.8)
            fig.suptitle("Channel selection", fontsize=20, y=0.95)

        return fig

    def track_axons(self):
        raise NotImplementedError

    def plot_velocities(self, fig=None):
        raise NotImplementedError


class SimpleAxonTracking(AxonTracking):
    """
    Class to perform simple axonal tracking.
    """
    default_params = dict(upsample=1, init_delay=0, detect_threshold=0.1,
                          kurt_threshold=0.3, peak_std_threshold=None, peak_std_distance=30,
                          detection_type="relative", remove_isolated=True, min_selected_points=30, min_path_length=100,
                          min_path_points=5, r2_threshold=None)

    def __init__(self, template, locations, fs, upsample=1, init_delay=0, detect_threshold=0.1,
                 kurt_threshold=0.3, peak_std_threshold=None, peak_std_distance=30, remove_isolated=True,
                 detection_type="relative", min_selected_points=30, min_path_length=100, min_path_points=5,
                 r2_threshold=None, verbose=False):
        AxonTracking.__init__(self, template, locations, fs, upsample=upsample, init_delay=init_delay,
                              detect_threshold=detect_threshold, detection_type=detection_type,
                              kurt_threshold=kurt_threshold, peak_std_threshold=peak_std_threshold,
                              peak_std_distance=peak_std_distance, remove_isolated=remove_isolated,
                              min_selected_points=min_selected_points,
                              min_path_length=min_path_length, min_path_points=min_path_points,
                              r2_threshold=r2_threshold, verbose=verbose)
        self.distances = None

    def track_axons(self):
        self.select_channels()

        if self.compute_velocity:
            distances = []
            peaks = []
            for ch in self.selected_channels:
                peaks.append(self.peak_times[ch] - self.peak_times[self.init_channel]) / self.fs * 1000
                distances.append(np.linalg.norm(self.locations[ch] - self.locations[self.init_channel]))
            velocity, offset, r_value, p_value, std_err = linregress(peaks, distances)
            r2 = r_value ** 2

            branch_dict = {}
            branch_dict['channels'] = self.selected_channels
            branch_dict['velocity'] = velocity
            branch_dict['offset'] = offset
            branch_dict['r2'] = r2
            branch_dict['pval'] = p_value
            branch_dict['distances'] = np.array(distances)
            branch_dict['peak_times'] = np.array(peaks)
            branch_dict['raw_path_idx'] = 0
            self.branches = [branch_dict]
        else:
            raise Exception("Not enough channels selected to compute velocity")

    def plot_velocities(self, fig=None, cmap='rainbow'):
        if fig is None:
            fig = plt.figure()
        ax_vel = fig.add_subplot(111)

        cm = plt.get_cmap(cmap)
        color = cm(0)
        branch = self.branches[0]
        plot_velocity(np.array(branch['peak_times']), np.array(branch['distances']),
                      branch['velocity'], branch['offset'], color=color, r2=branch['r2'],
                      ax=ax_vel)

        return fig


class GraphAxonTracking(AxonTracking):
    """
    Class to perform graph-based axonal tracking.
    """
    default_params = dict(upsample=1, init_delay=0,
                          detect_threshold=0.01, kurt_threshold=0.3, peak_std_threshold=None, peak_std_distance=30,
                          remove_isolated=True, detection_type="relative",
                          min_selected_points=30, min_path_length=100, min_path_points=5,
                          min_points_after_branching=3, r2_threshold=0.9, max_distance_for_edge=300,
                          max_distance_to_init=200, mad_threshold=8, n_neighbors=3, init_amp_peak_ratio=0.2,
                          edge_dist_amp_ratio=0.3, distance_exp=2, max_peak_latency_for_splitting=1,
                          theilsen_maxiter=2000, neighbor_selection="amp", neighbor_radius=100, split_paths=True)

    def __init__(self, template, locations, fs, upsample=1, init_delay=0,
                 detect_threshold=0.1, kurt_threshold=0.3, peak_std_threshold=None, remove_isolated=True,
                 peak_std_distance=30, detection_type="relative",
                 min_selected_points=30, min_path_length=100, min_path_points=5, r2_threshold=0.9,
                 max_distance_for_edge=300, max_distance_to_init=200, n_neighbors=3, init_amp_peak_ratio=0.2,
                 min_points_after_branching=3, max_peak_latency_for_splitting=1, theilsen_maxiter=2000,
                 edge_dist_amp_ratio=0.3, mad_threshold=5, distance_exp=2, neighbor_selection="amp",
                 neighbor_radius=100, split_paths=True, verbose=False):
        AxonTracking.__init__(self, template, locations, fs, upsample=upsample, init_delay=init_delay,
                              detect_threshold=detect_threshold, kurt_threshold=kurt_threshold,
                              peak_std_threshold=peak_std_threshold, peak_std_distance=peak_std_distance,
                              detection_type=detection_type, remove_isolated=remove_isolated,
                              min_selected_points=min_selected_points, min_path_length=min_path_length,
                              min_path_points=min_path_points,
                              r2_threshold=r2_threshold, verbose=verbose)
        self._n_neighbors = n_neighbors
        self._max_distance_for_edge = max_distance_for_edge
        self._max_distance_to_init = max_distance_to_init
        self._init_amp_peak_ratio = init_amp_peak_ratio
        self._mad_threshold = mad_threshold
        self._distance_exp = distance_exp
        self._neighbor_radius = neighbor_radius
        assert neighbor_selection in ["dist", "amp"]
        self._edge_dist_amp_ratio = edge_dist_amp_ratio
        self._neighbor_selection = neighbor_selection
        self._max_peak_latency_for_splitting = max_peak_latency_for_splitting
        self._min_points_after_branching = min_points_after_branching
        self._theilsen_maxiter = theilsen_maxiter
        self._split_paths = split_paths

        self._node_heuristic = None
        self._selected_channels_sorted = None
        self._removed_neighbors = None
        self._paths_raw = None
        self._paths_clean = None
        self._paths_clean_idxs = None
        self._path_avg_heuristic = None
        self._nodes_searched = []
        self._full_paths = []

        self.graph = None

    def track_axons(self):
        # select channels
        self.select_channels()
        self.build_graph()
        self.find_paths()
        self.clean_paths()

    def build_graph(self, init_amp_peak_ratio=None, n_neighbors=None, max_distance_for_edge=None,
                    neighbor_selection=None):
        if self.compute_velocity:
            if init_amp_peak_ratio is not None:
                self._init_amp_peak_ratio = init_amp_peak_ratio
            if n_neighbors is not None:
                self._n_neighbors = n_neighbors
            if max_distance_for_edge is not None:
                self._max_distance_for_edge = max_distance_for_edge
            if neighbor_selection is not None:
                assert neighbor_selection in ["dist", "amp"]
                self._neighbor_selection = neighbor_selection

            # use both peak time and amplitude to sort channels
            peak_times_init = self.peak_times[self.selected_channels] - self.peak_times[self.init_channel]
            amp_init = np.log10(self.amplitudes[self.selected_channels])
            peak_times_init_norm = (peak_times_init - np.min(peak_times_init)) / np.ptp(peak_times_init)
            amp_init_norm = (amp_init - np.min(amp_init)) / np.ptp(amp_init)

            heuristic_init = self._init_amp_peak_ratio * amp_init_norm + \
                             (1 - self._init_amp_peak_ratio) * peak_times_init_norm

            ind_frame = np.argsort(heuristic_init)[::-1]
            self._node_heuristic = heuristic_init[ind_frame]
            self._selected_channels_sorted = self.selected_channels[ind_frame]

            # Create directed graph
            graph = nx.DiGraph()

            # Add nodes
            for chan, heur in zip(self._selected_channels_sorted, self._node_heuristic):
                graph.add_node(chan, location=self.locations[chan], amplitude=self.amplitudes[chan],
                               peak_time=self.peak_times[chan], h_init=heur)

            if self._verbose > 0:
                print(f'Added {len(graph.nodes)} nodes')

            # Add edges
            for node in list(graph.nodes()):
                if node != self.init_channel:
                    neighbors_channel_idxs = self._get_earlier_channels_within_distance(node)

                    # Add data to edges
                    for node_n in neighbors_channel_idxs:
                        if node == node_n:
                            raise Exception()
                        peak_diff = (self.peak_times[node] - self.peak_times[node_n])
                        dist = np.linalg.norm(self.locations[node] - self.locations[node_n])
                        amp = 0.5 * (self.log_amplitudes[node] + self.log_amplitudes[node_n])

                        if node_n in list(graph.nodes):
                            if (node_n, node) not in graph.edges:
                                graph.add_edge(node, node_n, peak_diff=peak_diff, dist=dist, amp=amp)

            edges_to_init = len([e for e in graph.edges() if e[1] == self.init_channel])

            if self._verbose > 0:
                print(f'Added {len(graph.edges)} edges')
                print(f'{edges_to_init} connected to init channel')

            # Compute and add edge heuristic
            all_peaks = []
            all_dists = []
            all_amps = []
            for edge in list(graph.edges(data=True)):
                n1, n2, d = edge
                all_peaks.append(d['peak_diff'])
                all_dists.append(d['dist'])
                all_amps.append(d['amp'])

            max_amp = np.max(all_amps)
            ptp_amp = np.ptp(all_amps)

            self._max_dist_node = np.max(all_dists)
            self._min_dist_node = np.min(all_dists)
            for n1, n2, d in graph.edges.data():
                amp = (max_amp - d['amp']) / ptp_amp
                heur = amp

                # the initial channel should be penalised
                if n2 == self.init_channel:
                    d['heur'] = 2
                else:
                    d['heur'] = heur

            self.graph = graph
        else:
            raise Exception("Not enough channels selected to compute velocity")

    def find_paths(self, min_path_length=None, min_path_points=None, neighbor_radius=None, distance_exp=None,
                   include_source=True, merging=True, pruning=True):
        from copy import deepcopy
        if self.compute_velocity:
            if min_path_length is not None:
                self._min_path_length = min_path_length
            if min_path_points is not None:
                self._min_path_points = min_path_points
            if neighbor_radius is not None:
                self._neighbor_radius = neighbor_radius
            if distance_exp is not None:
                self._distance_exp = distance_exp
            # Find paths
            all_nodes = self.graph.nodes
            paths = []
            target_node = self.init_channel
            self._removed_neighbors = []
            self._path_neighbors = []
            self._branching_points = []

            i = 0
            init_path_ = []
            self._snap_channels = []

            for source_node in list(all_nodes):
                # check if source node is a local maximum in the h_node space
                adj_chans = self._get_adjacent_channels(source_node, distance_thresh=self._neighbor_radius)
                h_init_neigh = np.array([self.graph.nodes[adj]['h_init'] for adj in adj_chans])
                # if not, continue
                if np.any(h_init_neigh > self.graph.nodes[source_node]['h_init']):
                    continue
                init_path_.append(source_node)

                # if node is not in black list
                if source_node in self.graph.nodes and source_node not in self._removed_neighbors and \
                        source_node != self.init_channel:
                    # check if there is a path to target
                    if nx.has_path(self.graph, source_node, target_node):
                        path = nx.astar_path(self.graph, source_node, target_node, heuristic=self._graph_dist,
                                             weight='heur')
                        if not include_source:
                                path.remove(target_node)
                        i += 1

                        # remove nodes which are already part of other paths (excluding source nodes)
                        full_path = path
                        self._full_paths.append(deepcopy(full_path))

                        # remove channels when a branch is found
                        snap_channel = None
                        for adj_p_i, adj_nodes in enumerate(self._path_neighbors):
                            for p_i, p in enumerate(path):
                                if p in adj_nodes:
                                    for p1 in path[p_i:]:
                                        path.remove(p1)
                                    # add closes path in parent path
                                    dist_to_parent = np.array([np.linalg.norm(self.locations[p] - self.locations[op])
                                                               for op in paths[adj_p_i]])
                                    snap_channel = paths[adj_p_i][np.argmin(dist_to_parent)]
                                    path.append(snap_channel)
                                    break

                        if len(path) > 0:
                            # if conditions are satisfied: accept path
                            path = self.get_full_path(path, 5)
                            if self._is_path_valid(path):
                                paths.append(path)

                                if snap_channel is not None:
                                    self._snap_channels.append(snap_channel)
                                    if snap_channel not in self._branching_points:
                                        self._branching_points.append(snap_channel)

                                all_neighbors = self.get_full_path(path, min_dist=self._neighbor_radius)
                                # print(f"Len neigh channels {len(self._removed_neighbors)}")
                                self._path_neighbors.append(all_neighbors)
                                self._removed_neighbors.extend(all_neighbors)
                            else:
                                if self._verbose > 1:
                                    if self.compute_path_length(path) < self._min_path_length:
                                        print(f"Path starting at channel {path[0]} removed for minimum path length: "
                                              f"{self.compute_path_length(path)}um")
                                    elif len(path) < self._min_path_points:
                                        print(f"Path starting at channel {path[0]} removed for minimum path points: "
                                              f"{len(path)} points")
                        else:
                            if self._verbose > 1:
                                print(f"No points left for path starting at channel {source_node}")
                    else:
                        if self._verbose > 1:
                            print(f"No path between {source_node} and init channel {self.init_channel}")

            self._removed_neighbors = np.unique(self._removed_neighbors)
            self._nodes_searched = init_path_

            self._paths_raw_original = deepcopy(paths)
            self._branching_points_original = deepcopy(self._branching_points)

            # prune and merge based on branching point
            if self._verbose > 0:
                print("Pruning")
            merge_paths = []
            branch_points_to_remove = []
            paths_sharing_single_bp = {}
            for i_b, bp in enumerate(self._branching_points):
                # find paths with branching point
                paths_with_bp_idxs = [p_i for p_i, path in enumerate(paths) if bp in path]
                if len(paths_with_bp_idxs) == 2:
                    paths_sharing_single_bp[bp] = paths_with_bp_idxs

                for path_with_bp_i in paths_with_bp_idxs:
                    path_with_bp = paths[path_with_bp_i]
                    path_after_bp = path_with_bp[:path_with_bp.index(bp)]
                    has_other_bp = False
                    for bp_other in self._branching_points:
                        if bp_other != bp:
                            if bp_other in path_after_bp:
                                has_other_bp = True
                    if self._min_points_after_branching > len(path_after_bp) > 0:
                        if not has_other_bp:
                            if self._verbose > 0:
                                print(f"Removing {len(path_after_bp)} channels from {path_with_bp_i}")
                            if pruning:
                                for p in path_after_bp:
                                    path_with_bp.remove(p)
                        else:
                            if self._verbose > 0:
                                print(f"Not removing channels from {path_with_bp_i} because it contains "
                                      f"branching points")

            for bp, paths_with_bp_idxs in paths_sharing_single_bp.items():
                path_with_bp_1 = paths[paths_with_bp_idxs[0]]
                path_with_bp_2 = paths[paths_with_bp_idxs[1]]

                should_merge = False
                if (path_with_bp_1[0] == path_with_bp_2[-1] == bp) or (path_with_bp_1[-1] == path_with_bp_2[0] == bp):
                    should_merge = True
                    branch_points_to_remove.append(bp)

                if should_merge:
                    already_in_merged_paths = False
                    for mg in merge_paths:
                        if paths_with_bp_idxs[0] in mg or paths_with_bp_idxs[1] in mg:
                            already_in_merged_paths = True

                    if not already_in_merged_paths:
                        if self._verbose > 0:
                            print(f"Adding path to merge: {paths_with_bp_idxs}")
                        if path_with_bp_1[0] == path_with_bp_2[-1]:
                            merge_paths.append(paths_with_bp_idxs)
                            branch_points_to_remove.append(bp)
                        if path_with_bp_1[-1] == path_with_bp_2[0]:
                            merge_paths.append(paths_with_bp_idxs)
                            branch_points_to_remove.append(bp)
                    else:
                        for mg in merge_paths:
                            if paths_with_bp_idxs[0] in mg:
                                mg.append(paths_with_bp_idxs[1])
                                break
                            elif paths_with_bp_idxs[0] in mg:
                                mg.append(paths_with_bp_idxs[0])
                                break
                        branch_points_to_remove.append(bp)
                        print(f"Extending already existing paths to merge: {mg}")

            self._paths_pruned = deepcopy(paths)
            paths_merged = deepcopy(paths)

            if merging:
                if len(merge_paths) > 0:
                    for merge_idxs in merge_paths:
                        if self._verbose > 0:
                            print(f"Merging {merge_idxs}")
                        new_path = []
                        median_latencies = []
                        for mg in merge_idxs:
                            median_latencies.append(np.median(self.peak_times[paths[mg]]))

                        for idx in np.argsort(median_latencies)[::-1]:
                            for p in paths[merge_idxs[idx]]:
                                if p not in new_path:
                                    new_path.extend(paths[merge_idxs[idx]])

                        for mg in merge_idxs:
                            paths_merged.remove(paths[mg])

                        paths_merged.append(new_path)

                for bp in np.unique(branch_points_to_remove):
                    self._branching_points.remove(bp)
            paths = paths_merged

            if len(paths) == 0:
                raise Exception("No branches found")

            if self._verbose > 0:
                print(f'Searched paths from {len(init_path_)} nodes')
                print(f'Number of raw branches {len(paths)}')
            self._paths_raw = paths
        else:
            raise Exception("Not enough channels selected to compute velocity")

    def clean_paths(self, r2_threshold=None, mad_threshold=None, remove_outliers=True):
        # Sort based on length
        if self.compute_velocity:
            if r2_threshold is not None:
                self._r2_threshold = r2_threshold
            if mad_threshold is not None:
                self._mad_threshold = mad_threshold
            path_lengths = []
            for path in self._paths_raw:
                path_lengths.append(self.compute_path_length(path))

            # sorted_idxs = np.argsort(path_lengths)[::-1]
            # sorted_paths = [self._paths_raw[sorted_idx] for sorted_idx in sorted_idxs]

            # Create branch dictionary
            branches = []
            self._paths_clean = []
            for p_i, path in enumerate(self._paths_raw):
                # skip branch point
                path_rev = path[::-1][1:]
                peaks = (self.peak_times[path_rev] - self.peak_times[path_rev[0]]) / self.fs * 1000

                dists = []
                cum_dist = 0
                for i, p in enumerate(path_rev):
                    if i == 0:
                        cum_dist += 0
                    else:
                        cum_dist += np.linalg.norm(self.locations[p] - self.locations[path_rev[i - 1]])
                    dists.append(cum_dist)
                peaks = np.array(peaks)
                dists = np.array(dists)

                # Use TheilSenRegressor for more robust model amd to remove outliers
                initial_len = len(path_rev)
                path_rev, velocity, offset, r2, p_value, std_err, dists, peaks \
                    = self.robust_velocity_estimator(path_rev, peaks, dists, remove_outliers)
                if path_rev is not None:
                    final_len = len(path_rev)

                    if final_len < initial_len and self._verbose:
                        print(f"Removed {initial_len - final_len} outliers from branch {p_i} - final r2: {r2}")

                    add_full_path = True
                    branches_to_add = []

                    peak_diff = np.diff(peaks)
                    if np.max(peak_diff) > self._max_peak_latency_for_splitting and self._split_paths:
                        # split
                        split_points = np.where(peak_diff > 1)[0]
                        if len(split_points) > 0:
                            if self._verbose > 0:
                                print(f"Attempting to split branch {p_i} in {len(split_points) + 1}")

                            subpath_idxs = []
                            for i_sp, sp in enumerate(split_points):
                                if i_sp == 0:  # first path
                                    subpath_idxs.append(np.arange(0, sp))
                                else:  # other paths if more than one selected
                                    subpath_idxs.append(np.arange(split_points[i_sp - 1], sp))
                            subpath_idxs.append(np.arange(split_points[-1], len(path_rev)))

                            full_path_r2 = r2
                            sub_branches = []
                            for i_s, idxs in enumerate(subpath_idxs):
                                subpeaks = peaks[idxs]
                                subdists = dists[idxs]
                                subpath = list(np.array(path_rev)[idxs])

                                if len(subpath) > 2:
                                    subpath, velocity, offset, r2, p_value, std_err, subdists, subpeaks \
                                        = self.robust_velocity_estimator(subpath, subpeaks, subdists, remove_outliers)

                                    if subpath is not None:
                                        if i_s == 0:
                                            # re append branch point
                                            subpath = np.array([path[-1]] + list(subpath))

                                        if self._is_path_valid(subpath):
                                            branch_dict = {}
                                            branch_dict['channels'] = subpath.astype(int)
                                            branch_dict['velocity'] = velocity
                                            branch_dict['offset'] = offset
                                            branch_dict['r2'] = r2
                                            branch_dict['pval'] = p_value
                                            branch_dict['distances'] = np.array(subdists)
                                            branch_dict['peak_times'] = np.array(subpeaks)
                                            branch_dict['raw_path_idx'] = p_i #sorted_idxs[p_i]
                                            sub_branches.append(branch_dict)
                                            if self._verbose > 0:
                                                print(f"Added split sub path {i_s} from parent path: {p_i}")

                            if len(sub_branches) > 0:
                                r2s = [br["r2"] for br in sub_branches]
                                if np.mean(r2s) > full_path_r2:
                                    # In this case do not add the ful path, but only the sub-branches
                                    add_full_path = False
                                    for branch_dict in sub_branches:
                                        branches_to_add.append(branch_dict)

                    if add_full_path:
                        # re-append branch point
                        path_rev = np.array([path[-1]] + list(path_rev))
                        if self._is_path_valid(path_rev):
                            branch_dict = {}
                            branch_dict['channels'] = path_rev.astype(int)
                            branch_dict['velocity'] = velocity
                            branch_dict['offset'] = offset
                            branch_dict['r2'] = r2
                            branch_dict['pval'] = p_value
                            branch_dict['distances'] = np.array(dists)
                            branch_dict['peak_times'] = np.array(peaks)
                            branch_dict['raw_path_idx'] = p_i #sorted_idxs[p_i]
                            branches_to_add.append(branch_dict)

                    for branch in branches_to_add:
                        add_branch = False
                        if self._r2_threshold is None:
                            add_branch = True
                        elif branch['r2'] > self._r2_threshold:
                            add_branch = True
                        else:
                            if self._verbose > 0:
                                print(f"Branch {p_i} removed for low R2: {r2}")
                        if add_branch:
                            # for path_to_add in paths_to_add:
                            self._paths_clean.append(branch['channels'])
                            branches.append(branch)
                else:
                    if self._verbose > 0:
                        print(f"No points remaining for paht {p_i}")

            if len(branches) == 0:
                raise Exception("No branches left after cleaning")

            if self._verbose > 0:
                print(f'Number of clean branches {len(self._paths_clean)}')

            self.branches = branches
        else:
            raise Exception("Not enough channels selected to compute velocity")

    def compute_path_length(self, path):
        length = 0
        for p, node in enumerate(path[:-1]):
            node1 = self.graph.nodes[node]
            node2 = self.graph.nodes[path[p + 1]]
            loc1 = node1['location']
            loc2 = node2['location']
            distance = np.linalg.norm(loc1 - loc2)
            length += distance

        return length

    def robust_velocity_estimator(self, path, peaks, dists, remove_outliers):
        velocity, offset, r_value, p_value, std_err = linregress(peaks, dists)
        r2 = r_value ** 2
        path_clean = path
        if remove_outliers and self._mad_threshold:
            theil = linear_model.TheilSenRegressor(max_iter=self._theilsen_maxiter)
            theil.fit(peaks.reshape(-1, 1), dists)

            dists_pred = theil.predict(peaks.reshape(-1, 1))
            errors = np.abs(dists - dists_pred)

            # remove points +- 5 MAD
            mad = median_abs_deviation(errors)
            inlier_mask = np.where(np.abs(errors - np.median(errors)) < self._mad_threshold * mad)[0]

            path_clean = np.array(path_clean)[inlier_mask]

            # remove isolated channels
            for p in path_clean:
                dists_path = np.array([np.linalg.norm(self.locations[p] - self.locations[pi]) for pi in path_clean
                                       if pi != p])
                if np.all(dists_path > self._max_distance_for_edge):
                    inlier_mask = np.delete(inlier_mask, np.where(path_clean == p))
                    path_clean = np.delete(path_clean, np.where(path_clean == p))

            if len(inlier_mask) > 2:
                dists = dists[inlier_mask]
                peaks = peaks[inlier_mask]
                velocity, offset, r_value, p_value, std_err = linregress(peaks, dists)
                r2 = r_value ** 2
            else:
                path_clean, velocity, offset, r2, p_value, std_err, dists, peaks \
                    = None, None, None, None, None, None, None, None

        return path_clean, velocity, offset, r2, p_value, std_err, dists, peaks

    def plot_graph(self, fig=None, cmap_nodes='viridis', cmap_edges='rainbow', node_search_labels=False):
        if fig is None:
            fig = plt.figure()
        gs = gridspec.GridSpec(20, 20)
        ax_nodes = fig.add_subplot(gs[:, :7])
        ax_nodes_cb = fig.add_subplot(gs[:, 8:9])
        ax_edges = fig.add_subplot(gs[:, 11:18])
        ax_edges_cb = fig.add_subplot(gs[:, 19:])

        cm_nodes = plt.get_cmap(cmap_nodes)
        cm_edges = plt.get_cmap(cmap_edges)

        ax_nodes = self._plot_nodes(cmap_nodes=cmap_nodes, node_searched_labels=node_search_labels, ax=ax_nodes)

        # colorbar nodes
        norm_nodes = mpl.colors.Normalize(vmin=np.min(self._node_heuristic), vmax=np.max(self._node_heuristic))
        cb_nodes = mpl.colorbar.ColorbarBase(ax_nodes_cb, cmap=cm_nodes, norm=norm_nodes, orientation='vertical')
        # cb_nodes.set_label('delay (ms)')
        cb_nodes.set_label('heuristic init (a.u.)')

        # draw edges
        heuristics = []
        for n1, n2, d in self.graph.edges.data():
            heuristics.append(d['heur'])

        ax_edges = self._plot_edges(cmap_edges=cmap_edges, ax=ax_edges)

        # edges colorbar
        norm_edges = mpl.colors.Normalize(vmin=np.min(heuristics), vmax=np.max(heuristics))
        cb_edges = mpl.colorbar.ColorbarBase(ax_edges_cb, cmap=cm_edges, norm=norm_edges, orientation='vertical')
        cb_edges.set_label('heuristic (a.u.)')

        ax_nodes.set_title("Nodes \n(by node heuristic)")
        ax_edges.set_title("Edges \n(by edge heuristic)")

        fig.subplots_adjust(top=0.85)
        fig.suptitle("Graph", fontsize=20, y=0.95)

        return fig

    def plot_branches(self, fig=None, cmap='rainbow', pitch=None):
        if fig is None:
            fig = plt.figure()
        ax_all = fig.add_subplot(1, 2, 1)
        ax_clean = fig.add_subplot(1, 2, 2)

        cm = plt.get_cmap(cmap)

        ax_all.set_title(f"All branches {len(self._paths_raw)}")
        plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=self._vscale, ax=ax_all,
                      pitch=pitch)
        plot_template(self.template, self.locations, colors='k', channels=self.selected_channels,
                      vscale=self._vscale, ax=ax_all, pitch=pitch)
        plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                      vscale=self._vscale, ax=ax_all, pitch=pitch)

        branch_colors = []
        for i, path in enumerate(self._paths_raw):
            color = cm(i / len(self._paths_raw))
            branch_colors.append(color)
            ax_all.plot(self.locations[path, 0], self.locations[path, 1], marker='o', ls='-',
                        color=color)

        ax_clean.set_title(f"Clean branches {len(self._paths_clean)}")
        plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=self._vscale, ax=ax_clean,
                      pitch=pitch)
        plot_template(self.template, self.locations, colors='k', channels=self.selected_channels,
                      vscale=self._vscale, ax=ax_clean, pitch=pitch)
        plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                      vscale=self._vscale, ax=ax_clean, pitch=pitch)

        for i, branch in enumerate(self.branches):
            path = branch['channels']
            color = branch_colors[branch['raw_path_idx']]
            ax_clean.plot(self.locations[path, 0], self.locations[path, 1], marker='o', ls='-',
                          color=color)
        fig.subplots_adjust(top=0.8)
        fig.suptitle("Branches", fontsize=20, y=0.95)

        return fig

    def _plot_raw_branches(self, plot_full_template=False, ax=None, cmap="rainbow", pitch=None):
        if ax is None:
            fig, ax_raw = plt.subplots()
        else:
            ax_raw = ax
        cm = plt.get_cmap(cmap)

        if plot_full_template:
            plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=self._vscale, ax=ax_raw,
                          pitch=pitch)
        plot_template(self.template, self.locations, colors='k', channels=self.selected_channels,
                      vscale=self._vscale, ax=ax_raw, pitch=pitch)
        plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                      vscale=self._vscale, ax=ax_raw, pitch=pitch)

        branch_colors = []
        for i, path in enumerate(self._paths_raw):
            color = cm(i / len(self._paths_raw))
            branch_colors.append(color)
            ax_raw.plot(self.locations[path, 0], self.locations[path, 1], marker='o', ls='-',
                        color=color)
        return ax_raw

    def plot_clean_branches(self, plot_full_template=False, ax=None, cmap="rainbow", pitch=None):
        if ax is None:
            fig, ax_clean = plt.subplots()
        else:
            ax_clean = ax
        cm = plt.get_cmap(cmap)

        branch_colors = []
        for i, path in enumerate(self._paths_raw):
            color = cm(i / len(self._paths_raw))
            branch_colors.append(color)
        if plot_full_template:
            plot_template(self.template, self.locations, colors='gray', alpha=0.3, vscale=self._vscale, ax=ax_clean,
                          pitch=pitch)
        plot_template(self.template, self.locations, colors='k', channels=self.selected_channels,
                      vscale=self._vscale, ax=ax_clean, pitch=pitch)
        plot_template(self.template, self.locations, colors='r', channels=[self.init_channel],
                      vscale=self._vscale, ax=ax_clean, pitch=pitch)

        for i, branch in enumerate(self.branches):
            path = branch['channels']
            color = branch_colors[branch['raw_path_idx']]
            ax_clean.plot(self.locations[path, 0], self.locations[path, 1], marker='o', ls='-',
                          color=color)
        return ax_clean

    def _plot_nodes(self, cmap_nodes="viridis", node_searched_labels=False,
                    color_by='heuristic', ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax_nodes = ax
        else:
            ax_nodes = ax

        cm_nodes = plt.get_cmap(cmap_nodes)

        # draw nodes
        node_pos = {}
        for node in self.graph.nodes:
            node_pos[node] = self.locations[node]

        if color_by == 'heuristic':
            node_colors = [cm_nodes(h / np.max(self._node_heuristic)) for h in self._node_heuristic]
        elif color_by == 'latency':
            peak_times_selected = self.peak_times[np.array(self.graph.nodes)]
            peak_times_n = (self.peak_times - np.min(peak_times_selected)) / np.ptp(peak_times_selected)
            node_colors = [cm_nodes(peak_times_n[n]) for n in self.graph.nodes]
        elif color_by == 'amplitude':
            amps_selected = self.amplitudes[np.array(self.graph.nodes)]
            amps_n = (self.amplitudes - np.min(amps_selected)) / np.ptp(amps_selected)
            node_colors = [cm_nodes(amps_n[n]) for n in self.graph.nodes]
        else:
            raise ValueError
        nx.draw_networkx_nodes(self.graph, pos=node_pos, node_color=node_colors, ax=ax_nodes, node_size=10)
        ax_nodes.axis('off')

        for node_searched in self._nodes_searched:
            ax_nodes.scatter([self.locations[node_searched, 0]], [self.locations[node_searched, 1]],
                             marker='o', s=15, zorder=10,
                             facecolor=None, edgecolor='r')
            if node_searched_labels:
                ax_nodes.text(self.locations[node_searched, 0], self.locations[node_searched, 1], str(node_searched),
                              color='r')

        ax_nodes.plot(self.locations[self.init_channel, 0], self.locations[self.init_channel, 1],
                      color='r', marker='o', markersize=8)

        return ax_nodes

    def _plot_edges(self, cmap_edges="rainbow", ax=None):
        if ax is None:
            fig, ax = plt.subplots()
            ax_edges = ax
        else:
            ax_edges = ax

        cm_edges = plt.get_cmap(cmap_edges)

        # draw nodes
        node_pos = {}
        for node in self.graph.nodes:
            node_pos[node] = self.locations[node]
        heuristics = []
        for n1, n2, d in self.graph.edges.data():
            heuristics.append(d['heur'])

        edge_colors = [cm_edges(ec / np.max(heuristics)) for ec in heuristics]

        nx.draw_networkx_edges(self.graph, pos=node_pos, edge_color=edge_colors, ax=ax_edges, width=0.5,
                               arrowsize=2)
        ax_edges.axis('off')
        ax_edges.plot(self.locations[self.init_channel, 0], self.locations[self.init_channel, 1],
                      color='r', marker='o', markersize=8)

        return ax_edges

    def _is_path_valid(self, path):
        if self.compute_path_length(path) >= self._min_path_length and \
                len(path) >= self._min_path_points:
            return True
        else:
            return False

    def plot_velocities(self, fig=None, cmap='rainbow'):
        if fig is None:
            fig = plt.figure()
        ax_vel = fig.add_subplot(111)

        cm = plt.get_cmap(cmap)

        branch_colors = []
        for i, path in enumerate(self._paths_raw):
            color = cm(i / len(self._paths_raw))
            branch_colors.append(color)

        for i, branch in enumerate(self.branches):
            color = branch_colors[branch['raw_path_idx']]
            plot_velocity(np.array(branch['peak_times']), np.array(branch['distances']),
                          branch['velocity'], branch['offset'], color=color, r2=branch['r2'],
                          ax=ax_vel)

        ax_vel.spines['top'].set_visible(False)
        ax_vel.spines['right'].set_visible(False)

        return fig

    def _graph_dist(self, n1, n2):
        node1 = self.graph.nodes[n1]
        node2 = self.graph.nodes[n2]
        loc1 = node1['location']
        loc2 = node2['location']
        distance = np.linalg.norm(loc1 - loc2)
        return ((distance - self._min_dist_node) / (self._max_dist_node - self._min_dist_node)) ** self._distance_exp

    def _compute_path_avg_heur(self, path):
        heur = 0
        for p_i, p in enumerate(path[:-1]):
            heur += self.graph.get_edge_data(p, path[p_i + 1])['heur']
        heur /= len(path)
        return heur

    def _get_adjacent_channels(self, node, distance_thresh=None, n_neighbors=None):
        assert distance_thresh is not None or n_neighbors is not None, "either 'distance_thresh' or 'n_neighbors' " \
                                                                       "should be specified"
        distances = np.array([np.linalg.norm(self.locations[node] - self.locations[ch])
                              for ch in self.selected_channels])
        if distance_thresh is not None:
            dist_within = np.where(distances < distance_thresh)[0]
            adj_channels = self.selected_channels[dist_within]
        else:
            sorted_dist_idxs = np.argsort(distances)
            adj_channels = self.selected_channels[sorted_dist_idxs][:n_neighbors]
        adj_channels = np.delete(adj_channels, np.where(adj_channels == node)[0])
        return adj_channels

    def _get_earlier_channels_within_distance(self, node):
        selected_channels_no_init = np.delete(self.selected_channels,
                                              np.where(self.selected_channels == self.init_channel)[0])
        idx_earlier = np.where(self.peak_times[selected_channels_no_init] < self.peak_times[node])[0]
        channels_earlier = selected_channels_no_init[idx_earlier]
        amplitudes_earlier = self.amplitudes[selected_channels_no_init][idx_earlier]

        dist_to_init = np.linalg.norm(self.locations[node] - self.locations[self.init_channel])

        # if no peaks earlier --> connect to init_chan
        channel_neighbors = []
        if len(channels_earlier) == 0:
            if dist_to_init < self._max_distance_to_init:
                channel_neighbors = [self.init_channel]
        else:
            distances_earlier = np.array([np.linalg.norm(self.locations[node] - loc)
                                          for loc in self.locations[channels_earlier]])
            dist_from_init = np.linalg.norm(self.locations[node] - self.locations[self.init_channel])

            if self._neighbor_selection == "dist":
                neighbor_idxs = np.argsort(distances_earlier)[:int(self._n_neighbors)]
                channels_sorted = channels_earlier[neighbor_idxs]
                distances_earlier_sorted = distances_earlier[neighbor_idxs]
            elif self._neighbor_selection == "amp":
                # select 2*n_neigbors based on distance and amplitude
                neighbor_idxs = np.argsort(distances_earlier)[:int(2 * self._n_neighbors)]

                distance_neighbors = distances_earlier[neighbor_idxs]
                amps_neighbors = amplitudes_earlier[neighbor_idxs]

                if len(distance_neighbors) > 1:
                # combine dist and amp
                    dist_neighbors_norm = (distance_neighbors - np.min(distance_neighbors)) / np.ptp(distance_neighbors)
                    amps_neighbors_norm = 1 - (amps_neighbors - np.min(amps_neighbors)) / np.ptp(amps_neighbors)

                    heur_neighbors = self._edge_dist_amp_ratio * dist_neighbors_norm + \
                                     (1 - self._edge_dist_amp_ratio) * amps_neighbors_norm
                    neighbor_max_heur_idxs = np.argsort(heur_neighbors)[:self._n_neighbors]
                    distances_earlier_sorted = distances_earlier[neighbor_idxs][neighbor_max_heur_idxs]
                    channels_sorted = channels_earlier[neighbor_idxs][neighbor_max_heur_idxs]
                else:
                    channels_sorted = channels_earlier[neighbor_idxs]
                    distances_earlier_sorted = distance_neighbors


            # remove dist farther than max_distance
            neighbor_valid_idxs = np.where((distances_earlier_sorted > 0) &
                                           (distances_earlier_sorted < self._max_distance_for_edge))[0]
            channel_neighbors = channels_sorted[neighbor_valid_idxs]

            if np.all(distances_earlier_sorted > dist_from_init) and dist_to_init < self._max_distance_to_init:
                channel_neighbors = np.append(channel_neighbors, [self.init_channel])

        return channel_neighbors

    def get_full_path(self, path, min_dist=10):
        full_path_all = []
        all_other = np.array([], dtype="int")
        for p_i, p in enumerate(path[:-1]):
            full_path_all.append(p)
            p1 = p
            p2 = path[p_i + 1]

            dists = []
            for lp in self.locations[self.selected_channels]:
                dists.append(distance_numpy(self.locations[p1], self.locations[p2], lp))
            dists = np.array(dists)
            other_channels = self.selected_channels[np.where((dists < min_dist) & (dists > 0))]
            # pto = self.peak_times[other_channels]
            # other_channels = other_channels[(pto <= gtr.peak_times[p1]) & (pto >= gtr.peak_times[p2])]
            if len(other_channels) > 0:
                dist_to_p2 = np.array([np.linalg.norm(self.locations[o] - self.locations[p2]) for o in other_channels])
                other_channels = other_channels[np.argsort(dist_to_p2)[::-1]]
                for other in other_channels:
                    full_path_all.append(other)
                all_other = np.concatenate((all_other, other_channels))
        full_path_all.append(path[-1])

        return full_path_all
