from grakel.kernels import HadamardCode, Propagation, WeisfeilerLehmanOptimalAssignment
import igraph
import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Structure
from grakel.utils import graph_from_networkx
from pymatgen.analysis.local_env import MinimumDistanceNN
from wwl import ContinuousWeisfeilerLehman
from hashlib import blake2b


class AdvancedStructureGraph(StructureGraph):
    @staticmethod
    def with_local_env_strategy(structure, strategy, weights=False):
        """
        Borrowed from original pymatgen StructureGraph class
        """

        if not strategy.structures_allowed:
            raise ValueError("Chosen strategy is not designed for use with structures!")

        sg = AdvancedStructureGraph.with_empty_graph(structure, name="bonds")

        for n, neighbors in enumerate(strategy.get_all_nn_info(structure)):
            for neighbor in neighbors:
                # local_env will always try to add two edges
                # for any one bond, one from site u to site v
                # and another form site v to site u: this is
                # harmless, so warn_duplicates=False
                sg.add_edge(
                    from_index=n,
                    from_jimage=(0, 0, 0),
                    to_index=neighbor["site_index"],
                    to_jimage=neighbor["image"],
                    weight=neighbor["weight"] if weights else None,
                    warn_duplicates=False,
                )

        return sg

    def get_grakel_hash(self, kernel):
        for i in range(len(self.graph.nodes)):
            self.graph.nodes[i]["element"] = self.structure.sites[i].specie.name

        ug = self.graph.to_undirected()
        gs = graph_from_networkx([ug], node_labels_tag="element")

        kernel.fit_transform(gs)

        kname = kernel.__class__.__name__
        if kname == "WeisfeilerLehmanOptimalAssignment" or kname == "WeisfeilerLehman":
            labels = kernel._inv_labels
        elif kname == "HadamardCode":
            labels = kernel._labels_enum
        elif kname == "Propagation":
            labels = kernel._enum_labels
        else:
            raise NotImplementedError

        print(labels)

        my_hash = blake2b(str(labels).encode(), digest_size=16).hexdigest()

        return my_hash

    def get_wwl_matrix(self, iterations):
        ug = self.graph.to_undirected()
        ug = igraph.Graph.from_networkx(ug)

        atomic_numbers = np.array([[self.structure.atomic_numbers]]).T

        cwl = ContinuousWeisfeilerLehman()
        seq = cwl.fit_transform([ug], atomic_numbers, iterations)

        return seq


if __name__ == "__main__":
    s0 = Structure.from_file(
        "data/cifs/mp-6945.cif",
    )

    s1 = Structure.from_file(
        "data/cifs/mp-7648.cif",
    )

    s0_graph = AdvancedStructureGraph.with_local_env_strategy(s0, MinimumDistanceNN())
    s1_graph = AdvancedStructureGraph.with_local_env_strategy(s1, MinimumDistanceNN())

    kernel = HadamardCode(n_iter=10)
    h0 = s0_graph.get_grakel_hash(kernel)
    h1 = s1_graph.get_grakel_hash(kernel)
    assert h0 == h1

    kernel = WeisfeilerLehmanOptimalAssignment()
    h0 = s0_graph.get_grakel_hash(kernel)
    h1 = s1_graph.get_grakel_hash(kernel)
    assert h0 == h1

    kernel = Propagation(t_max=10, M="H")
    h0 = s0_graph.get_grakel_hash(kernel)
    h1 = s1_graph.get_grakel_hash(kernel)

    kernel = Propagation(t_max=10, M="TV")
    h0 = s0_graph.get_grakel_hash(kernel)
    h1 = s1_graph.get_grakel_hash(kernel)

    assert h0 == h1
