from itertools import combinations

import networkx as nx
import numpy as np

from graphix.pattern import Pattern


class MBQCGraph(nx.Graph):
    """MBQC graph

    Attributes
    ----------
        input: list
            input nodes
        output: list
            output nodes
        flow: dict
            (g)flow of the graph
        layers: list
            layers of the graph in terms of (g)flow
    """

    def __init__(self, inputs=[], outputs=[], **kwargs):
        super().__init__(**kwargs)
        """MBQC graph
        """
        self.flow = dict()
        self.layers = None
        self.input_nodes = inputs
        self.output_nodes = outputs

    def add_node(self, node: int, plane="XY", angle=0):
        """add a node to the graph

        Parameters
        ----------
        node: int
            node to add
        plane: str, optional
            measurement plane, by default "XY"
        angle: int, optional
            measurement angle, by default 0
        """
        super().add_node((node, {"plane": plane, "angle": angle}))

    def set_input_nodes(self, nodes: set):
        """add input nodes to graph

        Parameters
        ----------
        nodes: list
            list of input nodes
        """
        self.input_nodes = nodes
        self.add_nodes_from(nodes)

    def add_output_nodes(self, nodes: set):
        """add output nodes to graph

        Parameters
        ----------
        nodes: list
            list of output nodes
        """
        self.output_nodes.extend(list(nodes))
        super().add_nodes_from(nodes)

    def get_pattern(self):
        """returns the pattern of the graph

        Returns
        -------
        Pattern: graphix.pattern.Pattern
            pattern of the graph
        """
        pattern = Pattern(input_nodes=self.input_nodes, output_nodes=self.output_nodes)
        for node in self.nodes:
            pattern.add(["N", node])

        for edge in self.edges:
            pattern.add(["E", edge])

        x_signals, z_signals = self.collect_signals()

        depth = len(self.layers)
        for k in range(depth - 1, -1, -1):
            layer = self.layers[k]
            for node in layer:
                pattern.add(
                    [
                        "M",
                        node,
                        self.meas_planes[node],
                        self.meas_angles[node],
                        x_signals[node],
                        z_signals[node],
                    ]
                )

        for node in self.output_nodes:
            if len(x_signals[node]) > 0:
                pattern.add(["X", node, x_signals[node]])
            if len(z_signals[node]) > 0:
                pattern.add(["Z", node, z_signals[node]])

        return pattern

    def collect_signals(self):
        """collects the signals of the graph from the (g)flow

        Returns
        -------
        x_signals: dict
            dictionary of x signals
        z_signals: dict
            dictionary of z signals
        """
        x_signals = {node: set() for node in self.nodes}
        z_signals = dict()

        for node in self.nodes:
            for node_fg in self.flow[node]:
                x_signals[node_fg] |= {node}

            odd_neighbors = self.odd_neighbors(node)
            for node_fg in odd_neighbors:
                z_signals[node_fg] ^= {node}

        return x_signals, z_signals

    def local_complementation(self, target: int):
        """Apply local complementation to a node

        Parameters
        ----------
        target: int
            Node to apply local complementation
        """
        assert self.has_node(target), "Node not in graph"
        neighbors_target = list(self.neighbors(target))
        neighbors_complete_edges = combinations(neighbors_target, 2)
        local_complemented_edges = set(self.edges).symmetric_difference(
            neighbors_complete_edges
        )
        self.update(edges=local_complemented_edges)

        # modify measurement planes
        if self.meas_planes[target] == "XY":
            self.meas_planes[target] = "XZ"
        elif self.meas_planes[target] == "XZ":
            self.meas_planes[target] = "XY"
        elif self.meas_planes[target] == "YZ":
            pass

        non_output = set(self.nodes) - self.output_nodes - {target}
        for node in non_output:
            if self.meas_planes[node] == "XZ":
                self.meas_planes[node] = "YZ"
            elif self.meas_planes[node] == "YZ":
                self.meas_planes[node] = "XZ"
            elif self.meas_planes[node] == "XY":
                pass

        # modify flow
        if self.flow != None:
            if self.meas_planes[target] == "XY" or "XZ":
                self.flow[target] = self.flow[target].symmetric_difference({target})
            elif self.meas_planes[target] == "YZ":
                pass

            for node in non_output:
                odd_neighbors_node = self.find_odd_neighbors(self.flow[node])
                if target in odd_neighbors_node:
                    self.flow[node] = self.flow[node].symmetric_difference({target})
                    if self.meas_planes[node] != None:
                        self.flow[node] = self.flow[node].symmetric_difference(
                            self.flow[target]
                        )
                else:
                    pass

    def pivot(self, u: int, v: int):
        """Apply pivot to two nodes

        Parameters
        ----------
        u: int
            First node
        v: int
            Second node
        """
        u_neighbors = set(self.neighbors(u))
        v_neighbors = set(self.neighbors(v))
        uv_all_neighbors = u_neighbors.union(v_neighbors)

        uv_neighbors = u_neighbors.intersection(v_neighbors)
        u_vnot_neighbors = uv_all_neighbors.difference(v_neighbors)
        unot_v_neighbors = uv_all_neighbors.difference(u_neighbors)

        complete_edges_uv_uvnot = {
            (i, j) for i in uv_neighbors for j in u_vnot_neighbors
        }
        complete_edges_uv_unotv = {
            (i, j) for i in uv_neighbors for j in unot_v_neighbors
        }
        complete_edges_uvnot_unotv = {
            (i, j) for i in u_vnot_neighbors for j in unot_v_neighbors
        }

        E = set(self.edges)
        E = E.symmetric_difference(complete_edges_uv_uvnot)
        E = E.symmetric_difference(complete_edges_uv_unotv)
        E = E.symmetric_difference(complete_edges_uvnot_unotv)

        self.update(edges=E)
        self = nx.relabel_nodes(self, {u: v, v: u})

        # modify measurement planes
        for a in {u, v}:
            if self.meas_planes[a] == "XY":
                self.meas_planes[a] = "YZ"
            elif self.meas_planes[a] == "XZ":
                pass
            elif self.meas_planes[a] == "YZ":
                self.meas_planes[a] = "XY"

        # flow?

    def simulate_mbqc(self, **kwargs):
        """Simulate the graph using MBQC

        Returns
        -------
        simulator: graphix.simulator.Simulator
            Simulator object
        """
        pattern = self.get_pattern()
        return pattern.simulate(**kwargs)
