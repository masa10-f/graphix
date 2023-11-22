from itertools import combinations

import networkx as nx
import numpy as np
import pyzx as zx

from graphix.gflow import flow, gflow
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
        self.layers = dict()
        self.set_input_nodes(inputs)
        self.output_nodes = outputs

    def add_node(self, node: int, plane=None, angle=None, output=False):
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
        super().add_node(node, plane=plane, angle=angle, output=output)

    def assign_measurement_info(self, node: int, plane: str, angle: int):
        """asign measurement info to a node

        Parameters
        ----------
        node: int
            node to asign measurement info
        plane: str
            measurement plane
        angle: int
            measurement angle
        """
        self.nodes[node]["plane"] = plane
        self.nodes[node]["angle"] = angle

        self.nodes[node]["output"] = False

    def set_input_nodes(self, nodes: set):
        """add input nodes to graph

        Parameters
        ----------
        nodes: list
            list of input nodes
        """
        self.input_nodes = nodes
        for node in nodes:
            self.add_node(node, output=True)

    def get_meas_planes(self):
        """get measurement planes of the graph

        Returns
        -------
        meas_planes: dict
            dictionary of measurement planes
        """
        meas_planes = dict()
        for node in self.nodes:
            if self.nodes[node]["output"]:
                continue
            meas_planes[node] = self.nodes[node]["plane"]

        return meas_planes

    def get_pattern(self):
        """returns the pattern of the graph

        Returns
        -------
        Pattern: graphix.pattern.Pattern
            pattern of the graph
        """
        pattern = Pattern(input_nodes=self.input_nodes)
        for node in self.nodes:
            pattern.add(["N", node])

        for edge in self.edges:
            pattern.add(["E", edge])

        x_signals, z_signals = self.collect_signals()

        depth = len(self.layers)
        for k in range(depth - 1, 0, -1):
            layer = self.layers[k]
            for node in layer:
                pattern.add(
                    [
                        "M",
                        node,
                        self.nodes[node]["plane"],
                        self.nodes[node]["angle"],
                        x_signals[node],
                        z_signals[node],
                    ]
                )

        for node in self.output_nodes:
            if len(x_signals[node]) > 0:
                pattern.add(["X", node, x_signals[node]])
            if len(z_signals[node]) > 0:
                pattern.add(["Z", node, z_signals[node]])

        pattern.output_nodes = self.output_nodes

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
        z_signals = {node: set() for node in self.nodes}

        for node in self.nodes - set(self.output_nodes):
            for node_fg in self.flow[node]:
                x_signals[node_fg] |= {node}

            odd_neighbors = self.odd_neighbors(self.flow[node])
            for node_fg in odd_neighbors:
                if node_fg not in self.output_nodes:
                    if node == node_fg:
                        continue
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
        local_complemented_edges = set(self.edges).symmetric_difference(neighbors_complete_edges)
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
                        self.flow[node] = self.flow[node].symmetric_difference(self.flow[target])
                else:
                    pass

    def update_flow(self):
        """Update the flow of the graph"""
        fg, l_k = flow(
            self,
            input=set(self.input_nodes),
            output=set(self.output_nodes),
            meas_planes=self.get_meas_planes(),
        )
        if fg == None:
            fg, l_k = gflow(
                self,
                input=set(self.input_nodes),
                output=set(self.output_nodes),
                meas_planes=self.get_meas_planes(),
            )

        if fg == None:
            raise ValueError("No flow found")

        self.flow = fg
        for node, k in l_k.items():
            if k not in self.layers.keys():
                self.layers[k] = {node}
            else:
                self.layers[k] = self.layers[k] | {node}

    def odd_neighbors(self, nodes: set):
        """Find odd neighbors of a node

        Parameters
        ----------
        node: set
            Nodes to find odd neighbors

        Returns
        -------
        odd_neighbors: set
            Set of odd neighbors
        """
        odd_neighbors = set()
        for node in nodes:
            odd_neighbors ^= set(self.neighbors(node))
        return odd_neighbors

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

        complete_edges_uv_uvnot = {(i, j) for i in uv_neighbors for j in u_vnot_neighbors}
        complete_edges_uv_unotv = {(i, j) for i in uv_neighbors for j in unot_v_neighbors}
        complete_edges_uvnot_unotv = {(i, j) for i in u_vnot_neighbors for j in unot_v_neighbors}

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

    def to_pyzx(self):
        graph = zx.Graph()
        index_map = dict()
        for node in self.nodes - set(self.output_nodes):
            index = graph.add_vertex(type=zx.VertexType.Z, phase=node.angle)
            index_map[node] = index
        for node in set(self.output_nodes):
            index = graph.add_vertex(type=zx.VertexType.BOUNDARY)
            index_map[node] = index
        for edge in self.edges:
            graph.add_edge(
                graph.edge(index_map[edge[0]], index_map[edge[1]]),
                type=zx.EdgeType.HADAMARD,
            )

        return graph

    def simulate_mbqc(self, **kwargs):
        """Simulate the graph using MBQC

        Returns
        -------
        simulator: graphix.simulator.Simulator
            Simulator object
        """
        pattern = self.get_pattern()
        return pattern.simulate_pattern(**kwargs)
