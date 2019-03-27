# -*- coding: utf-8 -*-

"""
@author: Aghiles Salah <asalah@smu.edu.sg>
"""

import unittest
from cornac.data import GraphModule
from cornac.data import reader


class TestGraphModule(unittest.TestCase):

    def test_init(self):

        data = reader.read_uir('./tests/graph_data.txt', sep=' ')
        gmd = GraphModule(data=data)
        # gmd.build(global_id_map=None)
        self.assertEqual(len(gmd.raw_data), 12)
        self.assertEqual(len(gmd.map_data), 0)
        self.assertIsNone(gmd.matrix)


if __name__ == '__main__':
    unittest.main()