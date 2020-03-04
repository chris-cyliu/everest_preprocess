from __future__ import absolute_import

import operators


class Executor(object):
    def __init__(self, ops_type, ops_param, in_data, out_data):
        self.ops_type_ = ops_type
        self.ops_param_ = ops_param
        self.in_data_ = in_data
        self.out_data_ = out_data

    def _op_map(self, op_name):
        op_name_dict = {
            'topk': operators.topk.TopKOp
        }

        return op_name_dict[op_name]

    def execute(self):
        in_data = self.in_data_
        out_data = list()

        for op_type in self.ops_type_:
            op = self._op_map(op_type)(self.ops_param_)
            op.forward(in_data, out_data)
            in_data = out_data

        self.out_data_ = out_data
