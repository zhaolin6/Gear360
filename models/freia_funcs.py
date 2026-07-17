
import torch.nn as nn
from torch.autograd import Variable

VERBOSE = False

class dummy_data:
    def __init__(self, *dims):
        self.dims = dims

    @property
    def shape(self):
        return self.dims

class Node:
    def __init__(self, inputs, module_type, module_args, name=None):
        self.inputs = inputs

        self.outputs = []

        self.module_type = module_type

        self.module_args = module_args

        self.input_dims = None

        self.module = None

        self.computed = None

        self.computed_rev = None

        self.id = None

        if name:
            self.name = name
        else:
            self.name = hex(id(self))[-6:]

        for i in range(255):
            exec('self.out{0} = (self, {0})'.format(i))

    def build_modules(self, verbose=VERBOSE):
        if not self.input_dims:
            self.input_dims = [n.build_modules(verbose=verbose)[c]
                               for n, c in self.inputs]

            try:
                self.module = self.module_type(self.input_dims, **self.module_args)
                print('Displaying node %s ' % (self.name))
            except Exception as e:
                print('Error in node %s' % (self.name))
                raise e

            if verbose:
                print("Input dimensions for node %s:" % (self.name))
                for d, (n, c) in zip(self.input_dims, self.inputs):
                    print("\t from node %s output #%i:" % (n.name, c), d)
                print()

            self.output_dims = self.module.output_dims(self.input_dims)

            self.n_outputs = len(self.output_dims)

        return self.output_dims

    def run_forward(self, op_list):
        if not self.computed:
            self.input_vars = []

            for i, (n, c) in enumerate(self.inputs):
                self.input_vars.append(n.run_forward(op_list)[c])

                n.outputs.append((self, i))

            self.computed = [(self.id, i) for i in range(self.n_outputs)]

            op_list.append((self.id, self.input_vars, self.computed))

        return self.computed

    def run_backward(self, op_list):
        assert len(self.outputs) > 0, "run_forward must be called first"

        if not self.computed_rev:
            output_vars = [(self.id, i) for i in range(self.n_outputs)]

            for n, c in self.outputs:
                n.run_backward(op_list)

            self.computed_rev = self.input_vars

            op_list.append((self.id, output_vars, self.computed_rev))

        return self.computed_rev


class InputNode(Node):
    def __init__(self, *dims, name='node'):
        super().__init__(inputs=[], module_type=None, module_args={})
        self.name = name
        self.data = dummy_data(*dims)
        self.outputs = []
        self.module = None
        self.computed_rev = None
        self.n_outputs = 1
        self.input_vars = []
        self.out0 = (self, 0)

    def build_modules(self, verbose=VERBOSE):
        return [self.data.shape]

    def run_forward(self, op_list):
        return [(self.id, 0)]

class OutputNode(Node):
    class dummy(nn.Module):
        def __init__(self, *args):
            super(OutputNode.dummy, self).__init__()

        def __call__(*args):
            return args

        def output_dims(*args):
            return args

    def __init__(self, inputs, name='node'):
        self.module_type, self.module_args = self.dummy, {}
        self.output_dims = []
        self.inputs = inputs
        self.input_dims, self.module = None, None
        self.computed = None
        self.id = None
        self.name = name

        for c, inp in enumerate(self.inputs):
            inp[0].outputs.append((self, c))

    def run_backward(self, op_list):
        return [(self.id, 0)]


class ReversibleGraphNet(nn.Module):

    def __init__(self, node_list, ind_in=None, ind_out=None, verbose=False, n_jac=1):
        super(ReversibleGraphNet, self).__init__()

        if ind_in is not None:
            self.ind_in = [ind_in] if isinstance(ind_in, int) else ind_in
        else:
            self.ind_in = [i for i, n in enumerate(node_list)
                           if isinstance(n, InputNode)]
            assert len(self.ind_in) > 0, "At least one input node must be specified"

        if ind_out is not None:
            self.ind_out = [ind_out] if isinstance(ind_out, int) else ind_out
        else:
            self.ind_out = [i for i, n in enumerate(node_list)
                            if isinstance(n, OutputNode)]
            assert len(self.ind_out) > 0, "At least one output node must be specified"

        self.return_vars = []
        self.input_vars = []

        self.node_list = node_list
        for i, n in enumerate(node_list):
            n.id = i
            print(n.id, ':', n.name)

        ops = []
        for i in self.ind_out:
            print('building node', i,type(node_list[i]))
            node_list[i].build_modules(verbose=verbose)
            node_list[i].run_forward(ops)

        variables = set()
        for o in ops:
            variables.update(o[1] + o[2])
        self.variables_ind = list(variables)

        self.indexed_ops = self.ops_to_indexed(ops)

        self.module_list = nn.ModuleList([n.module for n in node_list])
        self.variable_list = [Variable(requires_grad=True) for _ in variables]

        ops_rev = []
        for i in self.ind_in:
            node_list[i].run_backward(ops_rev)
        self.indexed_ops_rev = self.ops_to_indexed(ops_rev)

        self.n_jac = n_jac

    def ops_to_indexed(self, ops):
        result = []
        for o in ops:
            try:
                vars_in = [self.variables_ind.index(v) for v in o[1]]
            except ValueError:
                vars_in = -1

            vars_out = [self.variables_ind.index(v) for v in o[2]]

            if o[0] in self.ind_out:
                self.return_vars.append(self.variables_ind.index(o[1][0]))
                continue
            if o[0] in self.ind_in:
                self.input_vars.append(self.variables_ind.index(o[1][0]))
                continue

            result.append((o[0], vars_in, vars_out))

        self.return_vars.sort(key=lambda i: self.variables_ind[i][0])
        self.input_vars.sort(key=lambda i: self.variables_ind[i][0])

        return result

    def forward(self, x, rev=False):
        if rev:
            use_list = self.indexed_ops_rev
            input_vars, output_vars = self.return_vars, self.input_vars
        else:
            use_list = self.indexed_ops
            input_vars, output_vars = self.input_vars, self.return_vars

        if isinstance(x, (list, tuple)):
            assert len(x) == len(input_vars), (
                f"Input count mismatch: expected {len(input_vars)}, got {len(x)}")
            for i, var in enumerate(x):
                self.variable_list[input_vars[i]] = var
        else:
            assert len(input_vars) == 1, (
                f"Single input mismatch: expected 1, got {len(input_vars)}")
            self.variable_list[input_vars[0]] = x

        for o in use_list:
            module = self.module_list[o[0]]
            inputs = [self.variable_list[i] for i in o[1]]

            try:
                results = module(inputs, rev=rev)
            except TypeError:
                raise RuntimeError("Please make sure all used nodes are in the node list")

            for i, r in zip(o[2], results):
                self.variable_list[i] = r

        out = [self.variable_list[i] for i in output_vars]
        return out[0] if len(out) == 1 else out

    def jacobian(self, x=None, rev=False, run_forward=True):
        jacobian = [0.] * self.n_jac

        use_list = self.indexed_ops_rev if rev else self.indexed_ops

        if run_forward:
            if x is None:
                raise RuntimeError("Input data is required for reverse computation")
            self.forward(x, rev=rev)

        for o in use_list:
            try:
                node_jac = self.module_list[o[0]].jacobian(
                    [self.variable_list[i] for i in o[1]],
                    rev=rev
                )
                node_jac = [node_jac] if not isinstance(node_jac, list) else node_jac
                for i_j, jac in enumerate(node_jac):
                    jacobian[i_j] += jac
            except TypeError:
                raise RuntimeError("Please make sure all used nodes are in the node list")

        return jacobian



