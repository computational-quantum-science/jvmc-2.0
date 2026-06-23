from abc import abstractmethod
import jax.numpy as jnp
import inspect
import jax

from jVMC_exp.operator.discrete.base import Operator as BaseOperator
from jVMC_exp.global_defs import DT_OPERATORS_CPX
from jVMC_exp.sharding_config import MESH

def _has_kwargs(fun):
    sig = inspect.signature(fun)
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values())

class OperatorString(list):
    """
    A list of operators to be applied sequentially, with an associated scale factor.
    """
    def __init__(self, operators):
        super().__init__(operators)
        self._scale = [1]
        self._diagonal = False

    @property
    def scale(self):
        return lambda kw: jnp.prod(jnp.array([s(**kw) if callable(s) else s for s in self._scale]), dtype=DT_OPERATORS_CPX)
    
    @property
    def diagonal(self):
        return self._diagonal

class Operator(BaseOperator):
    def __init__(self, ldim, idx, diag, fermionic=False, *, spinless_fermion=None, spinful_fermion=False, spin=0):
        self._idx = idx
        self._diag = diag
        if spinless_fermion is None:
            spinless_fermion = bool(fermionic)
        if spinful_fermion and spin not in (0, 1):
            raise ValueError("spinful fermion operators require spin=0 (up) or spin=1 (down).")
        self._spinless_fermion = bool(spinless_fermion)
        self._spinful_fermion = bool(spinful_fermion)
        self._fermionic = self._spinless_fermion or self._spinful_fermion
        self._spin = int(spin)
        self._site_ldim = {idx: ldim} if ldim is not None else {}
        super().__init__(ldim)

    @property
    def idx(self):
        return self._idx
    
    @property
    def fermionic(self):
        return self._fermionic

    @property
    def spinless_fermion(self):
        return self._spinless_fermion

    @property
    def spinful_fermion(self):
        return self._spinful_fermion

    @property
    def spin(self):
        return self._spin
    
    @property
    def diag(self):
        return self._diag
    
    @property
    @abstractmethod
    def mat_els(self):
        pass
    
    @property
    @abstractmethod
    def map(self):
        pass
        
    def _get_list_of_strings(self):
        """
        Flatten the operator tree into a list of operator strings.
        Each operator string is an OperatorString (list subclass) of leaf operators 
        to be applied sequentially, with a scale attribute.
        
        Returns: List of OperatorString objects
        """
        strings_stack = []
        op_stack = [(self, False)]

        while op_stack:
            node, visited = op_stack.pop()

            if not visited:
                op_stack.append((node, True))

                if isinstance(node, CompositeOperator):
                    op_stack.append((node.O_2, False))
                    op_stack.append((node.O_1, False))

                elif isinstance(node, ScaledOperator):
                    op_stack.append((node.O, False))

            else:
                if isinstance(node, CompositeOperator):
                    right = strings_stack.pop()
                    left = strings_stack.pop()

                    if node._label == 'sum':
                        # Sum: concatenate the two lists of operator strings
                        strings_stack.append(left + right)
                    
                    elif node._label == 'mul':
                        # Product: create all combinations (distributive law)
                        out = []
                        for left_string in left:
                            for right_string in right:
                                # Combine the operators from both strings
                                combined = OperatorString(left_string + right_string)
                                # Multiply the prefactors
                                combined._scale.extend(left_string._scale)
                                combined._scale.extend(right_string._scale)
                                out.append(combined)
                        
                        strings_stack.append(out)

                elif isinstance(node, ScaledOperator):
                    lst = strings_stack.pop()
                    # Multiply the scale of each operator string
                    for s in lst:
                        s._scale.append(node.scalar)
                    strings_stack.append(lst)

                else:
                    # Leaf operator: wrap in an OperatorString
                    op_string = OperatorString([node])
                    strings_stack.append([op_string])        
        
        return strings_stack[0]

    def _compile(self):
        strings = self._get_list_of_strings()
        max_length = max(len(s) for s in strings)
        max_ldim = max(self._site_ldim.values())
        
        # Create identity operator for padding
        IdOp = IdentityOperator(max_ldim, 0)
        
        idxC = []
        mapC = []
        matElsC = []
        fermionicC = []
        spinlessFermionicC = []
        spinfulFermionicC = []
        spinC = []
        diagonal = []
        prefactors = []
        
        for op_string in strings:
            idx_row = []
            map_row = []
            matels_row = []
            fermionic_row = []
            spinless_fermionic_row = []
            spinful_fermionic_row = []
            spin_row = []
            d = 1
            
            n = len(op_string)
            for k in range(max_length):
                k_rev = n - 1 - k
                if k_rev >= 0:
                    op = op_string[k_rev]
                    idx_row.append(op.idx)
                    map_row.append(jnp.pad(op.map, (0, max_ldim - op.ldim)))
                    matels_row.append(jnp.pad(op.mat_els, (0, max_ldim - op.ldim)))
                    fermionic_row.append(1.0 if op.fermionic else 0.0)
                    spinless_fermionic_row.append(1.0 if op.spinless_fermion else 0.0)
                    spinful_fermionic_row.append(1.0 if op.spinful_fermion else 0.0)
                    spin_row.append(op.spin)
                    d *= op.diag
                else:
                    idx_row.append(IdOp.idx)
                    map_row.append(IdOp.map)
                    matels_row.append(IdOp.mat_els)
                    fermionic_row.append(0.0)
                    spinless_fermionic_row.append(0.0)
                    spinful_fermionic_row.append(0.0)
                    spin_row.append(0)
            
            idxC.append(idx_row)
            mapC.append(map_row)
            matElsC.append(matels_row)
            fermionicC.append(fermionic_row)
            spinlessFermionicC.append(spinless_fermionic_row)
            spinfulFermionicC.append(spinful_fermionic_row)
            spinC.append(spin_row)
            prefactors.append(op_string.scale)
            diagonal.append(d)
        
        self.idxC = jnp.array(idxC, dtype=jnp.int32)
        self.mapC = jnp.array(mapC, dtype=jnp.int32)
        self.matElsC = jnp.array(matElsC)
        self.fermionicC = jnp.array(fermionicC, dtype=jnp.int32)
        self.spinlessFermionicC = jnp.array(spinlessFermionicC, dtype=jnp.int32)
        self.spinfulFermionicC = jnp.array(spinfulFermionicC, dtype=jnp.int32)
        self.spinC = jnp.array(spinC, dtype=jnp.int32)
        self.diagC = jnp.array(diagonal, dtype=jnp.bool_)
        self.nondiagC = ~self.diagC
        self.first_diag_idx = jnp.where(self.diagC)[0][0] if jnp.any(self.diagC) else jnp.zeros((len(self.diagC)), dtype=jnp.bool_)
        self.prefactorsC = prefactors
        self._is_compiled = True

    def _get_conn_elements(self, s, kwargs):
        sampleShape = s.shape
        s = s.ravel()
        dim = s.shape[0]
        mask = jnp.tril(jnp.ones((dim, dim), dtype=jnp.int32), -1).T
        sting_ids = jnp.arange(len(self.prefactorsC))

        def proccess_string(s, id, idx, map, matEls, spinless_fermi, spinful_fermi, spin):
            """
            Apply one operator string to ``s``. Spinful fermion signs use the
            ordering (L-1, down), (L-1, up), ..., (0, down), (0, up), i.e.
            all modes on larger site indices first and down before up onsite.
            """
            def apply_operator(c, x):
                carry_sample, carry_matEl = c
                idx, map, matEl, spinless_fermi, spinful_fermi, spin = x

                sample_bits = carry_sample.astype(jnp.int32)

                spinless_occ = jnp.bitwise_and(sample_bits, 1)
                spinless_count = jnp.sum(mask[idx] * spinless_occ)
                spinless_parity = jnp.bitwise_and(spinless_count, 1)
                spinless_sign = 1 - 2 * spinless_parity

                up_occ = jnp.bitwise_and(sample_bits, 1)
                down_occ = jnp.bitwise_and(jnp.right_shift(sample_bits, 1), 1)
                total_occ = up_occ + down_occ
                right_count = jnp.sum(mask[idx] * total_occ)
                onsite_count = jnp.where(spin == 0, down_occ[idx], 0)
                spinful_parity = jnp.bitwise_and(right_count + onsite_count, 1)
                spinful_sign = 1 - 2 * spinful_parity

                fermi_sign = jnp.where(
                    spinful_fermi != 0,
                    spinful_sign,
                    jnp.where(spinless_fermi != 0, spinless_sign, 1),
                ).astype(DT_OPERATORS_CPX)
                carry_matEl_new = carry_matEl * matEl[carry_sample[idx]] * fermi_sign
                carry_sample_new = carry_sample.at[idx].set(map[carry_sample[idx]])

                return (carry_sample_new, carry_matEl_new), None
            
            prefactor = jax.lax.switch(id, self.prefactorsC, kwargs)
            prefactor = jax.lax.pcast(prefactor, MESH.axis_names, to='varying')
            (s_p, matEl), _ = jax.lax.scan(apply_operator, (s, prefactor), (idx, map, matEls, spinless_fermi, spinful_fermi, spin))

            return s_p.reshape(sampleShape), matEl
        
        s_p, mat_els = jax.vmap(proccess_string, in_axes=(None,) + (0,) * 7)(
            s, sting_ids, self.idxC, self.mapC, self.matElsC, self.spinlessFermionicC, self.spinfulFermionicC, self.spinC
        )
        
        mat_els_diag = jnp.sum(mat_els[self.diagC])
        s_p_nondiag = s_p[self.nondiagC]
        mat_els_nondiag = mat_els[self.nondiagC]

        s_p_out = jnp.concatenate([s.reshape(1, *sampleShape), s_p_nondiag], axis=0)
        mat_els_out = jnp.concatenate([mat_els_diag[None], mat_els_nondiag], axis=0)

        return s_p_out, mat_els_out
    
    @classmethod
    def _create_composite(cls, O_1, O_2, label):
        return CompositeOperator(O_1, O_2, label)
    
    @classmethod
    def _create_scaled(cls, O, scalar):
        return ScaledOperator(O, scalar)
    
class CompositeOperator(Operator):
    def __init__(self, O_1: Operator, O_2: Operator, label: str):
        for site in O_1._site_ldim.keys():
            if (site in O_2._site_ldim.keys()) and (O_1._site_ldim[site] != O_2._site_ldim[site]):
                raise ValueError(f"Can't combine two operators with different local dimensions. "
                                 f"Got ldim={O_1._site_ldim[site]} and ldim={O_2._site_ldim[site]} at site {site}")
        super().__init__(None, None, None, None)

        self._site_ldim = {**O_1._site_ldim, **O_2._site_ldim}
        self.O_1 = O_1
        self.O_2 = O_2
        self._label = label

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass

class ScaledOperator(Operator):
    def  __init__(self, O: Operator, scalar):
        if callable(scalar) and (not _has_kwargs(scalar)):
            raise ValueError('Any callable that multiplies an operator has to have **kwargs in its argument.')

        super().__init__(None, None, None, None)
        self._site_ldim = O._site_ldim
        self.scalar = scalar
        self.O = O

    @property
    def mat_els(self):
        pass
    
    @property
    def map(self):
        pass

class IdentityOperator(Operator):
    def __init__(self, ldim, idx):
        super().__init__(ldim, idx, True, False)

    @property
    def mat_els(self):
        return jnp.ones(self.ldim, dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.arange(self.ldim, dtype=jnp.int32)
    
class _Creation(Operator):
    def __init__(self, idx, fermionic):
        super().__init__(2, idx, False, spinless_fermion=fermionic)

    @property
    def mat_els(self):
        return jnp.array([1, 0], dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.array([1, 0], dtype=jnp.int32)

class _Annihilation(Operator):
    def __init__(self, idx, fermionic):
        super().__init__(2, idx, False, spinless_fermion=fermionic)

    @property
    def mat_els(self):
        return jnp.array([0, 1], dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.array([0, 0], dtype=jnp.int32)
    
class SigmaX(Operator):
    def __init__(self, idx):
        super().__init__(2, idx, False, False)

    @property
    def mat_els(self):
        return jnp.ones(self.ldim, dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.array([1, 0], dtype=jnp.int32)

class SigmaY(Operator):
    def __init__(self, idx):
        super().__init__(2, idx, False, False)

    @property
    def mat_els(self):
        return jnp.array([1j, -1j], dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.array([1, 0], dtype=jnp.int32)
    
class SigmaZ(Operator):
    def __init__(self, idx):
        super().__init__(2, idx, True, False)

    @property
    def mat_els(self):
        return jnp.array([1, -1], dtype=DT_OPERATORS_CPX)
    
    @property
    def map(self):
        return jnp.arange(self.ldim, dtype=jnp.int32)
    
class SigmaPlus(_Creation):
    def __init__(self, idx):
        super().__init__(idx, False)

class SigmaMinus(_Annihilation):
    def __init__(self, idx):
        super().__init__(idx, False)
    
class _Number(Operator):
    def __init__(self, idx):
        super().__init__(2, idx, True, False)

    @property
    def mat_els(self):
        return jnp.array([0, 1], dtype=DT_OPERATORS_CPX)

    @property
    def map(self):
        return jnp.array([0, 1], dtype=jnp.int32)


def _spin_index(spin):
    if isinstance(spin, str):
        spin_key = spin.lower()
        if spin_key in {"up", "u", "↑"}:
            return 0
        if spin_key in {"down", "dn", "d", "↓"}:
            return 1
        raise ValueError("spin must be 0/1 or 'up'/'down'.")
    spin_idx = int(spin)
    if spin_idx not in (0, 1):
        raise ValueError("spin must be 0/1 or 'up'/'down'.")
    return spin_idx


class _SpinfulCreation(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, False, spinful_fermion=True, spin=_spin_index(spin))

    @property
    def mat_els(self):
        if self.spin == 0:
            return jnp.array([1, 0, 1, 0], dtype=DT_OPERATORS_CPX)
        return jnp.array([1, 1, 0, 0], dtype=DT_OPERATORS_CPX)

    @property
    def map(self):
        if self.spin == 0:
            return jnp.array([1, 1, 3, 3], dtype=jnp.int32)
        return jnp.array([2, 3, 2, 3], dtype=jnp.int32)


class _SpinfulAnnihilation(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, False, spinful_fermion=True, spin=_spin_index(spin))

    @property
    def mat_els(self):
        if self.spin == 0:
            return jnp.array([0, 1, 0, 1], dtype=DT_OPERATORS_CPX)
        return jnp.array([0, 0, 1, 1], dtype=DT_OPERATORS_CPX)

    @property
    def map(self):
        if self.spin == 0:
            return jnp.array([0, 0, 2, 2], dtype=jnp.int32)
        return jnp.array([0, 1, 0, 1], dtype=jnp.int32)


class _SpinfulNumber(Operator):
    def __init__(self, idx, spin):
        super().__init__(4, idx, True, False, spin=_spin_index(spin))

    @property
    def mat_els(self):
        if self.spin == 0:
            return jnp.array([0, 1, 0, 1], dtype=DT_OPERATORS_CPX)
        return jnp.array([0, 0, 1, 1], dtype=DT_OPERATORS_CPX)

    @property
    def map(self):
        return jnp.arange(self.ldim, dtype=jnp.int32)


def Creation(idx, spin=None):
    if spin is None:
        return _Creation(idx, True)
    return _SpinfulCreation(idx, spin)


def Annihilation(idx, spin=None):
    if spin is None:
        return _Annihilation(idx, True)
    return _SpinfulAnnihilation(idx, spin)


def Number(idx, spin=None):
    if spin is None:
        return _Number(idx)
    return _SpinfulNumber(idx, spin)


def SpinfulCreation(idx, spin):
    return _SpinfulCreation(idx, spin)


def SpinfulAnnihilation(idx, spin):
    return _SpinfulAnnihilation(idx, spin)


def SpinfulNumber(idx, spin):
    return _SpinfulNumber(idx, spin)
