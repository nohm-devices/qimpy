import qimpy as qp
import numpy as np
import torch
import pathlib
import re
from typing import Optional, Union, List, TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils import RunConfig


class Ions:
    """TODO: document class Ions"""

    def __init__(self, *, rc: 'RunConfig',
                 coordinates: Optional[List] = None,
                 pseudopotentials: Optional[Union[str, List[str]]] = None):
        '''
        Parameters
        ----------
        TODO
        '''
        self.rc = rc
        qp.log.info('\n--- Initializing Ions ---')

        # Read ionic coordinates:
        if coordinates is None:
            coordinates = []
        assert isinstance(coordinates, list)
        self.n_ions: int = 0  #: number of ions
        self.n_types: int = 0  #: number of distinct ion types
        self.symbols: List[str] = []  #: symbol for each ion type
        self.positions = []  # position of each ion
        self.types = []      # type of each ion (index into symbols)
        self.ranges: List[range] = []     # range / slice to get each ion type
        self.M_initial = []  # initial magnetic moments
        type_start = 0
        for coord in coordinates:
            # Check for optional attributes:
            if len(coord) == 4:
                attrib = {}
            elif len(coord) == 5:
                attrib = coord[4]
                if not isinstance(attrib, dict):
                    raise ValueError('ion attributes must be a dict')
            else:
                raise ValueError('each ion must be 4 entries + optional dict')
            # Add new symbol or append to existing:
            symbol = str(coord[0])
            if (not self.symbols) or (symbol != self.symbols[-1]):
                self.symbols.append(symbol)
                self.n_types += 1
                if type_start != self.n_ions:
                    self.ranges.append(slice(type_start, self.n_ions))
                    type_start = self.n_ions
            # Add type and position of current ion:
            self.types.append(self.n_types-1)
            self.positions.append([float(x) for x in coord[1:4]])
            self.M_initial.append(attrib.get('M', None))
            self.n_ions += 1
        if type_start != self.n_ions:
            self.ranges.append(slice(type_start, self.n_ions))  # for last type

        # Check order:
        if len(set(self.symbols)) < self.n_types:
            raise ValueError(
                'coordinates must group ions of same type together')

        # Convert to tensors:
        self.positions = torch.tensor(self.positions, device=rc.device)
        self.types = torch.tensor(self.types, device=rc.device)
        # --- Fill in missing magnetizations (if any specified):
        M_lengths = set([(len(M) if isinstance(M, list) else 1)
                         for M in self.M_initial if M])
        if len(M_lengths) > 1:
            raise ValueError('All M must be same type: 3-vector or scalar')
        elif len(M_lengths) == 1:
            M_length = next(iter(M_lengths))
            assert((M_length == 1) or (M_length == 3))
            M_default = ([0., 0., 0.] if (M_length == 3) else 0.)
            self.M_initial = torch.tensor(
                [(M if M else M_default) for M in self.M_initial],
                device=rc.device, dtype=float)
        else:
            self.M_initial = None
        self.report()

        # Initialize pseudopotentials:
        self.pseudopotentials = []
        if pseudopotentials is None:
            pseudopotentials = []
        if isinstance(pseudopotentials, str):
            pseudopotentials = [pseudopotentials]
        for i_type, symbol in enumerate(self.symbols):
            fname = None  # full filename for this ion type
            symbol_variants = [
                symbol.lower(),
                symbol.upper(),
                symbol.capitalize()]
            # Check each filename provided in order:
            for ps_name in pseudopotentials:
                if ps_name.count('$ID'):
                    # wildcard syntax
                    for symbol_variant in symbol_variants:
                        fname_test = ps_name.replace('$ID', symbol_variant)
                        if pathlib.Path(fname_test).exists():
                            fname = fname_test  # found
                            break
                else:
                    # specific filename
                    basename = pathlib.PurePath(ps_name).stem
                    ps_symbol = re.split(r'[_\-\.]+', basename)[0]
                    if ps_symbol in symbol_variants:
                        fname = ps_name
                        if not pathlib.Path(fname).exists():
                            raise FileNotFoundError(fname)
                        break
                if fname:
                    break
            # Read pseudopotential file:
            if fname:
                self.pseudopotentials.append(
                    qp.ions.Pseudopotential(fname, rc))
            else:
                raise ValueError(f'no pseudopotential found for {symbol}')

        # Calculate total ionic charge (needed for number of electrons):
        self.Z_tot = sum(self.pseudopotentials[i_type].Z
                         * (slice_i.stop - slice_i.start)
                         for i_type, slice_i in enumerate(self.ranges))
        qp.log.info(f'\nTotal ion charge, Z_tot: {self.Z_tot:g}')

        # Initialize / check replica process grid dimension:
        n_replicas = 1  # this will eventually change for NEB / phonon DFPT
        rc.provide_n_tasks(0, n_replicas)

    def report(self):
        'Report ionic positions and attributes'
        qp.log.info(f'{self.n_ions} total ions of {self.n_types} types;'
                    ' positions:')
        # Fetch to CPU for reporting:
        positions = self.positions.to(self.rc.cpu).numpy()
        types = self.types.to(self.rc.cpu).numpy()
        M_initial = (None
                     if (self.M_initial is None)
                     else self.M_initial.to(self.rc.cpu).numpy())
        for i_ion, position in enumerate(positions):
            # Generate attribute string:
            attrib_str = ''
            attribs = {}
            if M_initial is not None:
                M_i = M_initial[i_ion]
                if np.linalg.norm(M_i):
                    attribs['M'] = M_i
            if attribs:
                attrib_str = ', ' + str(attribs).replace("'", '').replace(
                    'array(', '').replace(')', '')
            # Report:
            qp.log.info(
                f'- [{self.symbols[types[i_ion]]}, {position[0]:11.8f},'
                f' {position[1]:11.8f}, {position[2]:11.8f}{attrib_str}]')
