import os
import time
import datetime
import torch
import qimpy as qp
import numpy as np
from mpi4py import MPI
from psutil import cpu_count


class RunConfig:
    """Run configuration: hardware resources including CPU cores or GPU,
    and MPI communicators to be used by current QimPy instance.

    RunConfig must be created before any torch CUDA calls,
    so that a single CUDA context is associated with this process.
    Otherwise, on multi-GPU systems, CUDA MPI will subsequently fail."""

    def __init__(self, *, comm=None, cores=None, process_grid=(-1, -1, -1)):
        """
        Parameters
        ----------
        comm : mpi4py.MPI.COMM or None, optional
            Top-level MPI communicator.
            Default (None) corresponds to mpi4py.MPI.COMM_WORLD.
        cores : int or None, optional
            Number of CPU cores (and hence torch threads) to use per process.
            Default (None) will divide up available physical cores equally
            between processes on each node.
            Note that the environment variable SLURM_CPUS_PER_TASK (typically
            set by SLURM) will override cores, if set.
        process_grid : tuple of 3 ints, optional
            Division of processes into a grid, where the dimensions in order
            are replicas (eg. in NEB or phonon perturbations), k-points and
            bands/basis (split over basis usually, and over bands for FFTs).
            The product should equal the total number of processes.
            One or more dimensions can be set to -1; these are then
            determined to minimize communication by maximizing the number
            of processes for the replica and then k-point splits, limited by
            the number of available replicas / k-points, once they become
            known later in the initialization process. Note that this leaves
            the process grid uninitialized until provide_n_tasks() is called
            from Ion / Kpoints with the number of replicas / k-points.
            Default: (-1, -1, -1) asks for all process grid dimensions to be
            determined from the number of available replicas / k-points."""

        # Set and report start time:
        self.t_start = time.time()  # start time used by clock()
        qp.log.info('Start time: ' + time.ctime(self.t_start))

        # MPI initialization:
        self.MPI = MPI  # avoid having to import from mpi4py in each module
        self.comm = MPI.COMM_WORLD if (comm is None) else comm
        self.i_proc = self.comm.Get_rank()
        self.n_procs = self.comm.Get_size()
        self.is_head = (self.i_proc == 0)
        self.mpi_type = {  # Map from relevant torch to MPI datatypes
            torch.int32: MPI.INT,
            torch.int64: MPI.LONG,
            torch.float32: MPI.FLOAT,
            torch.float64: MPI.DOUBLE,
            torch.complex64: MPI.COMPLEX,
            torch.complex128: MPI.DOUBLE_COMPLEX
        }

        # Select GPU before initializing torch:
        self.comm_node = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
        i_proc_node = self.comm_node.Get_rank()
        n_procs_node = self.comm_node.Get_size()
        cuda_devs = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devs:
            # Select one GPU and make sure it's only one visible to torch:
            cuda_devs = [int(s) for s in cuda_devs.split(',')]
            cuda_dev_selected = cuda_devs[i_proc_node % len(cuda_devs)]
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_dev_selected)
            n_gpus = min(1., len(cuda_devs) / n_procs_node)
        else:
            # Disable GPUs unless explicitly requested:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            n_gpus = 0.

        # Initialize torch:
        self.cpu = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            self.use_cuda = True
            torch.cuda.device(self.device)
        else:
            self.device = self.cpu
            self.use_cuda = False
            n_gpus = 0.
        torch.set_default_tensor_type(torch.DoubleTensor)

        # Threads:
        slurm_threads = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_threads:
            self.n_threads = int(slurm_threads)
        elif cores is None:
            # Divide up threads available on node:
            n_cores = cpu_count(logical=False)
            core_start = (i_proc_node * n_cores) // n_procs_node
            core_stop = ((i_proc_node + 1) * n_cores) // n_procs_node
            self.n_threads = core_stop - core_start
        else:
            self.n_threads = cores
        assert(self.n_threads >= 1)
        torch.set_num_threads(self.n_threads)

        # Report total resources:
        run_totals = np.array([self.n_threads,  n_gpus])
        self.comm.Allreduce(MPI.IN_PLACE, run_totals, op=MPI.SUM)
        qp.log.info(
            'Run totals: {:g} processes, {:d} threads, {:d} GPUs'.format(
                self.n_procs, int(run_totals[0]), int(run_totals[1])))

        # Process grid:
        self.process_grid = np.array(process_grid, dtype=int)
        assert(self.process_grid.shape == (3,))
        self._setup_process_grid()

    def _setup_process_grid(self):
        '''Set up communicators if self.process_grid is fully specified, or
        report status and mark process grid determination as pending.'''
        # Check compatibility of known dimensions with total:
        prod_known = self.process_grid[self.process_grid != -1].prod()
        if self.n_procs % prod_known:
            raise ValueError(('Cannot distribute {:d} processes to'
                              + ' {:d} x {:d} x {:d} grid').format(
                                self.n_procs, *tuple(self.process_grid)))
        # Compute a single unknown dimension if present:
        n_unknown = np.count_nonzero(self.process_grid == -1)
        if n_unknown == 1:
            self.process_grid[self.process_grid == -1] = (self.n_procs
                                                          // prod_known)
            n_unknown = 0
        # Report grid as it is now:
        qp.log.info(('Process grid: {:d} replicas x {:d} k-points x {:d} '
                     + 'bands/basis{:s}').format(
                         *tuple(self.process_grid),
                         ' (-1\'s determined later)' if n_unknown else ''))
        # Initialize grid communicators whose dimensions are known:
        if self.process_grid[0] != -1 and (not hasattr(self, 'comm_r')):
            # split top-level communicator between replicas:
            # --- only needs outermost (replica) dimension to be known
            prod_inner = self.n_procs // self.process_grid[0]
            self.comm_r, self.comm_kb = comm_split_grid(
                self.comm, self.process_grid[0], prod_inner)
            self.n_procs_r = self.comm_r.Get_size()
            self.i_proc_r = self.comm_r.Get_rank()
            self.n_procs_kb = self.comm_kb.Get_size()
            self.i_proc_kb = self.comm_kb.Get_rank()
        if n_unknown == 0 and (not hasattr(self, 'comm_k')):
            # split intra-replica (kb) communicator to k and b:
            # --- needs all dimensions to be known
            self.comm_k, self.comm_b = comm_split_grid(
                self.comm_kb, self.process_grid[1], self.process_grid[2])
            self.n_procs_k = self.comm_k.Get_size()
            self.i_proc_k = self.comm_k.Get_rank()
            self.n_procs_b = self.comm_b.Get_size()
            self.i_proc_b = self.comm_b.Get_rank()

    def provide_n_tasks(self, dim, n_tasks):
        '''Inform RunConfig of the number of tasks to be split along a given
        dimension of the process grid. If that dimension is undetermined (-1),
        set it to a suitable value that is compatible with the total processes
        and any other known process grid dimensions, and with splitting n_tasks
        tasks with reasonable load balancing over this dimension.

        Parameters
        ----------
        dim : int
            Which dimension to provide n_tasks for (0 <= dim < 3)
        n_tasks : int
            Number of tasks available to split on this dimension of the
            process grid, used for setting or checking dimension of process
            grid for reasonable load balancing.'''
        def get_imbalance():
            'Compute cpu time% wasted in splitting n_tasks over n_procs_dim'
            n_tasks_each = qp.utils.ceildiv(n_tasks, n_procs_dim)
            return 100.*(1. - n_tasks / (n_tasks_each * n_procs_dim))
        imbalance_threshold = 20.  # max cpu time% waste to tolerate
        if self.process_grid[dim] == -1:
            # Dimension undetermined: set it based on n_tasks
            prod_known = self.process_grid[self.process_grid != -1].prod()
            n_procs_dim = self.n_procs // prod_known  # max possible value
            imbalance = get_imbalance()
            if imbalance > imbalance_threshold:
                # Drop primes factors starting from smallest till balanced:
                factors = qp.utils.prime_factorization(n_procs_dim)
                for factor in factors:
                    n_procs_dim //= factor
                    imbalance = get_imbalance()
                    if imbalance <= imbalance_threshold:
                        break
            assert(imbalance <= imbalance_threshold)
            # Set selected number of processes:
            self.process_grid[dim] = n_procs_dim
            self._setup_process_grid()

    def clock(self):
        "Time in seconds since start of this run."
        return time.time() - self.t_start

    def report_end(self):
        "Report end time and duration."
        t_stop = time.time()
        duration = datetime.timedelta(seconds=(t_stop - self.t_start))
        qp.log.info('\nEnd time: {:s} (Duration: {:s})'.format(
            time.ctime(t_stop), str(duration)))

    def fmt(self, tensor):
        'Standardized conversion of torch tensors for log'
        return np.array2string(
            tensor.to(self.cpu).numpy(),
            precision=8, suppress_small=True, separator=', ')


def comm_split_grid(comm, n_procs_o, n_procs_i):
    '''Split a communicator comm into an n_procs_o x n_procs_i grid,
    returning comm_o (with strided process ranks in comm) and
    comm_i (with contiguous process ranks in comm).'''
    # Check inputs:
    n_procs = comm.Get_size()
    i_proc = comm.Get_rank()
    assert(n_procs_o * n_procs_i == n_procs)
    # Determine size/ranks of o(uter) and i(nner) dimensions of process grid:
    i_proc_o = i_proc // n_procs_i
    i_proc_i = i_proc % n_procs_i
    # Initialize sub-groups:
    group = comm.Get_group()
    group_o = group.Incl(range(i_proc_i, n_procs, n_procs_i))
    group_i = group.Incl(range(i_proc_o*n_procs_i, (i_proc_o+1)*n_procs_i))
    # Initialize sub-communicators from groups:
    comm_o = comm.Create_group(group_o)
    comm_i = comm.Create_group(group_i)
    # Check communicator rank assignments and return:
    assert(comm_o.Get_size() == n_procs_o)
    assert(comm_i.Get_size() == n_procs_i)
    assert(comm_o.Get_rank() == i_proc_o)
    assert(comm_i.Get_rank() == i_proc_i)
    return comm_o, comm_i