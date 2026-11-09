"""Execute Week 10 and save cell outputs incrementally without changing its source."""

from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    directory = Path(__file__).resolve().parent
    runtime = directory / '.runtime'
    for folder in ['matplotlib', 'ipython', 'jupyter']:
        (runtime / folder).mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(runtime / 'matplotlib')
    os.environ['IPYTHONDIR'] = str(runtime / 'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(runtime / 'jupyter')
    os.environ['PYTHONUNBUFFERED'] = '1'
    path = directory / 'Sammy_TCN_261111.ipynb'
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']

    def save_progress(cell, cell_index, **kwargs):
        nbformat.write(notebook, path)
        errors = [output for output in cell.get('outputs', [])
                  if output.get('output_type') == 'error']
        print(f'Cell {cell_index + 1}/{len(notebook.cells)}: '
              f'{"ERROR" if errors else "complete"}', flush=True)
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'stream':
                print(output.get('text', '').rstrip(), flush=True)
        if errors:
            print(errors, flush=True)

    client = NotebookClient(notebook, km=manager, timeout=1800,
                            resources={'metadata': {'path': str(directory)}},
                            on_cell_executed=save_progress)
    try:
        if '--robustness-only' in sys.argv:
            prerequisite_starts = [
                'from pathlib import Path\nimport os',
                'daily = pd.read_pickle',
                'def build_compact_shapes',
                'class Chomp1d',
                "base = train_family('daily_base'",
                'def ridge_features',
                'def correction_target',
                '# This section can be resumed after a kernel restart',
                'def extension_apply',
            ]
            with client.setup_kernel(cleanup_kc=True):
                count = 0
                for prefix in prerequisite_starts:
                    index = next(i for i, cell in enumerate(notebook.cells)
                                 if cell.cell_type == 'code' and cell.source.startswith(prefix))
                    count += 1
                    client.execute_cell(notebook.cells[index], index, execution_count=count)
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == 'code' and cell.get('id', '').startswith('week10-robust-'):
                        count += 1
                        client.execute_cell(cell, index, execution_count=count)
        elif '--plots-only' in sys.argv:
            with client.setup_kernel(cleanup_kc=True):
                index = next(i for i, cell in enumerate(notebook.cells)
                             if cell.cell_type == 'code'
                             and 'Structured TCN and residual Ridge comparison' in cell.source)
                client.execute_cell(notebook.cells[index], index,
                                    execution_count=notebook.cells[index].execution_count)
        else:
            client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, path)
    print('Execution completed; results and notebook cell outputs are saved.', flush=True)


if __name__ == '__main__':
    main()
